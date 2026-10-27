"""
YZ Agent — kasboek + webshop-backend + WhatsApp-bridge in één proces.

Flow: WhatsApp (Baileys-bridge) / webshop -> vaste sjablonen + regex-commando's
      -> kasboek (data/ledger.json, optionele HF Dataset-sync)
      + stemnotities (Groq/Mistral/Gemini/local-whisper) als notitie in het kasboek
      + live dashboard op /admin.

Env vars (Render secrets):
  ACCESS_CODE       required  - toegangscode voor /admin, /wa-qr, /api/*, /wa-chats
  HF_DATASET        optional  - "username/yz-ledger" private dataset voor backup
  HF_TOKEN          optional  - HF write token, alleen nodig voor HF_DATASET-sync
  BASETAO_USER      optional  - server-side basetao-login voor /basetao + /checkout
  BASETAO_PASS      optional  - zie BASETAO_USER
  GROQ_API_KEY      optional  - gratis ASR: whisper-large-v3 (Mistral/Gemini/local als fallback)
  WHISPER_MODEL     optional  - tiny | base (default) | small (lokale faster-whisper)
  WHATSAPP_ENABLED  optional  - "1" = WhatsApp-bridge actief (start via start.sh)
  WA_WELCOME        optional  - welkomstbericht voor nieuwe nummers
  WA_IMG_REPLY      optional  - antwoord bij een foto zonder tekst
  BASE_URL          optional  - publieke basis-URL (fallback = request-headers)
  LEDGER_PATH       optional  - default data/ledger.json
"""
import asyncio
import base64
from datetime import datetime
import io
import json
import mimetypes
import os
import re
import threading
import time

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

# ---------------- config ----------------
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base")
LEDGER_PATH = os.environ.get("LEDGER_PATH", "data/ledger.json")
HF_DATASET = os.environ.get("HF_DATASET", "").strip()
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
ACCESS_CODE = os.environ.get("ACCESS_CODE", "").strip()  # dashboard-gate op publieke Space
BASETAO_COOKIE = os.environ.get("BASETAO_COOKIE", "").strip()  # DevTools cookie voor /basetao sync
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()  # gratis ASR: whisper-large-v3
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "").strip()  # Voxtral ASR (beste quadrant)
MISTRAL_ASR_MODEL = os.environ.get("MISTRAL_ASR_MODEL", "voxtral-small-latest")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()  # gratis ASR via AI Studio
BASE_URL = os.environ.get("BASE_URL", "").strip()  # publieke basis-URL (fallback: request-headers)
WHATSAPP_ENABLED = os.environ.get("WHATSAPP_ENABLED", "0") == "1"  # Baileys-bridge start via start.sh
# Auto-pauze BOTSHARD uit (wens Younes, 03-10): de Render env-var staat (onveranderbaar
# via de API) op 12 en overrulet de code-default, dus hier bewust NIET meer uitlezen.
WA_TAKEOVER_HOURS = 0.0
WA_IMG_REPLY = os.environ.get(
    "WA_IMG_REPLY",
    "Ontvangen 👍 Zet er even tekst bij (wat zoek je, kleur/maat)? Dan pak ik het direct op.").strip()
WA_SESSION_DIR = os.environ.get("WA_SESSION_DIR", os.path.join("data", "whatsapp-session"))

ORDER_STATUSES = ["interesse", "info_gevraagd", "prijs_gegeven", "wacht_op_antwoord",
                  "te_bestellen", "besteld", "onderweg", "binnen", "verpakken", "klaar",
                  "geen_interesse"]
PAYMENT_STATUSES = ["nog_niet_betaald", "deels_betaald", "betaald", "geen_interesse"]

def _now():
    return time.strftime("%Y-%m-%d %H:%M:%S")

def order_add(d, customer, items, price_eur, payment_method="onbekend", basetao_ids=None, note=""):
    orders = d.setdefault("orders", [])
    nxt = 1 + max([o.get("num", 0) for o in orders] or [0])
    o = {"num": nxt, "customer": customer or "?", "items": items or "",
         "price_eur": float(price_eur or 0), "order_status": "te_bestellen",
         "payment_status": "nog_niet_betaald", "payment_method": payment_method,
         "basetao_ids": basetao_ids or [], "created": _now(), "updated": _now(),
         "history": [{"ts": _now(), "event": "order aangemaakt"}]}
    if note:
        o["note"] = note
    orders.append(o)
    return o

def order_update(d, num, **fields):
    for o in d.get("orders", []):
        if o.get("num") == num:
            for k, v in fields.items():
                if v in (None, ""):
                    continue
                if k in ("order_status", "payment_status") and v not in (ORDER_STATUSES + PAYMENT_STATUSES):
                    continue
                if o.get(k) != v:
                    o.setdefault("history", []).append(
                        {"ts": _now(), "field": k, "from": o.get(k), "to": v})
                    o[k] = v
            o["updated"] = _now()
            return o
    return None

_ledlock = threading.Lock()

# ---------------- ledger ----------------
def ledger_load():
    with open(LEDGER_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def ledger_save(d):
    os.makedirs(os.path.dirname(LEDGER_PATH) or ".", exist_ok=True)
    with open(LEDGER_PATH, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)

def hf_sync_up():
    """Mirror the ledger to a private HF Dataset so restarts lose nothing."""
    if not (HF_DATASET and HF_TOKEN):
        return
    try:
        from huggingface_hub import HfApi
        HfApi(token=HF_TOKEN).upload_file(
            path_or_fileobj=LEDGER_PATH, path_in_repo="ledger.json",
            repo_id=HF_DATASET, repo_type="dataset")
    except Exception as e:  # noqa: BLE001
        print("hf sync up failed:", e)

def hf_sync_down():
    """Bij start: de HF-backup is leidend als die bestaat (repo-seed is alleen fallback)."""
    if not (HF_DATASET and HF_TOKEN):
        return
    try:
        import shutil
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(repo_id=HF_DATASET, repo_type="dataset",
                            filename="ledger.json", token=HF_TOKEN)
        os.makedirs(os.path.dirname(LEDGER_PATH) or ".", exist_ok=True)
        shutil.copy(p, LEDGER_PATH)
        print("ledger hersteld uit HF dataset")
    except Exception as e:  # noqa: BLE001
        print("hf sync down failed:", e)

# ---------------- voice-notities bewaren (altijd opnieuw transcribeerbaar) ----------------
def voice_backup_up(audio_bytes, naam):
    """Stemnotitie meesturen naar de HF-dataset (audio/<naam>), zodat de audio
    nooit meer verloren gaat (de notities van 20-09 waren onherstelbaar weg)."""
    if not (HF_DATASET and HF_TOKEN):
        return
    try:
        from huggingface_hub import HfApi
        HfApi(token=HF_TOKEN).upload_file(
            path_or_fileobj=io.BytesIO(audio_bytes),
            path_in_repo="audio/" + naam, repo_id=HF_DATASET, repo_type="dataset")
    except Exception as e:  # noqa: BLE001
        print("voice backup failed:", e)

def voice_note_opslaan(text, source, chat_key=None, file_id=None, audio_bytes=None):
    """Transcript als note in het kasboek zetten + audio naar HF backuppen."""
    e = {"ts": _now(), "type": "note", "note": "🎙️ " + text[:1500], "source": source}
    if file_id:
        e["voice_file_id"] = file_id
    if chat_key:
        e["chat"] = chat_key
    with _ledlock:
        d = ledger_load()
        d.setdefault("entries", []).append(e)
        ledger_save(d)
    threading.Thread(target=hf_sync_up, daemon=True).start()
    if audio_bytes and HF_DATASET and HF_TOKEN:
        threading.Thread(target=voice_backup_up, daemon=True,
                         args=(audio_bytes, "voice_%d_%s.ogg" % (int(time.time()), source))).start()

def compute_stats(d):
    inc_cash = inc_bank = inc_wallet = cost = topup = 0.0
    for e in d["entries"]:
        try:
            a = float(e.get("amount_eur") or 0)
        except (TypeError, ValueError):
            continue
        t = e.get("type")
        if t == "income":
            m = e.get("method")
            if m == "cash":
                inc_cash += a
            elif m == "basetao":
                inc_wallet += a
            else:
                inc_bank += a  # bank/tikkie/ideal ontvangen
        elif t == "cost":
            cost += a
        elif t == "topup":
            topup += a
    income = inc_cash + inc_bank + inc_wallet
    profit = income - cost
    margin = (profit / income * 100.0) if income else 0.0
    cash_pct = (inc_cash / income * 100.0) if income else 0.0
    bank_pct = (inc_bank / income * 100.0) if income else 0.0
    open_orders = [o for o in d.get("orders", []) if o.get("payment_status") != "betaald"]
    te_innen = sum(float(o.get("price_eur") or 0) for o in open_orders)
    return dict(
        inc_cash=round(inc_cash, 2), inc_bank=round(inc_bank, 2),
        inc_wallet=round(inc_wallet, 2),
        income=round(income, 2), cost=round(cost, 2), profit=round(profit, 2),
        topup_total=round(topup, 2), topup_vs_cost=round(topup - cost, 2),
        margin_pct=round(margin, 1), cash_pct=round(cash_pct, 1),
        bank_pct=round(bank_pct, 1), wallet_pct=round(100 - cash_pct - bank_pct, 1),
        gap_pct=round(cash_pct - margin, 1),
        open_orders=len(open_orders), te_innen=round(te_innen, 2))

def monthly_breakdown(d):
    """Per maand: omzet (cash/bank), topups en kosten — de stort-check."""
    rows = {}
    for e in d["entries"]:
        try:
            a = float(e.get("amount_eur") or 0)
        except (TypeError, ValueError):
            continue
        maand = (e.get("ts") or "")[:7]
        if not maand:
            continue
        r = rows.setdefault(maand, dict(cash=0.0, bank=0.0, wallet=0.0, topup=0.0, cost=0.0))
        t = e.get("type")
        if t == "income":
            m = e.get("method")
            if m == "cash":
                r["cash"] += a
            elif m == "basetao":
                r["wallet"] += a
            else:
                r["bank"] += a
        elif t == "cost":
            r["cost"] += a
        elif t == "topup":
            r["topup"] += a
    out = []
    for m in sorted(rows):
        r = rows[m]
        omzet = r["cash"] + r["bank"] + r["wallet"]
        out.append(dict(maand=m, omzet=round(omzet, 2), cash=round(r["cash"], 2),
                        bank=round(r["bank"], 2), topup=round(r["topup"], 2),
                        cost=round(r["cost"], 2), delta=round(r["topup"] - r["cost"], 2)))
    return out

# ---------------- ASR (faster-whisper, local CPU) ----------------
_whisper = None

def transcribe(path):
    # Groq whisper-large-v3 eerst (snel + beste NL-kwaliteit); Mistral/Gemini/local als fallback
    if GROQ_API_KEY:
        with open(path, "rb") as f:
            r = requests.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                files={"file": ("audio.ogg", f, "audio/ogg")},
                data={"model": "whisper-large-v3", "language": "nl",
                      "temperature": "0", "response_format": "json"},
                timeout=120)
        if r.status_code != 200:
            raise RuntimeError(f"groq {r.status_code}: {r.text[:140]}")
        return (r.json().get("text") or "").strip()
    if MISTRAL_API_KEY:
        for lang in ("nl", None):
            with open(path, "rb") as f:
                data = {"model": MISTRAL_ASR_MODEL}
                if lang:
                    data["language"] = lang
                r = requests.post(
                    "https://api.mistral.ai/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {MISTRAL_API_KEY}"},
                    files={"file": ("audio.ogg", f, "audio/ogg")},
                    data=data, timeout=120)
            if r.status_code == 400 and lang:
                continue  # taalparameter niet ondersteund -> opnieuw zonder
            r.raise_for_status()
            return (r.json().get("text") or "").strip()
    if GEMINI_API_KEY:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        model = os.environ.get("GEMINI_ASR_MODEL", "gemini-2.5-flash")
        r = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            headers={"x-goog-api-key": GEMINI_API_KEY},
            json={"contents": [{"parts": [
                {"text": "Transcribe this audio exactly, in the original language (Dutch). "
                         "Output only the transcription text."},
                {"inline_data": {"mime_type": "audio/ogg", "data": b64}}]}]},
            timeout=120)
        r.raise_for_status()
        d = r.json()
        return ((d.get("candidates") or [{}])[0].get("content", {}).get("parts", [{}])[0]
                .get("text") or "").strip()
    global _whisper
    if _whisper is None:
        from faster_whisper import WhisperModel
        print("loading whisper model:", WHISPER_MODEL)
        _whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    segs, _ = _whisper.transcribe(path, language="nl", beam_size=1)
    return " ".join(s.text.strip() for s in segs).strip()

def basetao_snapshot():
    """Basetao-sessie testen + saldo/tellers uit de server-HTML.
    CF-prerendering levert de data willekeurig mee -> tot 4x proberen."""
    headers = {
        "Cookie": BASETAO_COOKIE,
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/152.0.0.0 Safari/537.36",
        "Referer": "https://www.basetao.com/",
        "Accept": "text/html",
    }
    logged_in, balance, counters, http_code = False, None, {}, 0
    for attempt in range(4):
        try:
            r = requests.get(
                "https://www.basetao.com/best-taobao-agent-service/my_account/welcome.html",
                headers=headers, timeout=30,
                params={"r": str(int(time.time() * 1000)) + str(attempt)})
        except Exception as e:  # noqa: BLE001
            print("basetao fetch error:", e)
            break
        http_code = r.status_code
        t = r.text
        logged_in = "Welcome back" in t
        m = re.search(r'bi-currency-yen[\s\S]{0,150}?>\s*([\d.,]+)\s*<', t)
        balance = m.group(1) if m else None
        counters = dict(re.findall(
            r'id="(Ordered|Arrived|Cancelled|Shipped|Searching|Received|Pending)"'
            r'[\s\S]{0,300}?badge[^>]*>\s*(\d+)\s*</span>', t))
        if logged_in and (balance or counters):
            break
        time.sleep(1.2)
    return {"logged_in": logged_in, "http": http_code,
            "balance_cny": balance, "counters": counters,
            "note": "saldo/tellers komen mee wanneer basetao ze in de HTML serveert"}

# ---------------- web dashboard ----------------
DASH = """<!doctype html><html lang="nl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="60">
<title>YZ Shop — Admin</title><style>
body{font-family:system-ui,Segoe UI,sans-serif;background:#0f1716;color:#e8efec;margin:0;padding:24px}
h1{font-size:20px;margin:0 0 4px}.sub{color:#8aa39c;font-size:13px;margin-bottom:20px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:14px;max-width:980px}
.card{background:#182522;border:1px solid #24382f;border-radius:12px;padding:16px}
.k{color:#8aa39c;font-size:12px;text-transform:uppercase;letter-spacing:.06em}
.v{font-size:26px;font-weight:700;margin-top:6px}
.bar{height:14px;border-radius:7px;overflow:hidden;display:flex;margin-top:10px;background:#0c1210}
.bar span{height:100%}.cash{background:#4caf7d}.bank{background:#3d7dd8}.wallet{background:#8a6fd1}
.gap-ok{color:#4caf7d}.gap-bad{color:#e0a13d}
.v.ok{color:#4caf7d}.v.bad{color:#e0a13d}.v.lime{color:#d7ff3f}
h2{font-size:16px;margin:30px 0 0}
.legend{display:flex;gap:14px;margin-top:10px;font-size:12px;color:#c8d6d1;flex-wrap:wrap}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px}
tr.tot td{font-weight:700;border-top:2px solid #2b463c}
.ok{color:#4caf7d}.bad{color:#e0a13d}
table{width:100%;max-width:980px;border-collapse:collapse;margin-top:22px;font-size:13px}
td,th{padding:7px 9px;border-bottom:1px solid #1e2f28;text-align:left}
th{color:#8aa39c;font-weight:600}form{margin-top:26px;max-width:980px;background:#182522;
border:1px solid #24382f;border-radius:12px;padding:16px;display:grid;
grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px}
input,select{background:#0c1210;border:1px solid #2b463c;color:#e8efec;border-radius:8px;padding:8px;width:100%}
button{background:#d7ff3f;border:0;color:#10130f;border-radius:8px;padding:10px;font-weight:700;cursor:pointer}
.note{color:#8aa39c;font-size:12px;margin-top:18px;max-width:980px}
</style></head><body>
<h1>YZ SHOP — cash &amp; bank</h1>
<div class="sub">live · verversen elke 60s · regel: alleen inkoop+verzending storten → alle winst blijft cash</div>
<div class="grid">
<div class="card"><div class="k">Cash ontvangen</div><div class="v ok">€__CASH__</div>
<div class="k" style="margin-top:6px">__CASH_PCT__% van omzet</div></div>
<div class="card"><div class="k">Bank ontvangen (tikkie/ideal)</div><div class="v">€__BANK__</div>
<div class="k" style="margin-top:6px">__BANK_PCT__% van omzet</div></div>
<div class="card"><div class="k">Basetao-inkomsten</div><div class="v">€__WALLET__</div>
<div class="k" style="margin-top:6px">__WALLET_PCT__% van omzet</div></div>
<div class="card"><div class="k">Omzet</div><div class="v lime">€__INCOME__</div>
<div class="k" style="margin-top:6px">winst €__PROFIT__ · marge __MARGIN__%</div></div>
<div class="card"><div class="k">Kosten (inkoop + verzending)</div><div class="v">€__COST__</div>
<div class="k" style="margin-top:6px">gebetaald uit basetao-wallet</div></div>
<div class="card"><div class="k">Open orders</div><div class="v">__OPENORDERS__</div>
<div class="k" style="margin-top:6px">te innen: €__TEINNEN__</div></div>
</div>
<div class="grid" style="margin-top:14px">
<div class="card wide"><div class="k">Cash → bank ratio</div>
<div class="bar" style="height:20px"><span class="cash" style="width:__CASH_PCT__%"></span><span class="bank" style="width:__BANK_PCT__%"></span><span class="wallet" style="width:__WALLET_PCT__%"></span></div>
<div class="legend"><span><span class="dot" style="background:#4caf7d"></span>cash __CASH_PCT__%</span><span><span class="dot" style="background:#3d7dd8"></span>bank __BANK_PCT__%</span><span><span class="dot" style="background:#8a6fd1"></span>basetao __WALLET_PCT__%</span><span>doel: cash% ≥ marge __MARGIN__% · verschil __GAP__ pp __GAPCLS_TXT__</span></div></div>
<div class="card wide"><div class="k">Stort-check — topups vs kosten</div>
<div class="v __TOPUP_CLS__">__TOPUP_TXT__</div>
<div class="k" style="margin-top:6px">€__TOPUP__ gestort via iDEAL vs €__COST__ product+verzending · __STORT_ADVIES__</div></div>
<div class="card"><div class="k">Basetao saldo (live)</div><div class="v">¥__BTBAL__</div>
<div class="k" style="margin-top:6px">__BTCNT__</div></div>
</div>
<h2>📅 Per maand — stort-check</h2>
<table><tr><th>Maand</th><th>Omzet</th><th>Cash</th><th>Bank</th><th>Topups</th><th>Kosten</th><th>Topup − kosten</th></tr>
__MROWS__
</table>
<h2>📋 Orders &amp; betalingen</h2>
<table><tr><th>#</th><th>Klant</th><th>Items</th><th>€</th><th>Orderstatus</th><th>Betaalstatus</th><th>Basetao</th><th>Laatst</th><th></th></tr>
__OROWS__
</table>
<form onsubmit="addOrder(event)">
<input id="o_customer" placeholder="klant">
<input id="o_items" placeholder="items (bv. Ajax setje M)">
<input id="o_price" type="number" step="0.01" placeholder="prijs €">
<button>Nieuwe order</button></form>
<form onsubmit="updOrder(event)">
<input id="u_num" type="number" placeholder="order #">
<input id="u_customer" placeholder="klant (optioneel)">
<select id="u_os"><option value="">— orderstatus —</option>__OSOPT__</select>
<select id="u_ps"><option value="">— betaalstatus —</option>__PSOPT__</select>
<button>Status bijwerken</button></form>
<table><tr><th>Datum</th><th>Type</th><th>€</th><th>Methode</th><th>Klant</th><th>Notitie</th></tr>
__ROWS__
</table>
<form onsubmit="add(event)">
<input id="f_amount" type="number" step="0.01" placeholder="bedrag €">
<select id="f_type"><option value="income">inkomsten</option><option value="cost">kosten</option><option value="topup">topup (storting)</option></select>
<select id="f_method"><option value="cash">cash</option><option value="bank">bank/tikkie</option><option value="basetao">basetao</option></select>
<input id="f_customer" placeholder="klant">
<input id="f_note" placeholder="notitie">
<button>Toevoegen</button></form>
<form onsubmit="addNote(event)">
<input id="f_note2" placeholder="notitie / memo (bv. transcript of afspraak)">
<button>Notitie opslaan</button></form>
<script>const K=new URLSearchParams(location.search).get('key')||(document.cookie.split('; ').find(r=>r.startsWith('key='))||'').slice(4)||'';
if(K)document.cookie='key='+K+';path=/;max-age=31536000';
async function add(e){e.preventDefault();const g=i=>document.getElementById(i).value;
const t=g('f_type');const b={type:t,amount_eur:parseFloat(g('f_amount')),
customer:g('f_customer')||null,note:g('f_note')||null};
b.method=(t==='topup')?'ideal':g('f_method');
const r=await fetch('/api/entry',{method:'POST',headers:{'Content-Type':'application/json','X-Access-Code':K},
body:JSON.stringify(b)});
r.ok?location.reload():alert('mislukt');}
async function addNote(e){e.preventDefault();const n=document.getElementById('f_note2').value.trim();
if(!n)return;const r=await fetch('/api/note',{method:'POST',headers:{'Content-Type':'application/json','X-Access-Code':K},body:JSON.stringify({note:n})});
r.ok?location.reload():alert('mislukt');}
async function postOrder(b){const r=await fetch('/api/order',{method:'POST',
headers:{'Content-Type':'application/json','X-Access-Code':K},body:JSON.stringify(b)});
r.ok?location.reload():alert('mislukt');}
async function addOrder(e){e.preventDefault();const g=i=>document.getElementById(i).value;
postOrder({customer:g('o_customer'),items:g('o_items'),price_eur:parseFloat(g('o_price')||'0')});}
async function updOrder(e){e.preventDefault();const g=i=>document.getElementById(i).value;
const b={num:parseInt(g('u_num'))};if(g('u_os'))b.order_status=g('u_os');if(g('u_ps'))b.payment_status=g('u_ps');
if(g('u_customer'))b.customer=g('u_customer');
if(!g('u_num')||(!g('u_os')&&!g('u_ps')&&!g('u_customer'))){alert('vul order # en minstens één veld in');return;}
postOrder(b);}
async function delOrder(n){if(!confirm('Order #'+n+' verwijderen?'))return;
const r=await fetch('/api/order/'+n,{method:'DELETE',headers:{'X-Access-Code':K}});
r.ok?location.reload():alert('mislukt');}</script>
<div class="note">topup = iDEAL-storting naar je basetao-wallet (geen omzet) · bank = tikkie/overboeking ontvangen · checkout-links worden automatisch als topup geboekt · stemnotities worden bewaard in de HF-dataset (map audio/).</div>
</body></html>"""

def render_dashboard():
    w = basetao_wallet() or {}
    btbal = w.get("balance_cny") or "—"
    c = w.get("counters") or {}
    btc = ("ordered {0} · arrived {1} · shipped {2} · searching {3}".format(
        c.get("Pending", "?"), c.get("Arrived", "?"), c.get("Shipped", "?"),
        c.get("Searching", "?"))) if c else "live via basetao-API (cache 5 min)"
    with _ledlock:
        d = ledger_load()
        s = compute_stats(d)
        maanden = monthly_breakdown(d)
        rows = []
        for e in reversed(d["entries"][-25:]):
            rows.append(
                "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                    e.get("ts", ""), e.get("type", ""),
                    ("€%g" % e["amount_eur"]) if e.get("amount_eur") else "—",
                    e.get("method") or "—", e.get("customer") or "—",
                    (e.get("note") or "")[:80]))
        orows = []
        for o in reversed(d.get("orders", [])):
            bt = " ".join('<a href="https://www.basetao.com/best-taobao-agent-service/'
                          'purchase/order_img/{0}.html" target="_blank" rel="noopener">{0}</a>'
                          .format(i) for i in o.get("basetao_ids", [])) or "—"
            orows.append(
                "<tr><td>#{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}{}</td><td>{}</td><td>{}</td>"
                "<td><button onclick=\"delOrder({})\" style=\"background:#7d3a3a;border:0;color:#fff;"
                "border-radius:6px;cursor:pointer;padding:2px 7px\">✖</button></td></tr>".format(
                    o.get("num", ""), o.get("customer", ""), o.get("items", ""),
                    ("€%g" % o["price_eur"]) if o.get("price_eur") else "—",
                    o.get("order_status", ""),
                    o.get("payment_status", ""),
                    (" (" + o["payment_method"] + ")") if o.get("payment_method") not in (None, "onbekend") else "",
                    bt, o.get("updated", ""), o.get("num", "")))
    os_opts = "".join(f'<option value="{x}">{x}</option>' for x in ORDER_STATUSES)
    ps_opts = "".join(f'<option value="{x}">{x}</option>' for x in PAYMENT_STATUSES)
    gap_txt = "✅" if abs(s["gap_pct"]) < 5 else "⚠️"
    if s["topup_vs_cost"] >= 0:
        topup_txt = f"✅ gedekt (+€{s['topup_vs_cost']:g})"
        topup_cls = "ok"
        advies = "wallet heeft voorraad — niks bijstorten nodig"
    else:
        topup_txt = f"⚠️ te kort (−€{abs(s['topup_vs_cost']):g})"
        topup_cls = "bad"
        advies = f"stort nog €{abs(s['topup_vs_cost']):g}, dan blijft al je cash winst"
    mrows = []
    if maanden:
        t_om = t_c = t_b = t_t = t_k = 0.0
        for r in maanden:
            dcls = "ok" if r["delta"] >= 0 else "bad"
            ds = f"+€{r['delta']:g}" if r["delta"] >= 0 else f"−€{abs(r['delta']):g}"
            mrows.append(f"<tr><td>{r['maand']}</td><td>€{r['omzet']:g}</td><td>€{r['cash']:g}</td>"
                         f"<td>€{r['bank']:g}</td><td>€{r['topup']:g}</td><td>€{r['cost']:g}</td>"
                         f"<td class='{dcls}'>{ds}</td></tr>")
            t_om += r["omzet"]
            t_c += r["cash"]
            t_b += r["bank"]
            t_t += r["topup"]
            t_k += r["cost"]
        t_delta = round(t_t - t_k, 2)
        tds = f"+€{t_delta:g}" if t_delta >= 0 else f"−€{abs(t_delta):g}"
        mrows.append(f"<tr class='tot'><td>totaal</td><td>€{round(t_om, 2):g}</td>"
                     f"<td>€{round(t_c, 2):g}</td><td>€{round(t_b, 2):g}</td>"
                     f"<td>€{round(t_t, 2):g}</td><td>€{round(t_k, 2):g}</td><td>{tds}</td></tr>")
    return (DASH
            .replace("__CASH__", f"{s['inc_cash']:g}")
            .replace("__BANK__", f"{s['inc_bank']:g}")
            .replace("__WALLET__", f"{s['inc_wallet']:g}")
            .replace("__CASH_PCT__", str(s["cash_pct"]))
            .replace("__BANK_PCT__", str(s["bank_pct"]))
            .replace("__WALLET_PCT__", str(s["wallet_pct"]))
            .replace("__INCOME__", f"{s['income']:g}")
            .replace("__PROFIT__", f"{s['profit']:g}")
            .replace("__MARGIN__", str(s["margin_pct"]))
            .replace("__COST__", f"{s['cost']:g}")
            .replace("__GAP__", str(s["gap_pct"]))
            .replace("__GAPCLS_TXT__", gap_txt)
            .replace("__TOPUP__", f"{s['topup_total']:g}")
            .replace("__TOPUP_TXT__", topup_txt)
            .replace("__TOPUP_CLS__", topup_cls)
            .replace("__STORT_ADVIES__", advies)
            .replace("__OPENORDERS__", str(s["open_orders"]))
            .replace("__TEINNEN__", f"{s['te_innen']:g}")
            .replace("__BTBAL__", str(btbal))
            .replace("__BTCNT__", btc)
            .replace("__MROWS__", "\n".join(mrows) or '<tr><td colspan="7">— nog geen bedragen —</td></tr>')
            .replace("__OROWS__", "\n".join(orows) or '<tr><td colspan="9">— nog geen orders —</td></tr>')
            .replace("__OSOPT__", os_opts)
            .replace("__PSOPT__", ps_opts)
            .replace("__ROWS__", "\n".join(rows) or "<tr><td colspan=6>—</td></tr>"))

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

@app.middleware("http")
async def access_gate(request: Request, call_next):
    """Site is openbaar; alleen beheer-routes vereisen de toegangscode."""
    if request.method == "OPTIONS":  # CORS-voorvraag: altijd doorlaten
        return await call_next(request)
    if request.url.path == "/api/aanvragen":  # publieke foto-intake (YZ SHOP): geen key
        return await call_next(request)
    path = request.url.path
    gated = (path.startswith("/admin") or path.startswith("/api")
             or path.startswith("/stats") or path.startswith("/orders")
             or path.startswith("/basetao") or path.startswith("/docs")
             or path.startswith("/openapi") or path.startswith("/whatsapp")
             or path.startswith("/shop-dashboard"))
    if ACCESS_CODE and gated:
        code = (request.query_params.get("key")
                or request.headers.get("x-access-code")
                or request.cookies.get("key"))
        if code != ACCESS_CODE:
            return JSONResponse({"error": "toegang geweigerd: voeg ?key=... toe"},
                                status_code=401)
    return await call_next(request)

@app.on_event("startup")
def _start():
    hf_sync_down()
    wa_backup_down()
    if WA_TAKEOVER_HOURS <= 0:
        # auto-pauze staat uit: ruim oude pauzes op zodat chats direct weer werken
        try:
            with _ledlock:
                d = ledger_load()
                paused = d.get("whatsapp", {}).get("paused", {})
                if paused:
                    paused.clear()
                    ledger_save(d)
                    print("oude whatsapp-pauzes gewist (takeover uit)")
        except Exception as e:  # noqa: BLE001
            print("pauzes wissen mislukt:", e)
    try:
        shop_startup()  # webshop: wallet-controle-thread (betaald-detectie, best effort)
    except Exception as e:  # noqa: BLE001
        print("shop startup mislukt:", e)

    def _herinneringen():
        # Elke 4 uur: wat moet Younes nog doen? Orders die te bestellen zijn en
        # orders die "besteld" zijn maar waar geen Basetao-bevestiging aan hangt
        # ("ik ga het bestellen" gezegd, maar nooit echt gedaan — jij vergeet het
        # vaak, dus de bot blijft vragen tot het klopt).
        time.sleep(300)  # pas nadat alles opgestart is
        while True:
            try:
                regels = []
                with _ledlock:
                    d = ledger_load()
                    for o in d.get("orders", []):
                        st = o.get("order_status")
                        naam = o.get("customer") or "?"
                        items = o.get("items") or "?"
                        prijs = float(o.get("price_eur") or 0)
                        if st == "te_bestellen":
                            regels.append(f"❗ NOG BESTELLEN: #{o['num']} {naam} — {items} (€{prijs:g})")
                        elif st == "besteld" and not o.get("basetao_ids"):
                            ts = str(o.get("updated") or o.get("created") or "")
                            try:
                                oud = time.time() - time.mktime(time.strptime(ts, "%Y-%m-%d %H:%M:%S"))
                            except Exception:  # noqa: BLE001
                                oud = 999999
                            if oud > 12 * 3600:
                                regels.append(f"❓ #{o['num']} {naam} — echt al besteld in Basetao? "
                                              f"Ik zie nog geen bevestiging ({items}).")
                if regels:
                    _stuur_via_bridge(WA_SELF_JID,
                                      "⏰ Even jou ding, vergeet deze niet:\n" + "\n".join(regels[:8]))
                # Follow-up kansen: prijs gegeven, klant stil, nog geen order
                try:
                    with _ledlock:
                        d = ledger_load()
                        chats = d.get("whatsapp", {}).get("chats", {})
                        orders = d.get("orders", [])
                        volg = []
                        for num, c in chats.items():
                            if not c.get("prijs_gegeven") or c.get("status") in ("besteld", "afgerond"):
                                continue
                            laat = (c.get("laatste") or {})
                            if laat.get("van") != "klant":
                                continue
                            try:
                                oud_u = (time.time() - time.mktime(
                                    time.strptime(str(laat.get("ts")), "%Y-%m-%d %H:%M:%S"))) / 3600
                            except Exception:  # noqa: BLE001
                                continue
                            if oud_u >= 24:
                                known = (c.get("klantnaam") or c.get("naam") or "+" + num).lower()
                                besteld = any(known in str(o.get("customer", "")).lower()
                                              for o in orders)
                                if not besteld:
                                    volg.append(f"📌 Volg op: {known} — prijs gegeven, "
                                                f"stil sinds {int(oud_u)}u (gezocht: {c.get('gezocht') or '?'})")
                        if volg:
                            _stuur_via_bridge(WA_SELF_JID,
                                              "🔔 Kansen om binnen te halen:\n" + "\n".join(volg[:6]))
                except Exception as e:  # noqa: BLE001
                    print("follow-up check faalden:", e)
            except Exception as e:  # noqa: BLE001
                print("herinneringen faalden:", e)
            time.sleep(4 * 3600)

    threading.Thread(target=_herinneringen, daemon=True).start()

    def _wa_periodic_backup():
        # ververs de sessie-backup elke 30 min zolang we verbonden zijn,
        # zodat een herstart nooit een verouderde (Bad MAC) sessie terugzet
        while True:
            time.sleep(1800)
            try:
                if WA_STATE.get("status") == "connected":
                    wa_backup_up()
            except Exception as e:  # noqa: BLE001
                print("wa periodic backup failed:", e)

    threading.Thread(target=_wa_periodic_backup, daemon=True).start()

SHOP = """<!doctype html><html lang="nl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>YZ Shop — premium reps</title><style>
body{font-family:system-ui,Segoe UI,sans-serif;background:#0b0f0e;color:#eef2f0;margin:0;padding:0 0 60px}
header{padding:34px 24px 10px;max-width:1080px;margin:0 auto}
.brand{font-size:30px;font-weight:800;letter-spacing:.02em}
.brand span{color:#f7941d}
.sub{color:#8fa39b;margin-top:6px;font-size:14px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:16px;
max-width:1080px;margin:26px auto 0;padding:0 24px}
.pcard{background:#141c1a;border:1px solid #22332c;border-radius:14px;padding:18px;
display:flex;flex-direction:column;gap:10px}
.ptag{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:#f7941d}
.pcard h3{margin:0;font-size:16px;line-height:1.35;font-weight:650}
.pprice{font-size:20px;font-weight:800}
.pnote{color:#8fa39b;font-size:12px;margin-top:-4px}
.pbtn{margin-top:auto;background:#f7941d;color:#10130f;text-align:center;font-weight:800;
text-decoration:none;border-radius:10px;padding:11px;font-size:14px}
.pbtn:hover{filter:brightness(1.08)}
.empty{grid-column:1/-1;color:#8fa39b;background:#141c1a;border:1px dashed #22332c;
border-radius:14px;padding:30px;text-align:center}
footer{max-width:1080px;margin:44px auto 0;padding:0 24px;color:#5f7268;font-size:12px;
border-top:1px solid #1a2620;padding-top:18px}
footer a{color:#f7941d}
.wa{position:fixed;right:20px;bottom:20px;background:#25d366;color:#fff;font-weight:800;
border-radius:999px;padding:13px 20px;text-decoration:none;box-shadow:0 6px 24px rgba(0,0,0,.45)}
</style></head><body>
<header>
<div class="brand">YZ<span> SHOP</span></div>
<div class="sub">premium reps · handgepickte drops · levering 2–3 weken · betaling bij ontvangst</div>
</header>
<div class="grid">
__CARDS__
</div>
<footer>YZ Shop · bestellen en betalen regelen we persoonlijk via
<a href="https://wa.me/31684805378">WhatsApp</a> · leverage levering NL · geen voorraad = op aanvraag</footer>
<a class="wa" href="https://wa.me/31684805378">Bestel via WhatsApp</a>
</body></html>"""

_STATUS_NL = {"besteld": "in bestelling", "onderweg": "onderweg naar NL",
              "binnen": "op voorraad", "verpakken": "klaar gemaakt",
              "klaar": "direct leverbaar", "te_bestellen": "bestel ik zo",
              "interesse": "op aanvraag", "prijs_gegeven": "op aanvraag"}

def render_shop():
    with _ledlock:
        d = ledger_load()
    items = [o for o in d.get("orders", [])
             if (o.get("customer") or "").strip().lower() in ("onbekend", "", "?")]
    cards = []
    for o in reversed(items[-24:]):
        price = ("€%g" % o["price_eur"]) if o.get("price_eur") else "Prijs in DM"
        st = _STATUS_NL.get(o.get("order_status", ""), o.get("order_status", ""))
        bt = ""
        if o.get("basetao_ids"):
            bt = ('<div class="pnote">QC: <a style="color:#f7941d" target="_blank" rel="noopener" '
                  'href="https://www.basetao.com/best-taobao-agent-service/purchase/order_img/%s.html">'
                  "foto's</a></div>" % o["basetao_ids"][0])
        cards.append(
            '<div class="pcard"><div class="ptag">%s</div><h3>%s</h3>'
            '<div class="pprice">%s</div>%s%s'
            '<a class="pbtn" href="https://wa.me/31684805378">Bestel via WhatsApp</a></div>'
            % (st, o.get("items") or "Custom item", price, "", bt))
    if not cards:
        cards.append('<div class="empty">Nieuwe drop onderweg — DM voor de huidige voorraad 📦</div>')
    return (SHOP.replace("__CARDS__", "\n".join(cards))
                .replace("__YEAR__", time.strftime("%Y")))

SITE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "static", "index.html")

@app.get("/admin", response_class=HTMLResponse)
def admin():
    return render_dashboard()

@app.get("/healthz")
def healthz():
    return {"ok": True,
            "whatsapp": WA_STATE["status"] if WHATSAPP_ENABLED else "uit"}

@app.get("/stats")
def stats():
    with _ledlock:
        d = ledger_load()
    return JSONResponse({"stats": compute_stats(d),
                         "maanden": monthly_breakdown(d),
                         "recent": list(reversed(d["entries"][-25:]))})

@app.post("/api/entry")
async def api_entry(req: Request):
    b = await req.json()
    t, amt = b.get("type"), b.get("amount_eur")
    if t not in ("income", "cost", "topup") or not amt:
        return JSONResponse({"ok": False, "error": "type of bedrag ongeldig"}, status_code=400)
    standaard = {"income": "cash", "cost": "basetao", "topup": "ideal"}[t]
    with _ledlock:
        d = ledger_load()
        d["entries"].append({
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "type": t,
            "amount_eur": float(amt),
            "method": b.get("method") or standaard,
            "customer": b.get("customer"), "items": None,
            "note": b.get("note") or "website", "source": "website"})
        ledger_save(d)
        s = compute_stats(d)
        hf_sync_up()
    return JSONResponse({"ok": True, "stats": s})

@app.post("/api/note")
async def api_note(req: Request):
    """Vrije notitie in het kasboek (bijv. herstelde transcripts, memo's)."""
    try:
        b = await req.json()
    except Exception:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": "ongeldige json"}, status_code=400)
    note = (b.get("note") or "").strip()
    if not note:
        return JSONResponse({"ok": False, "error": "geen notitie"}, status_code=400)
    with _ledlock:
        d = ledger_load()
        d["entries"].append({"ts": _now(), "type": "note", "note": note[:2000],
                             "source": b.get("source") or "website"})
        ledger_save(d)
        hf_sync_up()
    return JSONResponse({"ok": True})

@app.get("/orders")
def orders():
    with _ledlock:
        d = ledger_load()
    return JSONResponse({"orders": d.get("orders", [])})

_wallet_cache = {"ts": 0.0, "data": None}

def basetao_wallet(max_age=300):
    """Live basetao-saldo + tellers, max 1x per 5 min daadwerkelijk opgehaald."""
    if _wallet_cache["data"] and time.time() - _wallet_cache["ts"] < max_age:
        return _wallet_cache["data"]
    if BASETAO_COOKIE:
        try:
            _wallet_cache["data"] = basetao_snapshot()
            _wallet_cache["ts"] = time.time()
        except Exception as e:  # noqa: BLE001
            print("basetao wallet error:", e)
    return _wallet_cache["data"]

@app.get("/basetao")
def basetao_route():
    if not BASETAO_COOKIE:
        return JSONResponse({"ok": False, "error": "BASETAO_COOKIE niet ingesteld"}, status_code=400)
    snap = basetao_snapshot()
    return JSONResponse({"ok": bool(snap and snap.get("logged_in")), **(snap or {})})

def basetao_fetch(path):
    headers = {
        "Cookie": BASETAO_COOKIE,
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/152.0.0.0 Safari/537.36",
        "Referer": "https://www.basetao.com/best-taobao-agent-service/my_account/welcome.html",
        "Accept": "text/html",
    }
    r = requests.get("https://www.basetao.com/best-taobao-agent-service/my_account/" + path,
                     headers=headers, timeout=40)
    return r.text if r.status_code == 200 else ""

@app.get("/basetao/rows")
def basetao_rows():
    """Debug: structuur van de orderlijst-pagina's zoals de server ze ziet."""
    out = {}
    for name, page in (("ordered", "order/ordered.html"), ("arrived", "order/arrived.html")):
        t = basetao_fetch(page)
        regions = []
        for m in list(re.finditer(r'order_img', t))[:4]:
            regions.append(t[max(0, m.start() - 700):m.start() + 400])
        out[name] = {"len": len(t), "has_login": "Login" in t[:3000],
                     "n_order_img": t.count("order_img"), "regions": regions}
    return JSONResponse(out)

@app.post("/api/basetao")
async def api_basetao(req: Request):
    """Browser-bridge: de ingelogde basetao-tab post hier haar dashboard-data."""
    try:
        b = await req.json()
    except Exception:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": "ongeldige json"}, status_code=400)
    with _ledlock:
        d = ledger_load()
        d.setdefault("entries", []).append({
            "ts": _now(), "type": "note",
            "note": "basetao-sync: " + json.dumps(b, ensure_ascii=False)[:900],
            "source": "basetao-bridge"})
        ledger_save(d)
        hf_sync_up()
    return JSONResponse({"ok": True})

@app.post("/api/order")
async def api_order(req: Request):
    b = await req.json()
    with _ledlock:
        d = ledger_load()
        if b.get("num"):
            o = order_update(d, int(b["num"]), order_status=b.get("order_status"),
                             payment_status=b.get("payment_status"),
                             payment_method=b.get("payment_method"),
                             customer=b.get("customer"),
                             items=b.get("items"),
                             price_eur=b.get("price_eur"))
            if not o:
                return JSONResponse({"ok": False, "error": "order niet gevonden"}, status_code=404)
        else:
            if not b.get("customer"):
                return JSONResponse({"ok": False, "error": "klant ontbreekt"}, status_code=400)
            o = order_add(d, b.get("customer"), b.get("items"), b.get("price_eur") or 0,
                          payment_method=b.get("payment_method") or "onbekend",
                          basetao_ids=b.get("basetao_ids") or [], note="website")
        ledger_save(d)
        hf_sync_up()
    return JSONResponse({"ok": True, "order": {k: o.get(k) for k in
                        ("num", "customer", "items", "price_eur", "order_status", "payment_status")}})

@app.delete("/api/order/{num}")
async def api_order_delete(num: int):
    with _ledlock:
        d = ledger_load()
        before = len(d.get("orders", []))
        d["orders"] = [o for o in d.get("orders", []) if o.get("num") != num]
        if len(d["orders"]) == before:
            return JSONResponse({"ok": False, "error": "niet gevonden"}, status_code=404)
        ledger_save(d)
        hf_sync_up()
    return JSONResponse({"ok": True})

# ---------------- webshop-backend (echte orders + iDEAL + dashboard) ----------------
# Router van shop_backend.py: publiek /shop/* (intent, checkout-formulier, order, pay,
# statuspagina) + admin /api/shop/* (dashboard-data, acties, CSV-export). Orders landen
# in dezelfde ledger-orders-lijst als de bot-agent gebruikt (source="webshop").
from shop_backend import router as shop_router, shop_startup  # noqa: E402
app.include_router(shop_router)

# ---------------- whatsapp (Baileys-bridge) ----------------
# De Node-bridge (whatsapp-bridge/bridge.js) koppelt het bestaande WhatsApp-nummer
# als "gekoppeld apparaat" (zoals WhatsApp Web) en post hier inkomende berichten.
# Typ Younes zelf iets in een chat (vanaf zijn telefoon), dan pauzeert de bot voor
# die chat zodat hij het gesprek handmatig kan overnemen. "/bot uit" = 7 dagen,
# "/bot aan" = weer inschakelen. Pauzes leven mee in ledger.json (overleeft restarts).
WA_STATE = {"qr": None, "qr_ts": 0.0, "status": "startend" if WHATSAPP_ENABLED else "uit",
            "error": "", "backup_ts": 0.0, "pair_code": None}

def _wa_key(jid):
    return str(jid or "").split("@")[0].split(":")[0]

def _wa_pause_map(d):
    paused = d.setdefault("whatsapp", {}).setdefault("paused", {})
    now = time.time()
    for k in [k for k, v in paused.items() if float(v or 0) < now]:
        paused.pop(k, None)
    return paused

def wa_is_paused(jid):
    with _ledlock:
        d = ledger_load()
    return _wa_key(jid) in _wa_pause_map(d)

def wa_pause(jid, hours=None, clear=False):
    with _ledlock:
        d = ledger_load()
        paused = _wa_pause_map(d)
        k = _wa_key(jid)
        if clear:
            paused.pop(k, None)
        else:
            paused[k] = time.time() + (hours if hours is not None else WA_TAKEOVER_HOURS) * 3600.0
        ledger_save(d)
        hf_sync_up()

def wa_backup_up():
    """Baileys-sessie meesturen naar de HF-dataset, zodat een redeploy geen
    nieuwe QR-scan vereist (sessie wordt bij het opstarten teruggezet)."""
    if not (HF_DATASET and HF_TOKEN):
        return
    try:
        import zipfile
        from huggingface_hub import HfApi
        zpath = os.path.join(os.environ.get("TMPDIR", "/tmp"), "wa-session.zip")
        if os.path.exists(zpath):
            os.remove(zpath)
        os.makedirs(WA_SESSION_DIR, exist_ok=True)
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for root, _dirs, files in os.walk(WA_SESSION_DIR):
                for fn in files:
                    full = os.path.join(root, fn)
                    z.write(full, os.path.relpath(full, WA_SESSION_DIR))
        HfApi(token=HF_TOKEN).upload_file(
            path_or_fileobj=zpath, path_in_repo="wa-session.zip",
            repo_id=HF_DATASET, repo_type="dataset")
        print("wa-sessie gebackupt naar HF dataset")
    except Exception as e:  # noqa: BLE001
        print("wa backup up failed:", e)

def wa_backup_down():
    """Bij start: eerder gegpairde sessie uit de HF-dataset terugzetten (als de
    lokale sessie weg is, bijv. na een redeploy op een verse container)."""
    if not (HF_DATASET and HF_TOKEN):
        return
    if os.path.isdir(WA_SESSION_DIR) and os.listdir(WA_SESSION_DIR):
        return  # lokale sessie is leidend
    try:
        import zipfile
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(repo_id=HF_DATASET, repo_type="dataset",
                            filename="wa-session.zip", token=HF_TOKEN)
        os.makedirs(WA_SESSION_DIR, exist_ok=True)
        with zipfile.ZipFile(p) as z:
            namen = z.namelist()
            z.extractall(WA_SESSION_DIR)
        print(f"wa-sessie hersteld uit HF dataset: {len(namen)} bestanden "
              f"(creds.json aanwezig: {'creds.json' in namen})")
    except Exception as e:  # noqa: BLE001
        print("wa backup down failed:", e)

WA_SELF_JID = "31684805378@s.whatsapp.net"  # 'bericht jezelf'-chat: notitie-bevestigingen landen hier

def _stuur_via_bridge(to, tekst):
    """Bericht direct via de bridge versturen (bericht-jezelf of een klant-chat)."""
    try:
        r = requests.post("http://127.0.0.1:7861/send",
                          json={"to": to, "text": tekst}, timeout=25)
        return r.ok
    except Exception as e:  # noqa: BLE001
        print("stuur via bridge faalde:", e)
        return False

WA_WELCOME = """Yo, welkom bij YZ Shop

Stuur eerst je voornaam, zodat ik weet met wie ik app — WhatsApp toont soms alleen een nummer. Stuur daarna je Snapchatnaam, plus een foto of link van wat je zoekt en je maat.

Voetbalshirt met bedrukking: €30
Met broekje: €40

Stuur voor de bedrukking ook de naam en het rugnummer door. Ik app je zo terug!"""

# Vaste sjabloon-antwoorden voor klantberichten (de LLM-agent is eruit):
# eerste keyword-match wint, max 1 antwoord per bericht, geen match = stil.
WA_TEMPLATES = [
    (("prijs", "prijzen", "hoeveel", "kosten"),
     "Prijzen: shirt €30 · set (shirt+broek) €40 · sneakers vanaf €100. "
     "Stuur een foto/link van wat je wilt + je maat, dan check ik het 👍"),
    (("levertijd", "hoelang", "wanneer", "klaar"),
     "Levertijd is meestal 2-3 weken. Ik laat het meteen weten zodra je bestelling binnen is!"),
    (("maat", "maten", "welke size"),
     "Geef bij kleding je lengte + gewicht, bij schoenen je maat. Dan adviseer ik de juiste size 👍"),
]

@app.post("/whatsapp/incoming")
async def wa_incoming(req: Request):
    """Inkomend WhatsApp-bericht van de bridge -> antwoord (of None = zwijgen)."""
    try:
        b = await req.json()
    except Exception:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": "ongeldige json"}, status_code=400)
    jid = str(b.get("chat") or "")
    if not jid:
        return JSONResponse({"ok": False, "error": "chat ontbreekt"}, status_code=400)
    key = _wa_key(jid)
    text = (b.get("text") or "").strip()
    mtype = b.get("type") or "text"

    # Laatste bericht + gespreksgeschiedenis per chat vastleggen (staat-van-zaken
    # + de bot kan teruglezen wat de klant al gezegd heeft, ook na herstart)
    try:
        with _ledlock:
            d = ledger_load()
            c = d.setdefault("whatsapp", {}).setdefault("chats", {}).setdefault(key, {})
            if b.get("name"):
                c.setdefault("naam", str(b["name"])[:80])
            if jid:
                c["jid"] = jid  # volledige adres (kan een LID zijn) voor direct sturen
            c["laatste"] = {"ts": _now(), "van": "jij" if b.get("from_me") else "klant",
                            "tekst": (text[:200] or f"[{mtype}]")}
            if text:
                g = c.setdefault("gesprek", [])
                g.append({"r": "j" if b.get("from_me") else "k", "t": text[:200]})
                del g[:-30]
            ledger_save(d)
    except Exception:  # noqa: BLE001
        pass

    if b.get("from_me"):
        low = text.lower()
        if low in ("/bot aan", "bot aan"):
            wa_pause(jid, clear=True)
            return JSONResponse({"reply": "🤖 Bot weer AAN in deze chat.", "to_me": True})
        if low in ("/bot uit", "bot uit"):
            wa_pause(jid, hours=24 * 7)
            return JSONResponse({"reply": "🤖 Bot UIT in deze chat (7 dagen). "
                                          "Typ 'bot aan' om weer in te schakelen.", "to_me": True})
        # Prijslijst vullen: "prijs alo runner 155" -> staat voortaan in vaste_prijzen
        if low.startswith("prijs "):
            rest = text[len("prijs "):].strip()
            m = re.search(r"(\d+(?:[.,]\d{1,2})?)", rest)
            if not m:
                return JSONResponse({"reply": "📝 Format: prijs <product> <bedrag> "
                                              "(bijv: prijs alo runner 155)",
                                     "to_me": True, "to_chat": WA_SELF_JID})
            bedrag = float(m.group(1).replace(",", "."))
            prod = (rest[:m.start()] + " " + rest[m.end():]).strip().lower()[:60] or "onbekend"
            with _ledlock:
                d = ledger_load()
                d.setdefault("whatsapp", {}).setdefault("vaste_prijzen", {})[prod] = bedrag
                ledger_save(d)
                hf_sync_up()
            return JSONResponse({"reply": f"✅ Staat in je prijslijst: {prod} = €{bedrag:g}",
                                 "to_me": True, "to_chat": WA_SELF_JID})
        # Openstaande klantvragen: "open vragen" -> lijst met nummers
        if low in ("open vragen", "vragen", "openstaande vragen"):
            with _ledlock:
                d = ledger_load()
            pend = [p for p in d.get("whatsapp", {}).get("pending", []) if not p.get("answered")]
            if not pend:
                txt = "Geen open vragen 👍"
            else:
                txt = "❓ Open vragen:\n" + "\n".join(
                    f"#{p['n']} {p['klant']}: {p['vraag']} [{str(p['ts'])[11:16]}]"
                    for p in pend[-10:])
            return JSONResponse({"reply": txt, "to_me": True, "to_chat": WA_SELF_JID})
        # Jouw antwoord op een doorgestuurde klantvraag: "antwoord 2 kost 130, stuur ik vanavond"
        if low.startswith("antwoord"):
            rest = text[len("antwoord"):].strip()
            mm = re.match(r"(\d+)\s+(.+)", rest, re.S)
            if not mm:
                return JSONResponse({"reply": "📝 Format: antwoord <nummer> <je antwoord> "
                                              "(nummers: stuur 'open vragen')",
                                     "to_me": True, "to_chat": WA_SELF_JID})
            n, antw = int(mm.group(1)), mm.group(2).strip()[:600]
            with _ledlock:
                d = ledger_load()
                pend = d.get("whatsapp", {}).get("pending", [])
                hit = next((p for p in reversed(pend)
                            if p.get("n") == n and not p.get("answered")), None)
                if not hit:
                    return JSONResponse({"reply": f"Vraag #{n} niet gevonden (al beantwoord?).",
                                         "to_me": True, "to_chat": WA_SELF_JID})
                hit["answered"] = True
                hit["antwoord"] = antw
                mp = re.search(r"€\s?(\d+(?:[.,]\d{1,2})?)", antw)
                bedrag = float(mp.group(1).replace(",", ".")) if mp else None
                # prijs-geheugen: prijzen die je een keer geeft, weet de bot voortaan zelf
                if bedrag and re.search(r"prijs|kosten|hoeveel", hit.get("vraag", ""), re.I):
                    vm = re.search(r"(?:prijs|kosten)\s*(?:van)?\s*[:\-]?\s*(.+)",
                                   hit.get("vraag", ""), re.I)
                    prod = (vm.group(1) if vm else hit.get("vraag", ""))[:60].strip(" ?.")
                    d.setdefault("whatsapp", {}).setdefault("prijzen", {})[prod.lower()] = bedrag
                c = d.setdefault("whatsapp", {}).setdefault("chats", {}).setdefault(
                    hit["chat_key"][3:], {})
                if bedrag:
                    c["prijs_afgesproken"] = bedrag
                    c["prijs_gegeven"] = True
                target_jid = c.get("jid") or (hit["chat_key"][3:] + "@s.whatsapp.net")
                ledger_save(d)
                hf_sync_up()
            threading.Thread(target=_stuur_via_bridge,
                             args=(target_jid, antw),
                             daemon=True).start()
            return JSONResponse({"reply": f"✅ Doorgestuurd naar {hit['klant']}: {antw[:80]}",
                                 "to_me": True, "to_chat": WA_SELF_JID})
        # Dagoverzicht: "stand van vandaag" / "overzicht" -> per klant wat hij wil
        # en wat er nog mist, plus vandaag genoteerde bestellingen.
        if any(kw in low for kw in ("stand van", "overzicht", "wat is de stand", "dagoverzicht")):
            with _ledlock:
                d = ledger_load()
            chats = d.get("whatsapp", {}).get("chats", {})
            regels = []
            for num, c in sorted(chats.items(),
                                 key=lambda kv: str((kv[1].get("laatste") or {}).get("ts") or ""),
                                 reverse=True):
                naam = c.get("klantnaam") or c.get("naam") or ("+" + num)
                deel = "• " + naam
                if c.get("gezocht"):
                    deel += " — " + c["gezocht"]
                if c.get("ontbreekt") and c["ontbreekt"].lower() not in ("leeg", "niets", "niks", "-"):
                    deel += " (mist: " + c["ontbreekt"] + ")"
                laat = c.get("laatste") or {}
                if laat.get("ts"):
                    deel += " [" + str(laat["ts"])[11:16] + "]"
                regels.append(deel)
            vandaag = _now()[:10]
            bestellingen = [o for o in d.get("orders", [])
                            if str(o.get("created", "")).startswith(vandaag)]
            if bestellingen:
                regels.append("Vandaag besteld: " + ", ".join(
                    f"#{o['num']} {o['customer']} ({o.get('items') or '?'})"
                    for o in bestellingen[-8:]))
            for p in [p for p in d.get("whatsapp", {}).get("pending", [])
                      if not p.get("answered")][-8:]:
                regels.append(f"❓ {p['klant']}: {p['vraag']} → antwoord {p['n']} <antwoord>")
            if not regels:
                txt = "📋 Nog niks voor vandaag — geen open klant-chats."
            else:
                txt = "📋 Stand van vandaag:\n" + "\n".join(regels[:18])
            return JSONResponse({"reply": txt, "to_me": True, "to_chat": WA_SELF_JID})
        # Jouw eigen notities vanaf je telefoon: "besteld <klant> <wat> (€bedrag)" of
        # "betaald <klant> (€bedrag)" — zo weet de bot wat je al hebt besteld/ontvangen.
        if low.startswith("besteld"):
            rest = text[len("besteld"):].strip()
            delen = rest.split()
            if not delen:
                return JSONResponse({"reply": "📝 Format: besteld <klant> <wat> (€bedrag)", "to_me": True})
            naam = delen[0]
            itemstxt = " ".join(delen[1:]) or "?"
            m = re.search(r"(?:€|eur\s?)\s?(\d+(?:[.,]\d{1,2})?)", itemstxt, re.I)
            prijs = float(m.group(1).replace(",", ".")) if m else 0
            if m:
                itemstxt = (itemstxt[:m.start()] + " " + itemstxt[m.end():]).strip() or "?"
            with _ledlock:
                d = ledger_load()
                o = order_add(d, naam, itemstxt, prijs, note="notitie in WhatsApp")
                order_update(d, o["num"], order_status="besteld")
                ledger_save(d)
                hf_sync_up()
            out = (f"order #{o['num']} aangemaakt: {o['customer']} — {o['items'] or '?'} "
                   f"€{o['price_eur']:g} (besteld, nog niet betaald)")
            # bevestiging naar je eigen 'bericht jezelf'-chat, niet in de klant z'n chat
            return JSONResponse({"reply": "📝 " + out, "to_me": True, "to_chat": WA_SELF_JID})
        if low.startswith("betaald"):
            rest = text[len("betaald"):].strip()
            naam = rest.split()[0] if rest.split() else ""
            m = re.search(r"(?:€|eur\s?)\s?(\d+(?:[.,]\d{1,2})?)", rest, re.I)
            with _ledlock:
                d = ledger_load()
                hit = None
                for o in reversed(d.get("orders", [])):
                    if naam and naam.lower() in str(o.get("customer", "")).lower() \
                            and o.get("payment_status") != "betaald":
                        hit = o
                        break
                if hit:
                    hit["payment_status"] = "betaald"
                    hit.setdefault("history", []).append({"ts": _now(), "event": "betaald (notitie in WhatsApp)"})
                    d.setdefault("entries", []).append({
                        "ts": _now(), "type": "income",
                        "amount_eur": float(m.group(1).replace(",", ".")) if m else float(hit.get("price_eur") or 0),
                        "method": "bank", "customer": hit.get("customer"),
                        "note": "notitie in WhatsApp"})
                    ledger_save(d)
                    hf_sync_up()
                    return JSONResponse({"reply": f"✅ Order #{hit['num']} ({hit['customer']}) op betaald gezet "
                                                  f"en geld bijgeschreven.", "to_me": True,
                                         "to_chat": WA_SELF_JID})
            return JSONResponse({"reply": f"❓ Geen open order gevonden voor '{naam}'. "
                                          f"Check de klantnaam (zelfde spelling als in de order).",
                                 "to_me": True, "to_chat": WA_SELF_JID})
        if WA_TAKEOVER_HOURS > 0:
            wa_pause(jid)  # human takeover: jij hebt zelf geantwoord
            return JSONResponse({"reply": None, "note": f"pauze {WA_TAKEOVER_HOURS:g}u"})
        return JSONResponse({"reply": None, "note": "takeover uit — bot blijft actief"})

    if wa_is_paused(jid):
        return JSONResponse({"reply": None, "note": "pauze"})

    # Eerste bericht van dit nummer ooit -> welkomstbericht van Younes (1x per nummer,
    # staat in de ledger dus dit overleeft restarts/redeploys).
    with _ledlock:
        d = ledger_load()
        welcomed = d.setdefault("whatsapp", {}).setdefault("welcomed", {})
        is_new = key not in welcomed
        if is_new:
            welcomed[key] = True
            ledger_save(d)
    if is_new:
        threading.Thread(target=hf_sync_up, daemon=True).start()
        return JSONResponse({"reply": WA_WELCOME})

    if mtype == "audio":
        b64 = b.get("audio_b64") or ""
        if not b64:
            return JSONResponse({"reply": None})
        path = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"wa_{int(time.time() * 1000)}.ogg")
        with open(path, "wb") as f:
            f.write(base64.b64decode(b64))
        audio_bytes = b""
        try:
            text = await asyncio.to_thread(transcribe, path)
            with open(path, "rb") as f:
                audio_bytes = f.read()
        except Exception as e:  # noqa: BLE001
            print("wa asr error:", e)
            return JSONResponse({"reply": "Stemmetje kon ik niet verwerken 🙈 typ even wat je zoekt."})
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
        if not text:
            return JSONResponse({"reply": "❓ Geen spraak herkend, typ even."})
        voice_note_opslaan(text, "whatsapp", chat_key="wa:" + key, audio_bytes=audio_bytes)
    elif mtype == "image":
        # Foto onthouden in de chat-status: de bot mag daarna NIET meer om een
        # foto vragen (klantklacht: "hij zegt steeds dat er geen foto's zijn").
        try:
            with _ledlock:
                d = ledger_load()
                c = d.setdefault("whatsapp", {}).setdefault("chats", {}).setdefault(key, {})
                c["foto_ontvangen"] = True
                ledger_save(d)
        except Exception:  # noqa: BLE001
            pass
        if not text:
            return JSONResponse({"reply": WA_IMG_REPLY})
        text = "[foto] " + text

    if not text:
        return JSONResponse({"reply": None})

    # Vaste sjabloon-antwoorden (geen LLM meer): eerste keyword-match wint.
    # Geen match = stil — Younes ziet de chat en reageert zelf in de app.
    low = text.lower()
    answer = None
    for keywords, antw in WA_TEMPLATES:
        if any(k in low for k in keywords):
            answer = antw
            break
    if not answer:
        return JSONResponse({"reply": None})
    # Harde anti-dubbel-beveiliging: identiek aan het vorige bot-bericht? -> "👍"
    try:
        def _norm(s):
            return re.sub(r"\W+", "", str(s).lower())[:120]
        laatste_bot = ""
        with _ledlock:
            d = ledger_load()
            c = d.setdefault("whatsapp", {}).setdefault("chats", {}).setdefault(key, {})
            for g in reversed(c.get("gesprek", [])):
                if g.get("r") == "b":
                    laatste_bot = str(g.get("t", ""))
                    break
        if _norm(answer) and _norm(answer) == _norm(laatste_bot):
            answer = "👍"
    except Exception:  # noqa: BLE001
        pass
    try:  # bot-antwoorden ook in het opgeslagen gesprek (teruglezen na herstart)
        with _ledlock:
            d = ledger_load()
            c = d.setdefault("whatsapp", {}).setdefault("chats", {}).setdefault(key, {})
            g = c.setdefault("gesprek", [])
            g.append({"r": "b", "t": str(answer)[:200]})
            del g[:-30]
            ledger_save(d)
    except Exception:  # noqa: BLE001
        pass
    return JSONResponse({"reply": answer})

@app.post("/whatsapp/history_chats")
async def wa_history_chats(req: Request):
    """Bridge stuurt de chat-jids uit de WhatsApp-historie (iedereen die ooit
    appte) -> opslaan als broadcast-doelen."""
    try:
        b = await req.json()
    except Exception:  # noqa: BLE001
        return JSONResponse({"ok": False}, status_code=400)
    nieuw = 0
    with _ledlock:
        d = ledger_load()
        known = d.setdefault("whatsapp", {}).setdefault("known_chats", {})
        for jid in (b.get("chats") or [])[:800]:
            if isinstance(jid, str) and jid.endswith("@s.whatsapp.net"):
                k = _wa_key(jid)
                if k and k not in known:
                    known[k] = {"jid": jid}
                    nieuw += 1
        ledger_save(d)
    print(f"historie-chats opgeslagen: {nieuw} nieuw")
    if nieuw:
        threading.Thread(target=hf_sync_up, daemon=True).start()
    return JSONResponse({"ok": True, "nieuw": nieuw})

@app.get("/whatsapp/contacts")
async def wa_contacts(req: Request):
    """Alle bekende chats (CRM + historie) met hun volledige adres."""
    key = req.query_params.get("key", "")
    if ACCESS_CODE and key != ACCESS_CODE:
        return JSONResponse({"ok": False, "error": "geen toegang"}, status_code=401)
    with _ledlock:
        d = ledger_load()
    chats = d.get("whatsapp", {}).get("chats", {})
    known = d.get("whatsapp", {}).get("known_chats", {})
    merged = {}
    for k, c in chats.items():
        merged[k] = {"key": k, "jid": (c.get("jid") or k + "@s.whatsapp.net"),
                     "naam": (c.get("klantnaam") or c.get("naam") or "")}
    for k, c in known.items():
        if k not in merged:
            merged[k] = {"key": k, "jid": (c.get("jid") or k + "@s.whatsapp.net"), "naam": ""}
    return JSONResponse({"ok": True, "chats": list(merged.values())})

@app.post("/whatsapp/send")
async def wa_send(req: Request):
    """Direct een WhatsApp-bericht versturen via de bridge (test/debug).
    key in query is verplicht. Body: {"to": "<nummer|jid>", "text": "..."}"""
    key = req.query_params.get("key", "")
    if ACCESS_CODE and key != ACCESS_CODE:
        return JSONResponse({"ok": False, "error": "geen toegang"}, status_code=401)
    try:
        b = await req.json()
    except Exception:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": "ongeldige json"}, status_code=400)
    to = str(b.get("to") or "").strip()
    tekst = str(b.get("text") or "").strip()
    if not to or not tekst:
        return JSONResponse({"ok": False, "error": "to en text verplicht"}, status_code=400)
    ok = await asyncio.to_thread(_stuur_via_bridge, to, tekst)
    return JSONResponse({"ok": ok})

@app.post("/whatsapp/notify_unreadable")
async def wa_notify_unreadable(req: Request):
    """Bridge kan een inkomend bericht niet ontsleutelen (CIPHERTEXT-stub, oude
    sessie na herkoppelen) -> waarschuw Younes in zijn bericht-jezelf-chat."""
    try:
        b = await req.json()
    except Exception:  # noqa: BLE001
        return JSONResponse({"ok": False}, status_code=400)
    num = str(b.get("num") or _wa_key(b.get("chat") or "")).strip()
    return JSONResponse({
        "reply": ("⚠️ Bericht van +" + num + " kon ik niet lezen (oude versleuteling na "
                  "herkoppelen). De chat ziet het wél op je telefoon. Vraag de afzender "
                  "het gesprek met je zakelijke nummer te verwijderen en opnieuw te "
                  "sturen — daarna kan de bot het wel lezen."),
        "to_me": True, "to_chat": WA_SELF_JID})


@app.post("/whatsapp/qr")
async def wa_qr_post(req: Request):
    """Bridge post hier elke nieuwe QR (data-URL) en/of de koppelcode."""
    try:
        b = await req.json()
    except Exception:  # noqa: BLE001
        return JSONResponse({"ok": False}, status_code=400)
    if "qr" in b:
        WA_STATE["qr"] = b.get("qr") or None
        WA_STATE["qr_ts"] = time.time()
        WA_STATE["status"] = "wacht op scan"
    if b.get("pair_code"):
        WA_STATE["pair_code"] = str(b["pair_code"])[:12]
        if WA_STATE["status"] in ("startend", "verbinden"):
            WA_STATE["status"] = "wacht op scan"
    return JSONResponse({"ok": True})

@app.get("/whatsapp/qr")
async def wa_qr_get():
    return JSONResponse({"status": WA_STATE["status"], "qr": WA_STATE["qr"],
                         "qr_age": int(time.time() - WA_STATE["qr_ts"]) if WA_STATE["qr"] else None,
                         "pair_code": WA_STATE["pair_code"],
                         "error": WA_STATE["error"], "enabled": WHATSAPP_ENABLED})

@app.post("/whatsapp/status")
async def wa_status_post(req: Request):
    try:
        b = await req.json()
    except Exception:  # noqa: BLE001
        return JSONResponse({"ok": False}, status_code=400)
    st = str(b.get("status") or "")
    if st:
        WA_STATE["status"] = st
    WA_STATE["error"] = str(b.get("error") or "")
    if st == "connected":
        WA_STATE["qr"] = None
        if (HF_DATASET and HF_TOKEN and time.time() - WA_STATE["backup_ts"] > 60):
            WA_STATE["backup_ts"] = time.time()

            def _bk():
                time.sleep(130)  # pas backuppen als de verbinding stabiel is (geen halve sessie)
                wa_backup_up()
            threading.Thread(target=_bk, daemon=True).start()
    return JSONResponse({"ok": True})

@app.get("/whatsapp/status")
async def wa_status_get():
    with _ledlock:
        d = ledger_load()
    return JSONResponse({"status": WA_STATE["status"], "enabled": WHATSAPP_ENABLED,
                         "error": WA_STATE["error"],
                         "gepauzeerde_chats": len(_wa_pause_map(d)),
                         "takeover_uren": WA_TAKEOVER_HOURS})

@app.post("/whatsapp/logout")
async def wa_logout_post():
    """Apparaat is op de telefoon afgemeld: de (nu dode) sessie ook uit de
    HF-backup wissen, zodat een herstart niet per ongeluk de oude sessie
    terugzet en de bridge in een 401-loop blijft hangen."""
    WA_STATE["status"] = "uitgelogd"
    WA_STATE["pair_code"] = None
    WA_STATE["qr"] = None
    if HF_DATASET and HF_TOKEN:
        try:
            from huggingface_hub import HfApi
            HfApi(token=HF_TOKEN).delete_file(
                path_in_repo="wa-session.zip", repo_id=HF_DATASET, repo_type="dataset")
            print("dode wa-sessie uit HF dataset verwijderd")
        except Exception as e:  # noqa: BLE001
            print("wa logout cleanup failed:", e)
    return JSONResponse({"ok": True})

WA_CHATS_PAGE = """<!doctype html><html lang="nl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="30">
<title>Stand van zaken — WhatsApp</title><style>
body{font-family:system-ui,Segoe UI,sans-serif;background:#0f1716;color:#e8efec;margin:0;padding:20px;max-width:760px}
h1{font-size:20px}.sub{color:#8aa39c;font-size:13px;margin-bottom:14px}
.card{background:#182522;border:1px solid #24382f;border-radius:12px;padding:14px;margin-top:12px}
.naam{font-weight:700;font-size:16px}.badge{display:inline-block;background:#25d366;color:#06281a;
border-radius:20px;padding:2px 10px;font-size:12px;font-weight:700;margin-left:8px}
.badge.grijs{background:#31473f;color:#b9cdc6}
.rij{font-size:14px;margin-top:6px;color:#c8d6d1}.rij b{color:#e8efec}
.leeg{color:#8aa39c}.laatste{color:#8aa39c;font-size:13px;margin-top:8px;border-top:1px solid #24382f;padding-top:8px}
.ontbreekt{color:#ffb86b;font-weight:600}
.banner{border-radius:12px;padding:12px 14px;margin-bottom:6px;font-weight:700;font-size:15px}
.banner.ok{background:#123a28;border:1px solid #25d366;color:#7dffb8}
.banner.slecht{background:#3a1212;border:1px solid #ff6b6b;color:#ffb0b0}
.banner .subfont{display:block;font-weight:400;font-size:13px;margin-top:4px;color:inherit;opacity:.85}
</style></head><body>
<h1>📋 Stand van zaken</h1>
<div class="sub">Elke klant-chat + wat er nog mist · ververst elke 30s · besteld/betaald noteren kan gewoon in WhatsApp: "besteld haroun groene dunks €110"</div>
__BANNER__
__CONTENT__
</body></html>"""

@app.get("/wa-chats", response_class=HTMLResponse, include_in_schema=False)
def wa_chats_page(request: Request):
    key = request.query_params.get("key", "")
    if ACCESS_CODE and key != ACCESS_CODE:
        return HTMLResponse("<h1>401 — toegangscode ontbreekt</h1><p>voeg ?key=... toe aan de URL</p>",
                            status_code=401)
    with _ledlock:
        d = ledger_load()
    chats = d.get("whatsapp", {}).get("chats", {})
    paused = _wa_pause_map(dict(d))
    orders = d.get("orders", [])
    # Status-balk: is de bot nu actief en wanneer kwam het laatste bericht binnen?
    verbonden = WA_STATE.get("status") == "connected"
    laatste_ts = max([str((c.get("laatste") or {}).get("ts") or "")
                      for c in chats.values()] or [""])
    try:
        laatste_min = int((time.time() - datetime.fromisoformat(laatste_ts).timestamp()) / 60)
    except Exception:  # noqa: BLE001
        laatste_min = None
    if laatste_min is None:
        laatste_txt = "nog geen berichten ontvangen"
    elif laatste_min < 1:
        laatste_txt = "laatste bericht: net nu"
    elif laatste_min < 60:
        laatste_txt = f"laatste bericht: {laatste_min} min geleden"
    else:
        laatste_txt = f"laatste bericht: {laatste_min // 60} uur geleden"
    if verbonden:
        banner = ('<div class="banner ok">✅ Bot is ACTIEF en verbonden met WhatsApp'
                  f'<span class="subfont">{laatste_txt} · check: {datetime.now().strftime("%H:%M")}</span></div>')
    else:
        banner = (f'<div class="banner slecht">❌ Bot is NIET verbonden (status: {WA_STATE.get("status")})'
                  '<span class="subfont">Herstart de service of koppel opnieuw via /wa-qr</span></div>')
    if not chats:
        content = '<div class="card leeg">Nog geen chats binnengekomen.</div>'
    else:
        def _sort(kv):
            return str((kv[1].get("laatste") or {}).get("ts") or "")
        delen = []
        for num, c in sorted(chats.items(), key=_sort, reverse=True):
            naam = c.get("klantnaam") or c.get("naam") or ("+" + num)
            status = c.get("status") or "onbekend"
            badge = f'<span class="badge">{status}</span>' if num not in paused \
                else '<span class="badge grijs">bot uit</span>'
            rijen = []
            if c.get("gezocht"):
                rijen.append(f'<div class="rij">🔎 <b>Zoekt:</b> {c["gezocht"]}</div>')
            if c.get("maat"):
                rijen.append(f'<div class="rij">📏 <b>Maat:</b> {c["maat"]}</div>')
            if c.get("prijs_afgesproken") is not None:
                rijen.append(f'<div class="rij">💶 <b>Afgesproken:</b> €{float(c["prijs_afgesproken"]):g}</div>')
            elif c.get("prijs_gegeven"):
                rijen.append('<div class="rij">💶 prijs genoemd (bedrag niet vastgelegd)</div>')
            else:
                rijen.append('<div class="rij ontbreekt">💶 nog geen prijs genoemd</div>')
            rijen.append(f'<div class="rij">📷 foto/link: {"✅" if c.get("foto_ontvangen") else "❌ nog geen"}</div>')
            if c.get("ontbreekt") and c.get("ontbreekt").lower() not in ("leeg", "niets", "niks", "-"):
                rijen.append(f'<div class="rij ontbreekt">⚠️ Mist nog: {c["ontbreekt"]}</div>')
            matches = [o for o in orders
                       if (c.get("klantnaam") or "").lower()
                       and c["klantnaam"].lower() in str(o.get("customer", "")).lower()][-3:]
            if matches:
                otxt = " · ".join(f"#{o['num']} {o.get('items') or '?'} €{float(o.get('price_eur') or 0):g} "
                                  f"📦{o.get('order_status')} 💶{o.get('payment_status')}" for o in matches)
                rijen.append(f'<div class="rij">🛒 <b>Orders:</b> {otxt}</div>')
            laat = c.get("laatste") or {}
            laatste = f'{laat.get("van", "?")}: {laat.get("tekst", "")}' if laat else "nog niks"
            delen.append(
                f'<div class="card"><div class="naam">{naam} <span style="color:#8aa39c;font-size:12px">+{num}</span>{badge}</div>'
                + "".join(rijen)
                + f'<div class="laatste">💬 {laatste}</div></div>')
        content = "".join(delen)
    return HTMLResponse(WA_CHATS_PAGE.replace("__BANNER__", banner).replace("__CONTENT__", content))

WA_QR_PAGE = """<!doctype html><html lang="nl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>WhatsApp koppelen — YZ Agent</title><style>
body{font-family:system-ui,Segoe UI,sans-serif;background:#0f1716;color:#e8efec;margin:0;padding:24px;max-width:560px}
h1{font-size:20px}.sub{color:#8aa39c;font-size:13px}
.card{background:#182522;border:1px solid #24382f;border-radius:12px;padding:18px;margin-top:16px}
input{background:#0c1210;border:1px solid #2b463c;color:#e8efec;border-radius:8px;padding:10px;width:100%;box-sizing:border-box}
button{background:#25d366;border:0;border-radius:8px;padding:10px 16px;font-weight:700;cursor:pointer;margin-top:10px}
#qr{max-width:320px;width:100%;background:#fff;border-radius:12px;padding:10px;margin-top:12px}
#stat{font-size:14px;margin-top:10px}
ol{color:#c8d6d1;font-size:14px;line-height:1.6}
.note{color:#8aa39c;font-size:12px;margin-top:14px}
</style></head><body>
<h1>📱 WhatsApp koppelen</h1>
<div class="sub">Koppel je bestaande nummer als apparaat — je telefoon blijft gewoon werken.</div>
<div class="card"><input id="key" placeholder="toegangscode"><button onclick="save()">Opslaan</button>
<div id="stat">laden…</div><img id="qr" style="display:none" alt="QR-code">
<div id="pair" style="display:none">
<div style="margin-top:16px;color:#8aa39c;font-size:13px">Geen tweede scherm? Koppel <b>met code</b>:</div>
<div style="font-size:34px;font-weight:800;letter-spacing:.1em;margin:6px 0" id="paircode"></div>
<div class="note">WhatsApp → <b>Instellingen → Gekoppelde apparaten → Apparaat koppelen</b> →
onderaan <b>'Koppelen met telefoonnummer in plaats daarvan'</b> → typ deze code.<br>
(De QR hierboven werkt alleen vanaf een ánder scherm, bijv. je laptop.)</div></div>
<ol style="margin-top:16px"><li>Open <b>WhatsApp</b> op je telefoon</li><li><b>Instellingen → Gekoppelde apparaten → Apparaat koppelen</b></li>
<li>Scan de QR <i>vanaf een ander scherm</i>, of gebruik de code hierboven</li></ol>
<div class="note">Na het koppelen blijft alles gewoon zichtbaar op je telefoon.<br>
__TAKEOVER_NOTE__
Typ <b>bot uit</b> in een chat = bot 7 dagen uit · <b>bot aan</b> = weer aan.</div>
</div>
<script>let K=new URLSearchParams(location.search).get('key')||(document.cookie.split('; ').find(r=>r.startsWith('key='))||'').slice(4)||'';
if(K)document.getElementById('key').value=K;
function save(){K=document.getElementById('key').value.trim();document.cookie='key='+K+';path=/;max-age=31536000';tick();}
async function tick(){try{const r=await fetch('/whatsapp/qr',{headers:{'X-Access-Code':K}});
if(r.status===401){document.getElementById('stat').textContent='Eerst je toegangscode invullen.';document.getElementById('qr').style.display='none';return;}
const d=await r.json();const s=document.getElementById('stat');const q=document.getElementById('qr');const p=document.getElementById('pair');
if(d.pair_code){p.style.display='block';document.getElementById('paircode').textContent=d.pair_code;}else{p.style.display='none';}
if(d.status==='connected'){s.textContent='✅ WhatsApp is verbonden! Je kunt dit tabblad sluiten.';q.style.display='none';p.style.display='none';}
else if(d.qr&&String(d.qr).startsWith('data:image')){s.textContent='⏳ Wachten op scan… (QR '+d.qr_age+'s oud)';q.src=d.qr;q.style.display='block';}
else if(d.status==='uitgelogd'){s.textContent='Sessie verlopen — herstart de service en koppel opnieuw.';q.style.display='none';}
else{s.textContent='⏳ Status: '+d.status+' — QR verschijnt vanzelf…';q.style.display='none';}}catch(e){document.getElementById('stat').textContent='geen verbinding met de bridge…';}}
setInterval(tick,5000);tick();</script></body></html>"""

@app.get("/wa-qr", response_class=HTMLResponse, include_in_schema=False)
def wa_qr_page():
    if WA_TAKEOVER_HOURS > 0:
        note = f"Reageer jij zelf in een chat? Dan stopt de bot daar voor {WA_TAKEOVER_HOURS:g} uur.<br>"
    else:
        note = "Jij en de bot kunnen allebei in een chat reageren — de bot gaat niet uit als jij typt.<br>"
    return HTMLResponse(WA_QR_PAGE.replace("__TAKEOVER_NOTE__", note))

# ---------------- aanvragen (YZ SHOP foto-intake) ----------------
AANVRAAG_MAP = os.path.join("static", "aanvraag-fotos")
AANVRAAG_LEDGER = os.path.join("data", "aanvragen.json")

def _aanvraag_ledger_add(item):
    try:
        with _ledlock:
            lijst = []
            if os.path.exists(AANVRAAG_LEDGER):
                try:
                    with open(AANVRAAG_LEDGER, "r", encoding="utf-8") as f:
                        lijst = json.load(f)
                except Exception:  # noqa: BLE001
                    lijst = []
            lijst.append(item)
            with open(AANVRAAG_LEDGER, "w", encoding="utf-8") as f:
                json.dump(lijst, f, ensure_ascii=False, indent=1)
    except Exception as e:  # noqa: BLE001
        print("aanvraag ledger fout:", e)

@app.post("/api/aanvragen")
async def api_aanvragen(request: Request):
    form = await request.form()
    if (form.get("website") or "").strip():
        return JSONResponse({"ok": True, "url": ""})  # honeypot: spampot stil weg
    naam = (form.get("name") or "?").strip()[:80]
    telefoon = (form.get("phone") or "").strip()[:30]
    notitie = (form.get("note") or "").strip()[:500]
    foto = form.get("photo")
    if foto is None or not getattr(foto, "filename", ""):
        return JSONResponse({"ok": False, "error": "foto ontbreekt"}, status_code=400)
    data = await foto.read()
    if len(data) > 12 * 1024 * 1024:
        return JSONResponse({"ok": False, "error": "te groot"}, status_code=413)
    ext = os.path.splitext(foto.filename or "")[1].lower()
    if ext not in (".jpg", ".jpeg", ".png", ".webp", ".heic"):
        ext = ".jpg"
    os.makedirs(AANVRAAG_MAP, exist_ok=True)
    bestand = time.strftime("%Y%m%d-%H%M%S") + "-" + os.urandom(2).hex() + ext
    with open(os.path.join(AANVRAAG_MAP, bestand), "wb") as f:
        f.write(data)
    item = {"ts": _now(), "naam": naam, "telefoon": telefoon, "note": notitie, "foto": bestand}
    _aanvraag_ledger_add(item)
    # Publieke basis-URL: BASE_URL env (Render zet X-Forwarded-*), anders uit de request.
    host = request.headers.get("x-forwarded-host") or request.headers.get("host")
    proto = request.headers.get("x-forwarded-proto") or request.url.scheme
    base = BASE_URL or (f"{proto}://{host}" if host else str(request.base_url).rstrip("/"))
    publiek = base + "/aanvraag-fotos/" + bestand
    try:
        _stuur_via_bridge(WA_SELF_JID, "NIEUWE AANVRAAG van " + naam + " (" + telefoon + ")" + (": " + notitie if notitie else "") + "\nFoto: " + publiek)
    except Exception as e:  # noqa: BLE001
        print("aanvraag bridge fout:", e)
    return JSONResponse({"ok": True, "url": publiek})
# ---------------- checkout via basetao wallet ----------------
BASETAO_BASE = "https://www.basetao.com/best-taobao-agent-service"
BASETAO_PAY_METHOD = os.environ.get("BASETAO_PAY_METHOD", "ideal").strip()
BASETAO_USER = os.environ.get("BASETAO_USER", "").strip()
BASETAO_PASS = os.environ.get("BASETAO_PASS", "").strip()
_BT_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
          "(KHTML, like Gecko) Chrome/152.0.0.0 Safari/537.36")
_bt_sessie = None  # module-cache: ingelogde requests.Session

def _bt_ajax_headers(referer):
    return {"X-Requested-With": "XMLHttpRequest",
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Referer": referer,
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8"}

def _bt_login(s):
    """Log zelf in met BASETAO_USER/PASS (endpoint /login/log_in, zelfde
    token+data-JSON patroon als de rest van de site). Werkt server-side,
    dus geen geoogste cookies meer nodig."""
    r = s.get(BASETAO_BASE + "/accounts/login.html", timeout=30,
              params={"r": str(int(time.time() * 1000))})
    m = re.search(r'name="bt_sb_token"[^>]*value="([0-9a-f]+)"', r.text)
    if not m:
        raise RuntimeError("basetao login-token niet gevonden (Cloudflare?)")
    r2 = s.post(BASETAO_BASE + "/login/log_in",
                headers=_bt_ajax_headers(BASETAO_BASE + "/accounts/login.html"),
                data={"bt_sb_token": m.group(1),
                      "data": json.dumps({"user-login": BASETAO_USER,
                                          "login-pass": BASETAO_PASS})},
                timeout=30)
    try:
        out = r2.json()
    except ValueError:
        raise RuntimeError("basetao login gaf geen JSON http " + str(r2.status_code))
    if str(out.get("value")) != "1":
        raise RuntimeError("basetao login weigerde: " + str(out.get("msg"))[:120])
    return s

def _bt_session(vers=False):
    """Ingelogde sessie: log in met user/pass (voorkeur), anders val terug op
    de oude BASETAO_COOKIE."""
    global _bt_sessie
    if _bt_sessie is not None and not vers:
        return _bt_sessie
    s = requests.Session()
    s.headers.update({"User-Agent": _BT_UA})
    if BASETAO_USER and BASETAO_PASS:
        _bt_login(s)
    elif BASETAO_COOKIE:
        s.headers.update({"Cookie": BASETAO_COOKIE})
    else:
        raise RuntimeError("BASETAO_USER/PASS of BASETAO_COOKIE niet ingesteld")
    _bt_sessie = s
    return s

def basetao_create_recharge(amount_eur: float, _retry=True):
    """Maak een basetao top-up order voor exact dit bedrag en geef de gehoste
    betaalpagina-URL terug: een normale iDEAL-checkout voor de klant, het
    bedrag komt op onze eigen basetao-wallet."""
    s = _bt_session()
    r = s.get(BASETAO_BASE + "/my_account/account/recharge.html", timeout=30,
              params={"r": str(int(time.time() * 1000))},
              headers={"Referer": BASETAO_BASE + "/my_account/welcome.html"})
    m = re.search(r'name="bt_sb_token"[^>]*value="([0-9a-f]+)"', r.text)
    if not m:
        if _retry and BASETAO_USER:
            _bt_session(vers=True)
            return basetao_create_recharge(amount_eur, _retry=False)
        raise RuntimeError("basetao-token niet gevonden (sessie verlopen?)")
    r2 = s.post(BASETAO_BASE + "/account/recharge",
                headers=_bt_ajax_headers(BASETAO_BASE + "/my_account/account/recharge.html"),
                data={"bt_sb_token": m.group(1),
                      "data": json.dumps({"option": BASETAO_PAY_METHOD,
                                          "money": f"{amount_eur:.2f}"})},
                timeout=30)
    try:
        out = r2.json()
    except ValueError:
        raise RuntimeError("basetao gaf geen JSON (Cloudflare?) http " + str(r2.status_code))
    if str(out.get("value")) != "1":
        msg = str(out.get("msg"))[:120]
        if "log in again" in msg.lower() and _retry and BASETAO_USER:
            _bt_session(vers=True)
            return basetao_create_recharge(amount_eur, _retry=False)
        raise RuntimeError("basetao weigerde: " + msg)
    torecharge = BASETAO_BASE + "/torecharge/index/" + str(out["msg"]) + ".html"
    # De torecharge-pagina is sessie-gebonden: een klant zonder basetao-login
    # komt daar op de homepage terecht. Met onze eigen bot-sessie gehaald ketent
    # hij door: torecharge -> redirect.ppro.com (JWT, cookies loos) ->
    # pay.ideal.nl (echte publieke iDEAL-pagina). We lossen de keten server-side
    # op en sturen de klant rechtstreeks naar iDEAL.
    try:
        r3 = s.get(torecharge, allow_redirects=False, timeout=30,
                   headers={"Referer": BASETAO_BASE + "/my_account/account/recharge.html",
                            "Accept": "text/html"})
        hop = r3.headers.get("Location", "") or ""
        if r3.status_code in (301, 302, 303, 307, 308) and "ppro.com" in hop:
            r4 = requests.get(hop, allow_redirects=False, timeout=30,
                              headers={"User-Agent": _BT_UA})
            ideal = r4.headers.get("Location", "") or ""
            if r4.status_code in (301, 302, 303, 307, 308) and "pay.ideal.nl" in ideal:
                return ideal
            print("checkout: ppro-hop zonder ideal-locatie:", r4.status_code, ideal[:100])
        else:
            print("checkout: torecharge zonder ppro-hop:", r3.status_code, hop[:100])
    except Exception as e:  # noqa: BLE001
        print("checkout: ppro-hop fout:", e)
    return torecharge

def _checkout_error(status: int, bericht: str):
    """Vriendelijke HTML-foutpagina voor /checkout i.p.v. kale JSON."""
    html = (
        '<!doctype html><html lang="nl"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Betaling niet mogelijk — YZ Shop</title><style>'
        'body{font-family:system-ui,Segoe UI,sans-serif;background:#0c0d10;color:#f2f3f5;'
        'margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;padding:24px}'
        '.card{max-width:460px;text-align:center}'
        'h1{font-size:22px;font-weight:700;margin:0 0 12px}'
        'p{color:#9aa1ad;font-size:15px;line-height:1.6;margin:0 0 28px}'
        'a{display:inline-block;color:#f7941d;font-weight:700;text-decoration:none;font-size:15px;'
        'border:1px solid #2a2d33;border-radius:10px;padding:10px 18px}'
        'a:hover{filter:brightness(1.15)}'
        '</style></head><body><div class="card">'
        '<h1>Betaling niet mogelijk</h1>'
        '<p>' + bericht + '<br>Probeer het over een paar minuten opnieuw '
        'of neem contact op via WhatsApp.</p>'
        '<a href="/">&larr; Terug naar de shop</a>'
        '</div></body></html>')
    return HTMLResponse(html, status_code=status)

@app.get("/checkout", include_in_schema=False)
def checkout(amount: str = ""):
    try:
        amt = round(float(str(amount).replace(",", ".")), 2)
    except ValueError:
        return _checkout_error(400, "Het opgegeven bedrag is ongeldig.")
    if amt < 10:
        return _checkout_error(400, "De minimale betaling is €10.")
    if amt > 5000:
        return _checkout_error(400, "De maximale betaling is €5000.")
    try:
        pay_url = basetao_create_recharge(amt)
    except Exception as e:  # noqa: BLE001
        print("checkout error:", e)
        return _checkout_error(502, "De betaalpartner accepteert de betaling nu even niet.")
    # topup meteen boeken: zo is "gestort via links" altijd vergelijkbaar met de kosten
    with _ledlock:
        d = ledger_load()
        d["entries"].append({
            "ts": _now(), "type": "topup", "amount_eur": amt, "method": "ideal",
            "note": "checkout-link aangemaakt (iDEAL → basetao wallet)", "source": "checkout"})
        ledger_save(d)
    threading.Thread(target=hf_sync_up, daemon=True).start()
    return RedirectResponse(pay_url, status_code=302)

_IMMUTABLE_EXTS = {".webp", ".png", ".jpg", ".jpeg", ".svg", ".ico",
                   ".woff", ".woff2", ".css", ".js"}

@app.get("/{full_path:path}", response_class=HTMLResponse, include_in_schema=False)
def page_fallback(full_path: str):
    """Volledige statische handler voor de gespiegelde site (incl. .html-mapping)."""
    if ".." in full_path:
        raise HTTPException(status_code=404)
    base = os.path.join("static", full_path.replace("/", os.sep))
    if full_path == "" or os.path.isdir(base):
        cand = os.path.join(base if os.path.isdir(base) else "static", "index.html")
    elif os.path.isfile(base):
        cand = base
    elif os.path.isfile(base + ".html"):
        cand = base + ".html"
    else:
        # generieke productpagina: elke catalogus-slug krijgt een werkende pagina
        if full_path.startswith("product/") and "/" not in full_path[8:]: 
            tmpl = os.path.join("static", "product", "_generic", "index.html")
            if os.path.isfile(tmpl):
                with open(tmpl, "rb") as f:
                    return Response(content=f.read(), media_type="text/html; charset=utf-8", headers={"Cache-Control": "no-cache, max-age=0"})
        raise HTTPException(status_code=404)
    if not os.path.isfile(cand):  # map zonder index.html -> 404 i.p.v. 500
        raise HTTPException(status_code=404)
    mime, _ = mimetypes.guess_type(cand)
    if mime is None and cand.endswith(".webp"):
        mime = "image/webp"
    ext = os.path.splitext(cand)[1].lower()
    if ext in _IMMUTABLE_EXTS:
        cache_hdr = "public, max-age=31536000, immutable"  # content-gehashde assets
    else:
        cache_hdr = "no-cache, max-age=0"  # .html en bestanden zonder extensie
    try:
        with open(cand, "rb") as f:
            data = f.read()
    except OSError:
        raise HTTPException(status_code=404)
    return Response(content=data, media_type=mime or "text/html",
                    headers={"Cache-Control": cache_hdr})

app.mount('/', StaticFiles(directory='static', html=True), name='site')

