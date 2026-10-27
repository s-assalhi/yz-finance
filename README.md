# YZ Agent — kasboek + webshop-backend + WhatsApp-bridge

FastAPI-service (één Docker-container, deploy op Render): kasboek op
`data/ledger.json` met dashboard op `/admin`, webshop-backend (`/shop/*`,
`/checkout` iDEAL via Basetao-wallet) en een WhatsApp-koppeling via de
Baileys-bridge in `whatsapp-bridge/` (gekoppeld apparaat, telefoon blijft werken).

## Deploy

Render **Docker**-service: push deze map naar GitHub, Render bouwt de
`Dockerfile` en start `start.sh` (uvicorn + optioneel de Node-bridge wanneer
`WHATSAPP_ENABLED=1`).

## Env-vars

| Var | Waarvoor |
|---|---|
| `ACCESS_CODE` | toegangscode `/admin`, `/wa-qr`, `/wa-chats`, `/api/*` |
| `HF_TOKEN` + `HF_DATASET` | backup kasboek + WhatsApp-sessie (optioneel) |
| `BASETAO_USER` / `BASETAO_PASS` | server-side Basetao-login voor `/basetao` + `/checkout` |
| `GROQ_API_KEY` | vocenote-transcriptie (whisper-large-v3; Mistral/Gemini/local-whisper als fallback) |
| `WHATSAPP_ENABLED` | `1` = WhatsApp-bridge starten |
| `WA_WELCOME` / `WA_IMG_REPLY` | welkomsttekst / antwoord bij foto zonder tekst |
| `BASE_URL` | publieke basis-URL (fallback: request-headers) |

## Belangrijkste endpoints

- `/` shop · `/admin` dashboard · `/healthz` status · `/stats` JSON
- `/api/entry` · `/api/note` · `/api/order` (+ DELETE) · `/api/aanvragen`
- `/checkout?amount=` iDEAL-topup · `/basetao` saldo · `/api/basetao` browser-sync
- `/shop/*` webshop (intent, checkout, order, pay, status) · `/api/shop/*` admin
- `/wa-qr` koppelen · `/wa-chats` stand van zaken · `/whatsapp/*` bridge-hooks
- WhatsApp-commando's (eigen nummer): `bot uit/aan`, `prijs <product> <bedrag>`,
  `open vragen`, `antwoord N`, `stand van vandaag`, `besteld/betaald <klant> …`
- Klantchats krijgen vaste sjabloon-antwoorden (prijzen, levertijd, maat); de
  rest blijft stil voor handmatige opvolging.
