# KHL PDF Parser (Cloud) — v1.1.0

## Endpoints
- `GET /health`
- `GET /extract?pdf_url=...&uid=897694&publish=true`
- `GET /extract_batch?uids=897683,897694&season=1369&publish=true`
- `GET /cron?ics_url=<public-ics>&season=1369&window_min=65&publish=true`

Publishes JSON to `/files/<uid>.json` (local ephemeral) and, if R2 creds provided, to R2 at `khl/json/<uid>.json`.

### Env for R2 (optional)
```
R2_ACCOUNT_ID=...
R2_BUCKET=...
R2_ACCESS_KEY_ID=...
R2_SECRET_ACCESS_KEY=...
R2_PUBLIC_BASE=https://your-public-host
```

## Run
```
pip install -r requirements.txt
uvicorn main:app --port 8080
```