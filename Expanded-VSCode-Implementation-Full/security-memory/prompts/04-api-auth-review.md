# Prompt 04 — FastAPI Ingestion API Review (Input validation + auth)

## Retrieve references
```bash
EDGE_API_KEY=$(grep -E '^EDGE_API_KEY=' .env | cut -d= -f2-)

curl -sS -X POST http://localhost:8088/memory/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $EDGE_API_KEY" \
  -d '{
    "query": "api key authentication input validation rate limiting logging secrets fastapi security",
    "top_k": 10
  }' > /tmp/api_security_refs.json
```

## IDE prompt
> Review `ingestion-api/app/main.py` and related ingestion code. Use `/tmp/api_security_refs.json` as constraints. Identify input validation gaps (size/type), auth bypass risks, secret leakage in logs, and propose minimal diffs.

## Validate
- `docker compose up -d --build ingestion-api`
- `/health` works through nginx
