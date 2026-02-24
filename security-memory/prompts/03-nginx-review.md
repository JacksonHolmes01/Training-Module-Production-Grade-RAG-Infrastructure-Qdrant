# Prompt 03 — NGINX Reverse Proxy Review (Auth + DoS + Headers)

## Retrieve references
```bash
EDGE_API_KEY=$(grep -E '^EDGE_API_KEY=' .env | cut -d= -f2-)

curl -sS -X POST http://localhost:8088/memory/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $EDGE_API_KEY" \
  -d '{
    "query": "nginx reverse proxy security auth enforcement timeouts headers rate limiting",
    "top_k": 10
  }' > /tmp/nginx_security_refs.json
```

## IDE prompt
> Review `nginx/templates/default.conf.template`. Use `/tmp/nginx_security_refs.json` as references. Ensure auth is enforced for all routes that should be protected, timeouts are safe, and headers do not leak sensitive info. Propose minimal diffs.

## Validate
- `curl http://localhost:8088/proxy-health`
- `curl -H "X-API-Key: $EDGE_API_KEY" http://localhost:8088/health`
