# Prompt 02 — docker-compose.yml Security Review (Grounded)

## Retrieve references
```bash
EDGE_API_KEY=$(grep -E '^EDGE_API_KEY=' .env | cut -d= -f2-)

curl -sS -X POST http://localhost:8088/memory/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $EDGE_API_KEY" \
  -d '{
    "query": "docker compose security best practices secrets ports privileged mounts networks resource limits",
    "tags": ["docker","cis"],
    "top_k": 10
  }' > /tmp/compose_security_refs.json
```

## IDE prompt
> Review `docker-compose.yml` using `/tmp/compose_security_refs.json` as constraints. Look for exposed ports, plaintext secrets, privileged mode, host mounts, network exposure, and resource limits. Propose minimal diffs that preserve lab behavior.

## Validate
- `docker compose up -d`
- confirm services healthy
