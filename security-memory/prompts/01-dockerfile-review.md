# Prompt 01 — Dockerfile Security Review (Grounded)

## Step 1: Retrieve security references
```bash
EDGE_API_KEY=$(grep -E '^EDGE_API_KEY=' .env | cut -d= -f2-)

curl -sS -X POST http://localhost:8088/memory/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $EDGE_API_KEY" \
  -d '{
    "query": "Dockerfile container hardening best practices",
    "tags": ["docker","cis","owasp"],
    "top_k": 10
  }' > /tmp/docker_security_refs.json
```

## Step 2: IDE prompt (paste + adapt)
> Review `Dockerfile` in this repo. Use `/tmp/docker_security_refs.json` as your grounding references. Identify security issues, explain impact, and propose minimal diffs. Keep the lab functional. If you cannot support a claim using retrieved references, say so.

## Step 3: Validate
- rebuild containers if needed
- run `/health`
- run a chat query
