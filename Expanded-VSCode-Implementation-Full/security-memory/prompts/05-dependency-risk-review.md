# Prompt 05 — Dependency & Supply Chain Risk Review

## Retrieve references
```bash
EDGE_API_KEY=$(grep -E '^EDGE_API_KEY=' .env | cut -d= -f2-)

curl -sS -X POST http://localhost:8088/memory/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $EDGE_API_KEY" \
  -d '{
    "query": "software supply chain security dependency pinning sbom vulnerability scanning container image tags",
    "top_k": 10
  }' > /tmp/supply_chain_refs.json
```

## IDE prompt
> Review Docker images and Python dependencies. Use `/tmp/supply_chain_refs.json` as references. Recommend pinning, scanning, SBOM, and update policies appropriate for a student lab.

## Validate
- no breaking changes; recommendations should be optional
