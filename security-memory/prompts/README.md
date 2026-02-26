# Prompt Library

These prompt templates walk you through reviewing lab files for security issues using your `ExpandedVSCodeMemory` collection as grounded reference material.

Before using any template, make sure:
- The lab is running (`docker compose up -d`)
- Security memory has been ingested (Lesson 4.1)
- `/memory/query` is responding (Lesson 4.2)

## How the workflow works

Each template follows the same four steps:

1. **Retrieve** — run a curl command to query your security memory and save the results to a temp file
2. **Extract** — run a Python one-liner to pull out just the plain text from the results
3. **Review** — paste the text and the file contents into your IDE chat along with the structured prompt
4. **Validate** — apply fixes one at a time and re-run the lab after each one to confirm nothing broke

## Templates

| File | What it reviews |
|------|----------------|
| `01-dockerfile-review.md` | `Dockerfile` — base images, root user, secrets, health checks |
| `02-compose-review.md` | `docker-compose.yml` — ports, secrets, privileged mode, resource limits |
| `03-nginx-review.md` | `nginx/templates/default.conf.template` — auth, headers, rate limiting |
| `04-api-auth-review.md` | `ingestion-api/app/main.py` — input validation, auth, secret leakage |
| `05-dependency-risk-review.md` | `requirements.txt` and image tags — pinning, CVEs, supply chain |
