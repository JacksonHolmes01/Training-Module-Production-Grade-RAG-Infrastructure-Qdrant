# Lesson 4.3 — Using Expanded VSCode Implementation in an IDE (Cursor / VS Code / Windsurf)

> **What you’re building:** not “another chatbot,” but a workflow where your IDE assistant can retrieve the right security references and use them to produce grounded fixes.

This lesson teaches a **tool-agnostic workflow** that works even if you don’t configure an MCP server.

---

## Learning outcomes
By the end of this lesson, you will:
- Use `/memory/query` to retrieve relevant security guidance
- Apply that guidance in IDE prompts to review and fix lab code
- Use the prompt library included in this repo
- Understand how to keep memory fresh (maintenance)

---

## 1) The simplest workflow (works everywhere)

### Step 1: Retrieve security context (terminal)
Example: you want to review docker-compose for security issues.

```bash
EDGE_API_KEY=$(grep -E '^EDGE_API_KEY=' .env | cut -d= -f2-)

curl -sS -X POST http://localhost:8088/memory/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $EDGE_API_KEY" \
  -d '{
    "query": "docker compose security best practices secrets ports privileged mounts",
    "tags": ["docker", "cis"],
    "top_k": 8
  }' > /tmp/security_context.json
```

### Step 2: Ask your IDE assistant to review a file with this context
In Cursor/VS Code/Windsurf, prompt something like:

> Review `docker-compose.yml` in this repo. Use the retrieved security references in `/tmp/security_context.json` as constraints. Identify security issues, explain why each issue matters, and propose minimal diffs that keep the lab functional. If a claim is not supported by the retrieved references, say so.

Then either:
- paste the relevant chunk texts, or
- paste the JSON (or attach it if your IDE supports files)

This is the “lowest-friction” approach and teaches the concept cleanly.

---

## 2) Prompt library (what students should use)
This pack ships prompts under:

```
security-memory/prompts/
```

Suggested student exercise:
- pick a file (Dockerfile, nginx config, compose)
- run a memory query
- run the corresponding prompt template in the IDE
- implement fixes and re-run the lab

Prompts included:
- `01-dockerfile-review.md`
- `02-compose-review.md`
- `03-nginx-review.md`
- `04-api-auth-review.md`
- `05-dependency-risk-review.md`

---

## 3) What “major integration” actually means (in this repo)
When your professor says “major integration,” it usually means:

- the repo includes a **working vector DB memory** (Qdrant collection + ingestion)
- students can **query the memory** through a stable API tool endpoint
- students have **instructions and prompts** to use it to do security analysis of the lab
- optionally: the IDE can call the memory tool automatically (MCP/tooling)

In other words, you are not being asked to invent new AI.
You are being asked to make the existing lab **useful for security work**.

---

## 4) Optional: MCP / tool integration (advanced)
Some IDEs can call a “tool” automatically (MCP servers, extension tool APIs, etc.).
This is a *nice-to-have*, not required for the core learning goals.

If you do implement MCP/tooling later, the tool endpoint you already built is perfect:
- `/memory/query` is the retrieval tool
- `/chat` is the RAG generation tool

This pack includes optional notes under:
```
security-memory/mcp/
```

---

## 5) Maintenance (keeping the memory up to date)

### The simple maintenance loop
1. Add/update docs under `security-memory/data/`
2. Re-ingest:

```bash
docker exec -i ingestion-api python -m app.security_memory.ingest
```

Because ingestion uses upserts, you can re-run safely.

### When do you need a NEW collection?
If you change:
- embedding model (different vector space)
- embedding dimension
- distance metric

Then you must create a new collection (e.g., `ExpandedVSCodeMemory_v2`) and update `SECURITY_COLLECTION`.

---

## 6) Instructor notes (if you’re shipping this to students)
A clean teaching pattern is:

1. Students run the base lab (prove RAG works)
2. Students ingest Expanded VSCode implementation
3. Students practice “grounded security review” on lab artifacts
4. Students propose minimal diffs and validate the lab still works

This naturally teaches:
- retrieval grounding
- controls frameworks
- secure configuration and code review
- change management

---

## Checkpoint
You’re done when you can:
- retrieve relevant security standards via `/memory/query`
- use them in your IDE prompts to propose fixes
- implement at least one fix and keep the lab running
