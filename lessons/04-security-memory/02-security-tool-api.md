# Lesson 4.2 — Expose Your Security Memory as an API Tool

> **What you're building:** a small set of API endpoints that turn your `ExpandedVSCodeMemory` Qdrant collection into something your IDE and AI assistant can actually *call* as a tool.
>
> **IDE** stands for **Integrated Development Environment** — it's the application you write code in (VS Code, for example). IDEs can be extended with AI tools that call APIs like the one you're building here.
>
> This is the step that bridges "we have a vector database full of security knowledge" to "my IDE can pull up the right standard and cite it while reviewing my code."

---

## Learning Outcomes
By the end of this lesson, you will:

- Add `/memory/health` and `/memory/query` endpoints to the FastAPI ingestion API
- Understand what a "retrieval tool contract" means and why the request/response shape matters
- (Optional) Connect memory retrieval results into your `/chat` pipeline

---

## 1) What Is a "Tool Contract" and Why Does It Matter?

Before writing any code, it helps to understand what we're building conceptually.

A **tool contract** is simply an agreement about how your API behaves — what it expects as input, and what it promises to return. This matters a lot in AI systems because the LLM (or your IDE extension — a plugin inside VS Code that adds AI capabilities) will be calling this endpoint programmatically. If the response shape is unpredictable or inconsistent, the tool breaks.

Think of it like a vending machine: you press B4, you always get the same thing. The machine doesn't return a surprise sometimes and nothing other times. Your API should work the same way.

### Request Shape
When something (your IDE, your assistant, a script) wants to query security memory, it sends:

```json
{
  "query": "review this docker-compose for security issues",
  "tags": ["docker", "cis"],
  "top_k": 6
}
```

- `query` — the natural language question or topic you're searching for
- `tags` — optional filters to narrow results to specific frameworks (e.g., only CIS, only OWASP)
- `top_k` — how many chunks to return (more = more context, but also more noise)

### Response Shape
The API returns a structured list of matching chunks:

```json
{
  "query": "...",
  "collection": "ExpandedVSCodeMemory",
  "top_k": 6,
  "results": [
    {
      "score": 0.83,
      "title": "CIS Docker Benchmark",
      "source": "docker-security",
      "tags": ["docker","cis"],
      "chunk_index": 12,
      "doc_path": "security-memory/data/docker-security/cis-docker-benchmark.md",
      "text": ".... chunk text ...."
    }
  ]
}
```

**Why is this shape useful?**

- `results[*].text` is the actual content you paste into a prompt or show to a user
- `tags` and `doc_path` tell you *where* the answer came from — making results explainable and auditable, not just "the AI said so"
- `score` is the similarity score from Qdrant — closer to 1.0 means a stronger match. This is useful for debugging when results seem off.

---

## 2) What Files Are Being Added and What Does Each One Do?

This lesson introduces a new Python package inside the `ingestion-api` container:

```
ingestion-api/app/security_memory/
  router.py
  schemas.py
  store.py
  ingest.py
  __init__.py
```

Here's what each file is responsible for:

**`router.py`** — defines the FastAPI endpoints (`/memory/health` and `/memory/query`). This is the "front door" of the tool — it receives HTTP requests and hands them off to `store.py`.

**`schemas.py`** — defines the Pydantic models that describe what valid requests and responses look like. Pydantic will automatically validate incoming data and return clear error messages if something is malformed, which saves a lot of debugging time.

**`store.py`** — the core logic. It takes a query string, converts it to an embedding using the text-embeddings service, then sends that embedding to Qdrant to find the closest matching chunks. It also powers the `/memory/health` check by pinging Qdrant and reporting collection status.

**`ingest.py`** — the ingestion script you ran in Lesson 4.1. It lives here so you can re-run it inside the container if you add new documents to `security-memory/data/`.

**`__init__.py`** — an empty file that tells Python "this folder is a package." Without it, the imports won't work.

> **All of these files already exist in the repo** — you don't need to create or copy anything.

---

## 3) Test Your Endpoints

All requests go through NGINX, which enforces your API key. First, pull your key from `.env`:

```bash
EDGE_API_KEY=$(grep -E '^EDGE_API_KEY=' .env | cut -d= -f2-)
```

This command reads your `.env` file, finds the line starting with `EDGE_API_KEY=`, and stores the value in a shell variable so you don't have to copy-paste it manually.

### Test 1: Health Check
```bash
curl -sS -H "X-API-Key: $EDGE_API_KEY" http://localhost:8088/memory/health | python -m json.tool
```

You should see something like:
```json
{
  "ok": true,
  "collection": "ExpandedVSCodeMemory",
  "points_count": 142
}
```

If `ok` is `false` or `points_count` is 0, it means ingestion hasn't run yet or the collection name doesn't match. Go back to Lesson 4.1 and re-run ingestion.

### Test 2: Query
```bash
curl -sS -X POST http://localhost:8088/memory/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $EDGE_API_KEY" \
  -d '{"query":"what is broken access control", "tags":["owasp"], "top_k": 5}' \
  | python -m json.tool
```

You should get back a list of results with real text from your OWASP documents. If results are empty, check that your OWASP docs were ingested and that the tag `owasp` was assigned correctly during ingestion.

---

## 4) Optional: Connect Memory to Your `/chat` Endpoint

Once `/memory/query` is working, you can go one step further and connect it to your `/chat` endpoint. This means that when a user sends a message, your app will automatically look up relevant security knowledge and include it in the prompt — so the AI's answer is grounded in real standards rather than just its training data.

**The idea:** before sending the user's message to the AI, make a quick separate call to Ollama asking "is this question security-related?" If the answer is yes, fetch memory and inject it into the prompt. If not, skip it and answer normally.

This is smarter than checking for specific keywords (like "owasp" or "cis") because it understands intent. Someone asking *"is this config safe?"* or *"how do I lock down my container?"* will correctly trigger a memory lookup even though they didn't use any technical terms.

---

**Step 1 — Create a function that asks Ollama whether the question is security-related**

Open `ingestion-api/app/main.py` in VS Code. At the very top of the file you'll see a block of `import` lines — add `import httpx` here if it isn't already there.

Then scroll down until you find the line that starts your chat route — it will look like `@app.post("/chat")`. Place the following function **directly above that line**, with one blank line separating them:

```python
import httpx

async def is_security_related(message: str) -> bool:
    """
    Asks Ollama to classify whether the user's message is security-related.
    Returns True if yes, False if no.
    """
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://ollama:11434/api/chat",
            json={
                "model": "llama3",  # use whichever model you have pulled
                "stream": False,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a classifier. Your only job is to decide if a message "
                            "is related to cybersecurity, infrastructure security, secure coding, "
                            "or security frameworks (OWASP, CIS, NIST, MITRE, etc.). "
                            "Reply with only the word YES or NO. No explanation."
                        )
                    },
                    {
                        "role": "user",
                        "content": message
                    }
                ]
            },
            timeout=10.0  # don't wait forever if Ollama is slow
        )
        result = response.json()

    # Pull out the model's reply and check if it said YES
    answer = result["message"]["content"].strip().upper()
    return answer.startswith("YES")
```

Breaking this down:
- We're making a second, separate call to Ollama — not to answer the question, just to classify it. This is sometimes called a **routing call**.
- The `system` message tells Ollama to act as a classifier and reply only with YES or NO. Keeping the instruction strict prevents it from replying with things like "Yes, this appears to be..." which would break our check.
- `answer.startswith("YES")` is defensive — even if the model adds a stray character, it'll still work.
- `timeout=10.0` prevents the app from hanging if Ollama takes too long to respond.

> **⚠️ Verify this:** The response shape `result["message"]["content"]` is correct for Ollama's `/api/chat` endpoint, but it can vary slightly depending on which version of Ollama you're running. If this function throws a `KeyError`, print out `result` to see the actual shape your Ollama returns and adjust the key path accordingly. (Use a LLM tool to troubleshoot and debug if you recieve this error)

---

**Step 2 — Create a helper function that fetches memory context**

Immediately below the `is_security_related` function you just added (still above `@app.post("/chat")`), paste this second function:

```python
async def get_memory_context(query: str, top_k: int = 4) -> str:
    """
    Calls /memory/query with the user's message and returns the results
    as a single formatted string ready to drop into a prompt.
    """
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/memory/query",
            json={"query": query, "top_k": top_k}
        )
        data = response.json()

    # If no results came back, return an empty string (nothing to inject)
    if not data.get("results"):
        return ""

    # Format each result as a labelled block so the AI knows where it came from
    chunks = []
    for r in data["results"]:
        chunks.append(f"[{r['title']} | {r['source']}]\n{r['text']}")

    # Join all blocks with a blank line between each
    return "\n\n".join(chunks)
```

- `top_k: int = 4` means "fetch 4 chunks by default" — increase this if you want more context, decrease it if responses feel noisy.

> **⚠️ Verify this:** The URL `http://localhost:8000/memory/query` assumes both functions are running inside the same `ingestion-api` container, where `localhost` correctly refers to itself. If you ever move this logic to a different container or service, you'd need to change `localhost` to the Docker service name instead (e.g., `http://ingestion-api:8000/memory/query`). If you're getting connection errors, this is the first thing to check. (Use a LLM tool to troubleshoot and debug if you recieve this error)

- The `for` loop labels each chunk with its title and source (e.g., `[CIS Docker Benchmark | docker-security]`) so the AI can cite where information came from.

---

**Step 3 — Wire both helpers into your chat route**

Now scroll down to your existing `@app.post("/chat")` route. You're not replacing the whole function — just adding three things inside it at the top, before the line that calls Ollama. Your updated route should look like this:

```python
@app.post("/chat")
async def chat(request: ChatRequest):
    # Ask Ollama: is this a security question?
    if await is_security_related(request.message):
        memory_context = await get_memory_context(request.message)
    else:
        memory_context = ""

    if memory_context:
        # Inject the retrieved chunks as background context for the AI
        system_prompt = (
            "You are a security-aware assistant. "
            "Use the reference material below to ground your answer:\n\n"
            + memory_context
        )
    else:
        # No security context needed — answer normally
        system_prompt = "You are a helpful assistant."

    # Pass system_prompt + request.message to Ollama as usual
    ...
```

The `system_prompt` is the instruction you give the AI at the start of every conversation — it sets the tone and provides background context. By injecting memory chunks here, you're essentially saying "here's some relevant reference material, now answer the user's question using it."

If the routing call decides the question isn't security-related, `memory_context` stays empty and the chat behaves exactly as it did before — no disruption to non-security questions.

---

## 5) Security Note: Keep These Endpoints Behind the API Key Gate

The memory endpoints are valuable — they contain curated security reference material that took effort to assemble. Even though the documents themselves aren't secrets, in a real organization this kind of governance material would be access-controlled.

This repo already enforces authentication via an API key header checked by NGINX. Make sure your memory endpoints follow the same rules:

- **Do not** expose `ingestion-api` directly on a host port — traffic should always go through `edge-nginx`
- **Do not** disable or bypass the `X-API-Key` check for convenience during testing (use the `$EDGE_API_KEY` variable instead)

If you're unsure whether your endpoints are protected, re-read the NGINX config to confirm that all `/memory/*` paths require the key.

---

## Checkpoint
You're done when all of the following are true:

- `/memory/health` returns `ok: true` through `http://localhost:8088`
- `/memory/query` returns relevant, non-empty results
- The base lab still works (chat, ingest, and retrieval are unaffected)

**Next:** Lesson 4.3 — how to use this memory in a real IDE workflow, without building a whole new chatbot.
