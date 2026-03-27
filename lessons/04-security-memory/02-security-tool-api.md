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

Once `/memory/query` is working, you can connect it to your existing `/chat` endpoint so that security questions are automatically answered using your curated standards rather than just the model's training data.

**The idea:** before the prompt is built, check whether the question is security-related. If it is, fetch relevant chunks from memory and inject them into the prompt as additional context. If not, the chat works exactly as before.

Your chat flow in `main.py` goes through `_chat_impl`, which calls `retrieve_sources`, then `build_prompt`, then `ollama_generate`. You don't need to rebuild any of that — you just need to fetch memory chunks before `build_prompt` is called and pass them in as extra context. Everything else stays the same.

---

**Step 1 — Add the security classifier function**

Open `ingestion-api/app/main.py`. Scroll to the line that reads `# Core chat implementation` and place this function directly above it:

```python
async def is_security_related(message: str) -> bool:
    """
    Asks Ollama to classify whether the message is security-related.
    Returns True if yes, False if no.
    """
    prompt = (
        "Your only job is to decide if the following message is related to "
        "cybersecurity, infrastructure security, secure coding, or security frameworks "
        "(OWASP, CIS, NIST, MITRE, etc.).\n"
        "Reply with only the word YES or NO. No explanation.\n\n"
        f"Message: {message}"
    )
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
            )
            r.raise_for_status()
            answer = (r.json().get("response") or "").strip().upper()
            return answer.startswith("YES")
    except Exception:
        # If the classifier fails for any reason, default to False
        # so the chat still works normally
        return False
```

Breaking this down:

- This uses the same `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, and `/api/generate` endpoint that your existing `ollama_generate` function already uses, so the response shape is guaranteed to match.
- The `try/except` block means if Ollama is slow or unavailable, the function returns `False` and the chat continues normally rather than throwing an error.
- `answer.startswith("YES")` is defensive — even if the model adds a stray word, it will still work correctly.

---

**Step 2 — Add the memory fetch function**

Directly below `is_security_related`, add this second function:

```python
async def get_memory_context(query: str, top_k: int = 4) -> str:
    """
    Calls /memory/query and returns the retrieved chunks as a
    formatted string ready to inject into a prompt.
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(
                "http://localhost:8000/memory/query",
                json={"query": query, "top_k": top_k}
            )
            r.raise_for_status()
            data = r.json()

        if not data.get("results"):
            return ""

        chunks = []
        for result in data["results"]:
            chunks.append(f"[{result['title']} | {result['source']}]\n{result['text']}")
        return "\n\n".join(chunks)

    except Exception:
        # If memory fetch fails, return empty string so chat still works
        return ""
```

- `top_k: int = 4` means fetch 4 chunks by default — increase this if you want more context, decrease it if responses feel noisy.
- Like the classifier, this has a `try/except` so a memory failure never breaks the chat endpoint.
- The `for` loop labels each chunk with its title and source (e.g., `[CIS Docker Benchmark | docker-security]`) so the AI can cite where information came from.

> **Note:** `http://localhost:8000/memory/query` works here because both functions live inside the same `ingestion-api` container, where `localhost` refers to itself. This is an internal call that never goes through NGINX, so no API key is needed.

---

**Step 3 — Update `_chat_impl` to use both functions**

Find the `_chat_impl` function. It currently starts like this:

```python
async def _chat_impl(
    message: str,
    rid: str,
    detail_level: Optional[Literal["basic", "standard", "advanced"]] = None,
):
    t0 = time.time()

    # 1) Retrieve
    t_retr0 = time.time()
    sources = await asyncio.wait_for(retrieve_sources(message), timeout=RETRIEVE_TIMEOUT_S)
```

Add the security check right at the top before the retrieve step, so it looks like this:

```python
async def _chat_impl(
    message: str,
    rid: str,
    detail_level: Optional[Literal["basic", "standard", "advanced"]] = None,
):
    t0 = time.time()

    # 0) Security memory injection (optional enhancement)
    if await is_security_related(message):
        memory_context = await get_memory_context(message)
    else:
        memory_context = ""

    # 1) Retrieve
    t_retr0 = time.time()
    sources = await asyncio.wait_for(retrieve_sources(message), timeout=RETRIEVE_TIMEOUT_S)
```

Then find the `build_prompt` call a few lines down:

```python
    prompt = await asyncio.wait_for(
        asyncio.to_thread(build_prompt, message, sources, detail_level),
        timeout=PROMPT_TIMEOUT_S,
    )
```

Replace it with this:

```python
    enriched_message = (
        f"Security reference material:\n{memory_context}\n\nQuestion: {message}"
        if memory_context else message
    )
    prompt = await asyncio.wait_for(
        asyncio.to_thread(build_prompt, enriched_message, sources, detail_level),
        timeout=PROMPT_TIMEOUT_S,
    )
```

This prepends the memory chunks to the message before it reaches `build_prompt`. The rest of `_chat_impl` — the Ollama call, the timing, and the response shape — stays completely unchanged.

---

**5) — Add follow-up question suggestions (optional)**

This step adds a second Ollama call after the answer is generated. It looks at the question and answer and suggests 2-3 relevant follow-up questions the user might want to ask next. These are returned alongside the answer and displayed as clickable buttons in the Gradio UI.

Add this function directly below `get_memory_context`:

```python
async def get_followup_suggestions(message: str, answer: str) -> list:
    """
    Asks Ollama to suggest 2-3 follow-up questions based on the question and answer.
    Returns a list of question strings, or an empty list if it fails.
    """
    prompt = (
        "Based on this question and answer, suggest 2-3 short follow-up questions "
        "the user might want to ask next. Return only the questions as a JSON array "
        "of strings with no extra text, no explanation, and no markdown.\n\n"
        f"Question: {message}\n\nAnswer: {answer}"
    )
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
            )
            r.raise_for_status()
            response = (r.json().get("response") or "").strip()
            return json.loads(response)
    except Exception:
        return []
```

Make sure `import json` is at the top of `main.py` with the other imports, and that `OLLAMA_BASE_URL` and `OLLAMA_MODEL` are defined in the config section:

```python
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
```

Then update `_chat_impl` to call it after the answer is generated. Find the return block at the bottom of `_chat_impl`:

```python
    return {
        "answer": answer,
        "sources": sources,
        "_timing_ms": { ... },
        "_prompt_chars": len(prompt),
    }
```

Replace it with:

```python
    # 4) Follow-up suggestions
    followups = await get_followup_suggestions(message, answer)

    return {
        "answer": answer,
        "sources": sources,
        "followups": followups,
        "_timing_ms": {
            "retrieve": round(t_retr, 1),
            "prompt": round(t_pr, 1),
            "generate": round(t_llm, 1),
            "total": round(total, 1),
        },
        "_prompt_chars": len(prompt),
    }
```

Also update the `/chat` endpoint to pass `followups` through to the response. Find:

```python
        return {"answer": result["answer"], "sources": result["sources"]}
```

Replace with:

```python
        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "followups": result.get("followups", []),
        }
```

Once this is in place, the `/chat` response will include a `followups` field — a list of suggested questions. The Gradio UI update in Lesson 4.3 will display these as clickable buttons.

---

**6) — Restrict the assistant to security topics only (optional)**

By default the chatbot answers any question — security questions get memory injection, everything else goes through the normal RAG pipeline. This step adds a hard restriction so non-security questions receive a polite redirect instead of an answer.

This turns the general RAG assistant into a focused security tool. It is a good example of how you scope an AI assistant to a specific domain in production.

Add this function directly below `is_security_related`:

```python
async def enforce_security_scope(message: str) -> str | None:
    """
    If the message is not security-related, return a redirect message.
    If it is security-related, return None so the chat continues normally.
    """
    is_security = await is_security_related(message)
    if not is_security:
        return (
            "This assistant is focused on cybersecurity topics including secure coding, "
            "infrastructure security, and frameworks such as NIST, CIS, OWASP, and MITRE. "
            "Your question doesn't appear to be security-related. Please ask a cybersecurity "
            "question and I'll do my best to help."
        )
    return None
```

Then update `_chat_impl` to call it at the very top, before anything else runs. Find the start of `_chat_impl`:

```python
async def _chat_impl(
    message: str,
    rid: str,
    detail_level: Optional[Literal["basic", "standard", "advanced"]] = None,
):
    t0 = time.time()

    # 0) Security memory injection
    if await is_security_related(message):
```

Replace the security memory injection block with this:

```python
async def _chat_impl(
    message: str,
    rid: str,
    detail_level: Optional[Literal["basic", "standard", "advanced"]] = None,
):
    t0 = time.time()

    # 0) Scope enforcement — redirect non-security questions
    redirect = await enforce_security_scope(message)
    if redirect:
        return {
            "answer": redirect,
            "sources": [],
            "followups": [],
            "_timing_ms": {"retrieve": 0, "prompt": 0, "generate": 0, "total": 0},
            "_prompt_chars": 0,
        }

    # 1) Security memory injection
    memory_context = await get_memory_context(message)
```

Notice that because `enforce_security_scope` already calls `is_security_related` internally, you no longer need the separate `if await is_security_related` check — if execution reaches step 1, the message is already confirmed as security-related so you can call `get_memory_context` directly.

### Testing it

Rebuild and restart after making the change:

```bash
docker compose up -d --build ingestion-api
docker cp patches/ingestion-api/app/security_memory ingestion-api:/app/app/security_memory
docker compose restart ingestion-api
```

Test with a non-security question:

```bash
curl -s -X POST http://localhost:8088/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $EDGE_API_KEY" \
  -d '{"message": "What is the capital of France?"}' \
  | python -m json.tool
```

Expected response:

```json
{
  "answer": "This assistant is focused on cybersecurity topics...",
  "sources": [],
  "followups": []
}
```

Test with a security question to confirm it still works normally:

```bash
curl -s -X POST http://localhost:8088/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $EDGE_API_KEY" \
  -d '{"message": "What does OWASP say about broken access control?"}' \
  | python -m json.tool
```

> **Note:** This step reuses the same Ollama classifier from Step 1, so there is no additional model or service required. The only trade-off is one extra Ollama round trip per message to run the classification — on a small model like `llama3.2:1b` this adds roughly 5-10 seconds per request.

---

## 7) Security Note: Keep These Endpoints Behind the API Key Gate

The memory endpoints contain curated security reference material that took effort to assemble. Even though the documents themselves aren't secrets, in a real organization this kind of governance material would be access-controlled.

This repo already enforces authentication via an API key header checked by NGINX. Make sure your memory endpoints follow the same rules:

- **Do not** expose `ingestion-api` directly on a host port — traffic should always go through `edge-nginx`
- **Do not** disable or bypass the `X-API-Key` check for convenience during testing (use the `$EDGE_API_KEY` variable instead)

If you're unsure whether your endpoints are protected, re-read the NGINX config to confirm that all `/memory/*` paths require the key.

---

## Rebuild ingestion_api

```bash
docker compose up -d --build ingestion-api
docker cp patches/ingestion-api/app/security_memory ingestion-api:/app/app/security_memory
docker compose restart ingestion-api
```

## Test the Chat Endpoint

```bash
curl -s -X POST http://localhost:8088/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $EDGE_API_KEY" \
  -d '{"message": "What are the CIS Docker benchmark recommendations for running containers as root?"}' \
  | python -m json.tool
```

## If Chat Times Out

Add this to your `.env` file:

```
CHAT_TOTAL_TIMEOUT_S=300
OLLAMA_TIMEOUT_S=240
```

Then restart:

```bash
docker compose restart ingestion-api
```

You can also test via the Gradio UI at `http://localhost:7860` — it uses the `/chat` endpoint under the hood and handles the API key automatically.

## Checkpoint
You're done when all of the following are true:

- `/memory/health` returns `ok: true` through `http://localhost:8088`
- `/memory/query` returns relevant, non-empty results
- The base lab still works (chat, ingest, and retrieval are unaffected)

[Lesson 3](https://github.com/JacksonHolmes01/Training-Module-Production-Grade-RAG-Infrastructure-Qdrant/blob/main/lessons/04-security-memory/03-ide-integration.md)


---
*License: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) — Jackson Holmes*
