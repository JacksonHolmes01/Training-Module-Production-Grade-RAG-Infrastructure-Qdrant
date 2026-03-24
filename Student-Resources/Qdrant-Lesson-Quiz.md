# Lab 1 — Qdrant RAG Infrastructure: Quiz Questions

Self-graded quiz for all 10 lessons. Answers are at the bottom of each lesson section.

---

## Lesson 1 — Lab Overview

**Q1.** What does RAG stand for and what problem does it solve?

**Q2.** In this lab, only two ports are exposed to your laptop. What are they and what service runs on each one?

**Q3.** True or False: Ollama can access the Qdrant database directly to look up answers.

**Q4.** What is the role of the `ingestion-api` (FastAPI) container in the RAG pipeline?

**Q5.** Why are internal services like Qdrant kept off the host network instead of being exposed directly?

<details>
<summary>Answers</summary>

1. Retrieval-Augmented Generation. It solves the problem of LLMs answering from training data alone — RAG retrieves relevant documents first and grounds the answer in real sources.
2. Port 7860 (Gradio UI) and port 8088 (NGINX gateway).
3. False. Ollama only receives a prompt string. It has no access to Qdrant or any other service.
4. FastAPI orchestrates the full RAG pipeline — it validates requests, calls Qdrant for retrieval, builds the prompt, calls Ollama for generation, and returns the structured response.
5. Exposing databases directly increases attack surface, bypasses application-level validation, and makes consistent authentication harder to enforce. Internal-only networking is a standard production security pattern.

</details>

---

## Lesson 2 — Setup & First Boot

**Q1.** What command do you run to start all services in the background and rebuild local images?

**Q2.** Why should you never commit your `.env` file to GitHub?

**Q3.** What is the minimum recommended RAM to run this lab comfortably, and why does this lab require more memory than a typical web app?

**Q4.** You run `docker compose up -d --build` and the UI loads at localhost:7860 but chat fails. What is the most likely reason?

**Q5.** What command quickly generates a cryptographically secure random string suitable for use as an `EDGE_API_KEY`?

<details>
<summary>Answers</summary>

1. `docker compose up -d --build`
2. The `.env` file contains secrets like `EDGE_API_KEY`. Committing it to GitHub would expose those secrets publicly.
3. 16 GB RAM is recommended. This lab runs a local LLM (Ollama) and a vector database (Qdrant) simultaneously, both of which are memory-intensive.
4. The Ollama model hasn't been pulled yet. Chat requires a model to be downloaded — this happens in Lesson 7.
5. `python -c "import secrets; print(secrets.token_urlsafe(32))"` or `openssl rand -hex 32`

</details>

---

## Lesson 3 — Compose Architecture & Resource Limits

**Q1.** What are the three top-level sections in `docker-compose.yml` that define how services communicate and store data?

**Q2.** Why does this lab set `mem_limit: 8g` on Qdrant and Ollama?

**Q3.** What two Docker volumes are used in this lab and what does each one store?

**Q4.** If you run `docker compose down`, is your ingested data deleted? What would you run to delete everything including data?

**Q5.** You have Docker Desktop set to 8 GB total memory and both Qdrant and Ollama are capped at 8 GB each. What problem will you likely experience?

<details>
<summary>Answers</summary>

1. `services:` (container definitions), `networks:` (how containers communicate), and `volumes:` (persistent data storage).
2. Without limits, a single container can consume all available RAM, causing the machine to slow to a crawl or other processes to be killed by the OS. Resource limits mirror real production deployments.
3. `qdrant_data` stores Qdrant objects and vectors. `ollama_data` stores downloaded Ollama models.
4. No — `docker compose down` stops containers but preserves volumes. Run `./bin/reset_all.sh` to delete all data including volumes.
5. The two containers will compete for the same 8 GB total, causing slow performance, container restarts, or out-of-memory errors. Either increase Docker Desktop's memory allocation or reduce the `mem_limit` values.

</details>

---

## Lesson 4 — Edge Authentication with NGINX

**Q1.** What HTTP status code does NGINX return when the `X-API-Key` header is missing entirely?

**Q2.** What HTTP status code does NGINX return when the `X-API-Key` header is present but contains the wrong value?

**Q3.** What is "defense in depth" and how does this lab implement it for API authentication?

**Q4.** Where does NGINX get the value of the API key to compare against incoming requests?

**Q5.** True or False: If you disable the API key check in NGINX for convenience during testing, the system is still protected because FastAPI also checks the key.

<details>
<summary>Answers</summary>

1. 401 Unauthorized — the key is missing.
2. 403 Forbidden — the key is wrong.
3. Defense in depth means using multiple independent security layers so that a failure in one doesn't expose the system. This lab implements it by checking the API key in both NGINX and inside FastAPI's `require_api_key()` function.
4. NGINX reads `EDGE_API_KEY` from the `.env` file, which is injected into the NGINX config template at container startup.
5. False — while FastAPI does check the key, disabling the NGINX check removes a layer of defense and is bad practice. Both layers exist for a reason.

</details>

---

## Lesson 5 — Qdrant Schema & Vectorization

**Q1.** In this lab, what generates the embeddings before they are stored in Qdrant?

**Q2.** What is the difference between a vector and a payload in Qdrant?

**Q3.** True or False: Adding a new field like `tags` to Qdrant requires a schema migration similar to a relational database.

**Q4.** After adding `tags: list[str] = []` to the `ArticleIn` Pydantic model, what command do you run to apply the change?

**Q5.** Why is it important to re-verify retrieval after making a schema change and rebuilding the API?

<details>
<summary>Answers</summary>

1. The ingestion API generates embeddings internally using a SentenceTransformer model — not Qdrant and not a separate service.
2. A vector is a list of floating point numbers representing the semantic meaning of the text. A payload is the metadata stored alongside the vector (title, url, tags, text, etc.).
3. False. Qdrant stores whatever JSON payload you send — there is no schema migration required. Only the API validation model (Pydantic) needs to be updated.
4. `docker compose up -d --build ingestion-api`
5. Schema changes can break ingestion or retrieval if the API contract is misaligned with stored data. Re-verifying confirms that semantic search still returns correct results after the change.

</details>

---

## Lesson 6 — Ingestion API

**Q1.** What HTTP status code does the API return when you submit a document with an invalid URL, and what does this code mean?

**Q2.** What is the purpose of the `/debug/retrieve` endpoint and when should you use it?

**Q3.** What are the four steps that happen when you POST a document to `/ingest`?

**Q4.** True or False: A 422 validation error means something went wrong with the server. You should check the Qdrant logs to debug it.

**Q5.** What does `./bin/smoke_test.sh` test and why is it useful?

<details>
<summary>Answers</summary>

1. 422 Unprocessable Entity. It means the request body failed validation — the data shape was incorrect (e.g., invalid URL format). This is expected and intentional behavior.
2. `/debug/retrieve` tests retrieval from Qdrant without involving Ollama. Use it to confirm that ingestion worked and vectors are being retrieved correctly before debugging generation.
3. (1) NGINX checks the API key, (2) FastAPI validates the JSON schema, (3) the document is sent to Qdrant for storage, (4) the vectorizer generates an embedding so the document becomes semantically searchable.
4. False. A 422 means the client sent invalid data — it is caught by Pydantic validation in FastAPI. Check the request body, not the server logs.
5. It tests proxy health, authentication, ingestion, retrieval, prompt construction, and chat generation in sequence. It provides strong evidence that all layers of the system are working correctly.

</details>

---

## Lesson 7 — RAG Chat

**Q1.** What is the correct debugging order when `/chat` returns a poor or empty answer?

**Q2.** What does the `/debug/prompt` endpoint return and why is it useful?

**Q3.** What does `RAG_TOP_K` control and what is the tradeoff between a low value (2) and a high value (10)?

**Q4.** True or False: Increasing `RAG_TOP_K` always improves answer quality.

**Q5.** What does "grounding" mean in the context of this RAG system and how is it enforced in the prompt?

<details>
<summary>Answers</summary>

1. Check retrieval first (`/debug/retrieve`), then prompt construction (`/debug/prompt`), then Ollama generation (`/debug/ollama`). Never debug generation before confirming retrieval works.
2. It returns the exact prompt string that would be sent to Ollama — including the system instruction, user question, and retrieved source excerpts — without actually calling the LLM. It lets you confirm the grounding is correct.
3. `RAG_TOP_K` controls how many documents are retrieved per query. Low values (2) are fast and focused but may miss context. High values (10) provide more coverage but add noise and increase prompt size.
4. False. More documents can add noise and irrelevant content, making answers less focused. The right value depends on your dataset and question type.
5. Grounding means the LLM is instructed to answer using only the provided source documents rather than its training data. It is enforced by including the retrieved excerpts in the prompt alongside an explicit instruction: "Answer using ONLY the provided sources."

</details>

---

## Lesson 8 — Gradio Chat UI

**Q1.** What does the Sources section in the Gradio UI show and why does it matter?

**Q2.** The Gradio UI talks to NGINX, not directly to FastAPI. Why is this important?

**Q3.** You open the UI at localhost:7860 and ask a question but get no answer. What is the first thing you check?

**Q4.** True or False: You can use the Gradio UI to directly query the Qdrant database.

**Q5.** What questions from the cybersecurity dataset are good to test with and why should your test questions relate to the dataset?

<details>
<summary>Answers</summary>

1. It shows the Qdrant documents that were retrieved and used to generate the answer, including title, URL, and distance score. It matters because it makes the answer auditable — you can verify where information came from.
2. It ensures authentication is always enforced. If Gradio called FastAPI directly, it could bypass the NGINX API key check.
3. Check if the proxy is running: `curl -i http://localhost:8088/proxy-health`. Then check if Ollama has a model pulled.
4. False. Gradio only calls the `/chat` endpoint via NGINX. It has no direct access to Qdrant.
5. Questions like "What is the CIA triad?", "Why is exposing databases risky?", or "What is supply chain risk in AI?". Test questions should relate to the dataset because RAG retrieves from what was ingested — asking about unrelated topics will return poor results since there are no matching documents.

</details>

---

## Lesson 9 — Operations & Troubleshooting

**Q1.** What is the first thing you should do when something in the lab stops working?

**Q2.** What is the difference between `docker compose down` and `./bin/reset_all.sh`?

**Q3.** In Failure Drill A (stop Ollama), why does `/debug/retrieve` still work but `/chat` fails?

**Q4.** In Failure Drill C (stop text2vec-transformers), what specific part of the pipeline breaks and why?

**Q5.** What does `docker stats` show and when is it useful during this lab?

<details>
<summary>Answers</summary>

1. Check the logs — `docker compose logs <service> --tail=200`. Do not guess. Logs show the root cause directly.
2. `docker compose down` stops all containers but preserves data volumes (Qdrant data and Ollama models survive). `./bin/reset_all.sh` deletes everything including volumes — a full wipe.
3. `/debug/retrieve` only involves Qdrant and the vectorizer, not Ollama. Retrieval still works because those services are running. `/chat` fails at the generation step because it needs Ollama to produce an answer.
4. The embedding step breaks. Qdrant cannot generate vectors for new ingestion and cannot embed new queries for search because text2vec-transformers is the embedding service that powers both.
5. `docker stats` shows real-time CPU and memory usage per container. It is useful during ingestion to confirm Ollama is actively processing, and when diagnosing memory pressure or resource limit issues.

</details>

---

## Lesson 10 — Conclusion

**Q1.** Name the five architectural layers of this system in order from edge to generation.

**Q2.** Why is retrieval and generation kept as two separate steps rather than having the LLM query the database directly?

**Q3.** How would the production equivalent of this lab differ from what you built? Name at least three differences.

**Q4.** What debugging discipline did this lab teach and why is it important for multi-service systems?

**Q5.** True or False: The security controls in this lab (API keys, internal networking, schema validation) are just for learning purposes and would not apply to real production AI systems.

<details>
<summary>Answers</summary>

1. NGINX (edge) → FastAPI (orchestration) → Qdrant (retrieval) → Prompt construction → Ollama (generation).
2. Keeping them separate gives you observability, replaceability, and security. You can debug each layer independently, swap out the LLM or database without touching the other, and each service has a clearly bounded responsibility.
3. Any three of: Qdrant replaced by a managed cloud vector database service; secrets stored in a secret manager (Vault, AWS SSM) instead of .env; API keys replaced with OAuth/JWT tokens; observability via Prometheus and Grafana instead of docker stats; GPU-backed LLM instead of local Ollama; rate limiting at the edge; centralized logging.
4. Isolate layers before diagnosing — check retrieval before generation, check generation before orchestration. This prevents wasted time debugging the wrong layer and applies to any multi-service system.
5. False. These are real production patterns. Internal-only networking, layered authentication, schema validation, and bounded LLM prompting are all used in production AI systems at scale.

</details>
