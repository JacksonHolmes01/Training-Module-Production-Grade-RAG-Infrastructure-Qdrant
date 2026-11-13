# Lesson 2: Setup

Part 3 is a copy of the Lab 2 chatbot (the main lab in the root of this repository) with the defenses added, all switched off. It also tightens the database:

| Lab 2 | Part 3 |
|---|---|
| Qdrant published on ports 6333 and 6334, with no API key | Qdrant has **no published ports** and refuses any request without a key |
| Anyone who reaches Qdrant can do anything | The API holds two **signed tokens**, each limited to one collection: read-only for chat, read/write for ingest |
| | The **admin key** is used once, by a setup container, then nothing running holds it |

---

## Step 1: Stop Lab 2

```bash
cd ~/Training-Module-Production-Grade-RAG-Infrastructure-Qdrant
docker compose down
```

## Step 2: Move into Part 3

```bash
cd ~/Training-Module-Production-Grade-RAG-Infrastructure-Qdrant/part-3-prompt-injection
ls
```

```
README.md  attacks  bin  data  docker-compose.yml  gradio-ui  ingestion-api  lessons  nginx  secrets
```

These things are new compared to Lab 2:

| Item | Purpose |
|---|---|
| `attacks/` | The attack cases, the poisoned documents, and `run_attacks.py`, which runs every attack and scores the result |
| `ingestion-api/app/guards.py` | The four defenses |
| `bin/guard.sh` | Turns a defense on or off and restarts the API |
| `bin/qdrant.sh` | Sends one request to Qdrant as `admin`, `writer`, `reader`, or `none`, so you can test what each identity may do |
| `ingestion-api/app/qdrant_init.py` | Creates the collection with the admin key, once |

## Step 3: Secrets, build, and start

```bash
bin/setup_secrets.sh
docker compose up -d --build
```

`setup_secrets.sh` creates Qdrant's admin key, signs the reader and writer tokens with it, generates the NGINX API key, and creates a model volume that Parts 3, 4, and 5 share.

Check that the setup container did its job and exited:

```bash
docker logs rag-qdrant-init
```

```
Created collection tenant_lab (768 dimensions, cosine).
```

### Prove the database is locked

```bash
bin/qdrant.sh none GET /collections
bin/qdrant.sh reader GET /collections
bin/qdrant.sh reader DELETE /collections/tenant_lab
```

```
401
Must provide an API key or an Authorization bearer token
200
{"result":{"collections":[{"name":"tenant_lab"}]}, ...}
403
{"status":{"error":"Forbidden: ..."}}
```

No key: refused. The reader can see its own collection (and only that one), but cannot delete it.

## Step 4: Models and sample data

```bash
bin/pull_models.sh
bin/ingest_sample.sh
```

The first time, this downloads about 1.6 GB. Parts 4 and 5 reuse the same download.

```
Total: 12  Success: 12  Failed: 0
```

## Step 5: Confirm every defense is off

```bash
bin/guard.sh status
```

```
GUARD_INPUT=false
GUARD_INGEST=false
GUARD_PROMPT=false
GUARD_OUTPUT=false
```

The API also reports which guards are on:

```bash
EDGE_API_KEY=$(grep -E '^EDGE_API_KEY=' .env | cut -d= -f2-)
curl -s -H "X-API-Key: $EDGE_API_KEY" http://localhost:8088/health | python3 -m json.tool | grep -A5 guards
```

```
    "guards": {
        "input": false,
        "ingest": false,
        "prompt": false,
        "output": false
    },
```

---

## One setting that makes experiments fair

```
OLLAMA_TEMPERATURE=0
```

**Temperature** controls randomness. At `0`, the model picks its most likely answer every time, so the same attack against the same setup gives the same result. That lets you compare "before" and "after" honestly.

Next: [Direct Attacks: Talking the Model Out of Its Rules](03-direct-attacks.md)
