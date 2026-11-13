# Lesson 9: Defense 4: Ingest Screening and Cleanup

The prompt layer neutralizes poisoned documents every time they are retrieved. It is better not to store them at all.

The ingest guard scores every document **before it is stored**, using the same patterns with a lower threshold (2 instead of 3). Normal documents describe things; they do not give orders to an AI.

---

## Turn it on

```bash
bin/guard.sh ingest on
bin/guard.sh status
```

All four guards should now be `true`.

---

## The poison is already inside

Turning on the ingest guard does nothing about documents stored **before** it existed. Your three poisoned documents are still in the database from Lesson 4.

`/debug/scan` scores every stored document with the ingest rules:

```bash
EDGE_API_KEY=$(grep -E '^EDGE_API_KEY=' .env | cut -d= -f2-)
curl -s -H "X-API-Key: $EDGE_API_KEY" http://localhost:8088/debug/scan | python3 -m json.tool
```

### What you will see

```json
{
    "scanned": 15,
    "flagged": [
        {"id": "8c9ee5c3-...", "title": "Password Reset Procedure", "score": 8,
         "reasons": ["fake-role-marker", "injected-instruction-header", "conditional-trigger", "concealment"]},
        {"id": "4b6aa8c3-...", "title": "Reporting Phishing Emails", "score": 8,
         "reasons": ["fake-role-marker", "addressed-to-ai", "markdown-image"]},
        {"id": "6c770425-...", "title": "Multi-Factor Authentication Guide", "score": 6,
         "reasons": ["fake-role-marker", "conditional-trigger", "invisible-characters"]}
    ]
}
```

All three poisoned documents are flagged. None of the 12 legitimate documents is. Qdrant IDs are random UUIDs, so yours will differ. The full IDs are in your output (shortened here).

---

## Remove them, as the administrator

Can the chat API delete them? No, and that is by design. `/chat` uses the **reader token**, which Qdrant only lets read. Try it:

```bash
bin/qdrant.sh reader POST /collections/tenant_lab/points/delete '{"points":["<one of your IDs>"]}'
```

```
403
{
  "status": {
    "error": "Forbidden: Write access to collection tenant_lab is required"
```

What about the **writer token** that `/ingest` uses? Here Qdrant differs from a SQL database. In PostgreSQL you can grant INSERT without DELETE. Qdrant's token access levels are coarser: `r` (read) or `rw` (read and write), and "write" includes deleting points. So the writer token *could* delete documents in `tenant_lab`, even though the API's code never asks it to. What keeps the damage small:

- The writer token is limited to **one collection**. It cannot see, change, or delete any other tenant's collection, and it cannot create, delete, or reconfigure collections.
- Only `/ingest` holds it. The chat path, the one the attacker talks to, never does.

Knowing exactly what each credential *can* do, not just what your code *does* with it, is the point of least privilege.

Removing data is an administrator's job. Use the admin key and the IDs from **your** scan:

```bash
bin/qdrant.sh admin POST '/collections/tenant_lab/points/delete?wait=true' '{"points":["<id 1>","<id 2>","<id 3>"]}'
```

```
200
{
  "result": {
    "operation_id": 16,
    "status": "completed"
  },
```

Scan again:

```bash
curl -s -H "X-API-Key: $EDGE_API_KEY" http://localhost:8088/debug/scan
```

```
{"scanned":12,"flagged":[]}
```

---

## Try to plant them again (expected to fail)

```bash
python3 attacks/run_attacks.py --plant --only control
```

### What you will see

```
Planting poisoned documents (this is the attacker's step)...
  REJECTED  Password Reset Procedure  (fake-role-marker, injected-instruction-header, conditional-trigger, concealment)
  REJECTED  Reporting Phishing Emails  (fake-role-marker, addressed-to-ai, markdown-image)
  REJECTED  Multi-Factor Authentication Guide  (fake-role-marker, conditional-trigger, invisible-characters)
```

The invisible-character trick is caught twice over: the zero-width characters are removed before matching, and their presence is itself a signal.

---

## Check for false positives

A document that **talks about** prompt injection must still be accepted:

```bash
curl -s -X POST http://localhost:8088/ingest \
  -H "X-API-Key: $EDGE_API_KEY" -H "Content-Type: application/json" \
  -d '{"title": "Prompt Injection Explained", "text": "Prompt injection is an attack where text tries to override an AI model'"'"'s instructions. Defenses include input screening, output filtering, and least privilege."}'
```

```
{"status":"ok","id":16,"tenant":"lab"}
```

Describing an attack is not the same as performing one. The patterns look for text **addressed to** an AI ("note to the assistant", "when the user asks"), not text **about** AI.

---

## Run the full suite one last time

```bash
python3 attacks/run_attacks.py
```

With the poison gone and all four guards on, every attack fails and the control question is answered, with one possible exception: `obfuscated-override` may still succeed on some runs. The next lesson explains why that one attack is different.

Next: [Operating the Defenses](10-operations.md)
