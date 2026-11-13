"""
qdrant_init.py: one-time setup, run by the qdrant-init container.

This is the ONLY code that ever holds Qdrant's admin API key. It waits
for Qdrant, then creates this tenant's collection if it does not exist
yet. After it exits, nothing running holds the admin key.
"""

import os
import sys
import time

import httpx

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
APP_TENANT = os.getenv("APP_TENANT", "lab")
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))
COLLECTION = f"tenant_{APP_TENANT}"
QDRANT_CA_CERT = os.getenv("QDRANT_CA_CERT", "")

with open("/run/secrets/qdrant_api_key", encoding="utf-8") as f:
    headers = {"api-key": f.read().strip()}

with httpx.Client(base_url=QDRANT_URL, headers=headers, timeout=10,
                  verify=QDRANT_CA_CERT or True) as c:
    for _ in range(60):
        try:
            if c.get("/readyz").status_code == 200:
                break
        except httpx.HTTPError:
            pass
        time.sleep(2)
    else:
        sys.exit("Qdrant did not become ready.")

    if c.get(f"/collections/{COLLECTION}").status_code == 200:
        print(f"Collection {COLLECTION} already exists.")
        sys.exit(0)

    r = c.put(f"/collections/{COLLECTION}",
              json={"vectors": {"size": EMBED_DIM, "distance": "Cosine"}})
    r.raise_for_status()
    print(f"Created collection {COLLECTION} ({EMBED_DIM} dimensions, cosine).")
