#!/usr/bin/env bash
# Send one request to Qdrant as a given identity, from inside the lab network.
# Qdrant has no published port, so this runs the request from the API container.
#
# Usage: bin/qdrant.sh <admin|writer|reader|none> <METHOD> <path> [json-body]
# Examples:
#   bin/qdrant.sh reader GET /collections/tenant_lab
#   bin/qdrant.sh reader DELETE /collections/tenant_lab        (should be refused)
#   bin/qdrant.sh admin  POST /collections/tenant_lab/points/delete '{"points":["<id>"]}'
set -euo pipefail
cd "$(dirname "$0")/.."
WHO="${1:?usage: bin/qdrant.sh <admin|writer|reader|none> <METHOD> <path> [json]}"
METHOD="${2:?method}"; P="${3:?path}"; BODY="${4:-}"
case "$WHO" in
  admin)  CRED="$(cat secrets/qdrant_api_key.txt)" ;;
  writer) CRED="$(cat secrets/qdrant_writer_token.txt)" ;;
  reader) CRED="$(cat secrets/qdrant_reader_token.txt)" ;;
  none)   CRED="" ;;
  *) echo "identity must be admin, writer, reader, or none"; exit 1 ;;
esac
URL="$(docker exec rag-api printenv QDRANT_URL)"
CA=""; docker exec rag-api test -f /run/secrets/qdrant_ca.crt 2>/dev/null && CA=/run/secrets/qdrant_ca.crt
# The credential travels on stdin (line 1), so it never appears in a process list.
{ printf '%s\n' "$CRED"; printf '%s' "$BODY"; } | docker exec -i rag-api python -c '
import json, sys, httpx
method, url, path, ca = sys.argv[1:5]
cred = sys.stdin.readline().strip()
body = sys.stdin.read()
h = {"Authorization": "Bearer " + cred} if cred else {}
if body:
    h["Content-Type"] = "application/json"
r = httpx.request(method, url + path, headers=h, content=body or None, verify=ca or True, timeout=30)
print(r.status_code)
try:
    print(json.dumps(r.json(), indent=2)[:4000])
except Exception:
    print(r.text[:2000])
' "$METHOD" "$URL" "$P" "$CA"
