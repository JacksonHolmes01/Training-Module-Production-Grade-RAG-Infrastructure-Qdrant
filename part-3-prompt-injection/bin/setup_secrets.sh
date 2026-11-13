#!/usr/bin/env bash
# Creates every secret this lab needs. Safe to run more than once:
# existing secrets are kept, never overwritten.
#
#   secrets/qdrant_api_key.txt      Qdrant's admin key (only qdrant-init uses it)
#   secrets/qdrant_config.yaml      Qdrant's settings: the admin key + JWT access control on
#   secrets/qdrant_writer_token.txt signed token: read/write on ONE collection
#   secrets/qdrant_reader_token.txt signed token: read-only on ONE collection
set -euo pipefail
cd "$(dirname "$0")/.."

gen() { openssl rand -base64 48 | tr -d '/+=\n' | cut -c1-48; }

mkdir -p secrets
if [[ -s secrets/qdrant_api_key.txt ]]; then
  echo "  kept     secrets/qdrant_api_key.txt"
else
  gen > secrets/qdrant_api_key.txt
  rm -f secrets/qdrant_writer_token.txt secrets/qdrant_reader_token.txt
  echo "  created  secrets/qdrant_api_key.txt"
fi

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "  created  .env (from .env.example)"
fi
TENANT="$(grep -E '^APP_TENANT=' .env | cut -d= -f2-)"; TENANT="${TENANT:-lab}"

# Qdrant reads this file as config/local.yaml. Keeping the key in a file
# (not an environment variable) keeps it out of `docker inspect`.
printf 'service:\n  api_key: "%s"\n  jwt_rbac: true\n' "$(cat secrets/qdrant_api_key.txt)" > secrets/qdrant_config.yaml

# Tokens are JSON Web Tokens signed with the admin key (HMAC-SHA256).
# Qdrant checks the signature, then allows only what the "access" claim lists.
make_token() {  # make_token <file> <access: r|rw>
  if [[ -s "$1" ]]; then echo "  kept     $1"; return; fi
  python3 - "$(cat secrets/qdrant_api_key.txt)" "tenant_${TENANT}" "$2" > "$1" <<'PY'
import base64, hashlib, hmac, json, sys
key, collection, access = sys.argv[1:4]
b64 = lambda b: base64.urlsafe_b64encode(b).rstrip(b"=").decode()
header = b64(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
claims = b64(json.dumps({"access": [{"collection": collection, "access": access}]}).encode())
sig = b64(hmac.new(key.encode(), f"{header}.{claims}".encode(), hashlib.sha256).digest())
print(f"{header}.{claims}.{sig}")
PY
  echo "  created  $1 ($2 on tenant_${TENANT})"
}
make_token secrets/qdrant_writer_token.txt rw
make_token secrets/qdrant_reader_token.txt r
chmod 600 secrets/*.txt secrets/*.yaml

if ! grep -qE '^EDGE_API_KEY=.{32,}' .env; then
  key="$(gen | cut -c1-40)"
  sed -i.bak "s|^EDGE_API_KEY=.*|EDGE_API_KEY=${key}|" .env && rm -f .env.bak
  echo "  created  EDGE_API_KEY in .env"
else
  echo "  kept     EDGE_API_KEY in .env"
fi
chmod 600 .env
# Parts 3, 4, and 5 share one model volume, so models download only once
docker volume create secure-ai-qdrant_ollama_data >/dev/null
echo "  ready    model volume secure-ai-qdrant_ollama_data"
echo "Secrets ready."
