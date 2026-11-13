# Lesson 9: Layer 5: Encrypted, Verified Traffic Between Services

Since Part 3, every request to Qdrant needs a key or a signed token. But those requests travel over the private Docker network as **plain HTTP**. Anyone who can watch that traffic sees the tokens in the `Authorization` header and every document the chatbot retrieves.

"It never leaves the Docker network" is a reasonable argument on one laptop. It is a weak one in production, where services run on different machines, networks are shared, and an attacker inside any container on the network could watch or alter traffic.

---

## See the current state

With Layers 1 to 4 on, ask Qdrant for its status over plain HTTP from inside the API container:

```bash
bin/harden.sh 4
docker exec rag-api python -c "import httpx; print(httpx.get('http://qdrant:6333/readyz').text)"
```

```
all shards are ready
```

Qdrant answers unencrypted requests. So does every request the API makes, tokens included.

---

## The certificates

`bin/setup_secrets.sh` created a small **private certificate authority** (CA) and used it to sign a certificate for Qdrant:

```bash
openssl verify -CAfile secrets/tls_ca.crt secrets/qdrant_server.crt
openssl x509 -in secrets/qdrant_server.crt -noout -text | grep -A1 "Alternative Name"
```

```
secrets/qdrant_server.crt: OK
            X509v3 Subject Alternative Name:
                DNS:qdrant
```

- The CA is the thing the API **trusts**. Anything the CA signed is accepted
- Qdrant's certificate is valid for the name `qdrant`, which is exactly the hostname the API connects to
- `secrets/tls_ca.key` (the CA's private key) is never mounted into any container. Whoever holds it can create certificates the API would trust

---

## Read the layer

Open `hardening/05-internal-tls.yml`.

### Qdrant turns on TLS

```yaml
  qdrant:
    environment:
      QDRANT__SERVICE__ENABLE_TLS: "true"
      QDRANT__TLS__CERT: /run/secrets/qdrant_server_crt
      QDRANT__TLS__KEY: /run/secrets/qdrant_server_key
```

Qdrant reads any setting from an environment variable named `QDRANT__<SECTION>__<SETTING>`. With TLS on, port 6333 speaks **only** HTTPS. There is no plain-HTTP fallback to disable, because there is none left.

The two `TLS` variables hold **paths** to secret files, not the secrets themselves. The audit knows the difference: a value that starts with `/run/secrets/` points at a secret, it is not one.

### The API verifies Qdrant

```yaml
  ingestion-api:
    environment:
      QDRANT_URL: "https://qdrant:6333"
      QDRANT_CA_CERT: /run/secrets/tls_ca_crt
```

`ingestion-api/app/db.py` passes `QDRANT_CA_CERT` to the HTTP client as the **only** certificate authority it trusts. `qdrant-init` gets the same two settings.

| Client setting | Encrypted? | Checks the certificate? | Checks the hostname? |
|---|:-:|:-:|:-:|
| `http://` (Layers 0 to 4) | no | no | no |
| `https://`, certificate checking turned off | yes | no | no |
| `https://`, trusting only the lab CA (this layer) | yes | yes | yes |

Only the last row stops an attacker who can intercept traffic from pretending to be the database. Encryption without verification just means you have an encrypted conversation with whoever answered.

---

## Apply it

```bash
bin/harden.sh 5
bin/smoke_test.sh
```

## Verify the connection is encrypted and verified

Make three requests from inside the API container: plain HTTP, HTTPS trusting the lab CA, and HTTPS trusting only the usual public authorities:

```bash
docker exec rag-api python -c "
import httpx
for url, v, label in [('http://qdrant:6333/readyz', True, 'plain http'),
                      ('https://qdrant:6333/readyz', '/run/secrets/tls_ca_crt', 'https, lab CA'),
                      ('https://qdrant:6333/readyz', True, 'https, public CAs')]:
    try:
        r = httpx.get(url, verify=v, timeout=5); print(f'{label:<18} {r.status_code} {r.text}')
    except Exception as e:
        print(f'{label:<18} refused: {str(e)[:70]}')
"
```

```
plain http         refused: illegal request line
https, lab CA      200 all shards are ready
https, public CAs  refused: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable
```

- **Plain HTTP is gone.** Qdrant no longer understands it
- **The lab CA works.** This is what the API and `bin/qdrant.sh` use
- **A public CA does not.** No outside authority vouches for this certificate, and none should

## Verify an impostor is refused (expected to fail)

Hostname checking matters too. Connect to the same Qdrant, with the right CA, but by its IP address instead of its name:

```bash
IP=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' rag-qdrant)
docker exec rag-api python -c "
import httpx
try:
    httpx.get('https://$IP:6333/readyz', verify='/run/secrets/tls_ca_crt', timeout=5); print('CONNECTED (bad)')
except Exception as e:
    print('refused:', str(e)[:110])
"
```

```
refused: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: IP address mismatch, certificate is not valid for '172.18.0.3'
```

The certificate says `qdrant`, so the client refuses anything else, even the right server under a different name. An attacker with a certificate for some other name gets the same answer.

## Audit

```bash
python3 bin/audit.py
```

The score does not change: TLS is not on the checklist, because the checklist looks at containers, not connections. Not everything important fits on one scorecard.

Next: [Layer 6: Scan and Pin Your Images](10-scan-and-pin.md)
