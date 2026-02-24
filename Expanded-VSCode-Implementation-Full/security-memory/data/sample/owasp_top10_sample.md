# OWASP Top 10 (Sample Excerpt)

**Recommended tags:** `owasp`, `appsec`, `web`

## A01: Broken Access Control
Access control prevents users from acting outside intended permissions.

Common failures:
- missing authorization checks on endpoints
- insecure direct object reference (IDOR)
- forced browsing to admin-only routes
- “deny by default” not enforced

Typical mitigations:
- enforce server-side authorization checks
- use least privilege roles
- log and monitor access failures
