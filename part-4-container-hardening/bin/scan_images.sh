#!/usr/bin/env bash
# Scan every image this lab runs for known vulnerabilities (CVEs) with Trivy.
#
# Usage:
#   bin/scan_images.sh              HIGH and CRITICAL only, one summary line per image
#   bin/scan_images.sh --details    also list each vulnerability
#
# Trivy runs in its own container. It never gets access to the Docker socket:
# registry images are scanned by name, and images built here are exported to
# a file first (docker save) and scanned from that file.
set -euo pipefail
cd "$(dirname "$0")/.."

TRIVY=aquasec/trivy:0.76.0
mkdir -p scan-results
DETAILS="${1:-}"

run_trivy() {  # run_trivy <label> <trivy args...>
  local label="$1"; shift
  docker run --rm -v secure-ai-trivy-cache:/root/.cache/ -v "$PWD/scan-results":/scan "$TRIVY" \
    image --quiet --scanners vuln --severity HIGH,CRITICAL --format json -o "/scan/${label}.json" "$@"
  python3 - "$label" "$DETAILS" <<'PY'
import json, sys, collections
label, details = sys.argv[1], sys.argv[2]
data = json.load(open(f"scan-results/{label}.json"))
vulns = [v for r in data.get("Results", []) for v in (r.get("Vulnerabilities") or [])]
sev = collections.Counter(v["Severity"] for v in vulns)
fixable = sum(1 for v in vulns if v.get("FixedVersion"))
print(f"  {label:<14} CRITICAL {sev.get('CRITICAL',0):>3}   HIGH {sev.get('HIGH',0):>3}   (fix available for {fixable})")
if details == "--details":
    for v in sorted(vulns, key=lambda v: (v["Severity"] != "CRITICAL", v["PkgName"]))[:25]:
        print(f"      {v['Severity']:<8} {v['VulnerabilityID']:<18} {v['PkgName']} {v.get('InstalledVersion','')} -> {v.get('FixedVersion') or 'no fix yet'}")
PY
}

echo "Scanning (the first run downloads Trivy's vulnerability database, about 1 minute)..."
run_trivy qdrant    qdrant/qdrant:v1.20.0-unprivileged
run_trivy ollama    ollama/ollama:0.40.6
run_trivy nginx     nginxinc/nginx-unprivileged:1.28.2-alpine

for svc in ingestion-api gradio-ui; do
  img=$(docker compose config --format json | python3 -c "import json,sys; c=json.load(sys.stdin); print(c['services']['$svc'].get('image') or c['name']+'-$svc')")
  docker save "$img" -o "scan-results/${svc}.tar"
  run_trivy "$svc" --input "/scan/${svc}.tar"
  rm -f "scan-results/${svc}.tar"
done
echo "Full reports: scan-results/*.json"
