# CIS Docker Benchmark (Sample Excerpt)

**Recommended tags:** `cis`, `docker`, `container`

## Container hardening patterns
- Avoid running as root unless required.
- Prefer read-only root filesystem for services that can support it.
- Minimize Linux capabilities.
- Avoid mounting the Docker socket into containers.
- Use explicit networks and minimal exposed ports.

## Why this matters (lab context)
A lab may expose ports for teaching, but production would:
- restrict bind addresses
- remove unnecessary published ports
- enforce least privilege between services
