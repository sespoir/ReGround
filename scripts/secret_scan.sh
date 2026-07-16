#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PATTERN="(sk-[A-Za-z0-9_-]{12,}|(api[_-]?key|access[_-]?token|client[_-]?secret)[[:space:]]*[:=][[:space:]]*[\"']?[A-Za-z0-9_+./=-]{16,}|(^|[^0-9a-fA-F])[0-9a-fA-F]{32}([^0-9a-fA-F]|$))"
if rg -n -i --hidden \
  --glob '!.git/**' \
  --glob '!.env' \
  --glob '!*.example' \
  "${PATTERN}" .; then
  echo "Potential secret detected." >&2
  exit 1
fi
echo "Secret scan passed."
