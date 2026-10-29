#!/bin/sh
# Module entrypoint (meta.json "entrypoint"). viam-server runs it with the module socket path as $1.
# First start creates a virtualenv next to this script and installs requirements.txt (torch: several minutes).
set -e
cd "$(dirname "$0")"
if [ ! -f .venv/.installed ]; then
  python3 -m venv .venv
  .venv/bin/pip install --quiet --upgrade pip
  # CPU-only torch first: the default Linux wheel pulls CUDA libraries (>2 GB), too slow for module startup
  .venv/bin/pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
  .venv/bin/pip install --quiet -r requirements.txt
  touch .venv/.installed
fi
exec .venv/bin/python -m module.main "$@"
