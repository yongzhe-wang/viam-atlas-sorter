#!/bin/sh
# Module entrypoint (meta.json "entrypoint"). viam-server runs it with the module socket path as $1.
# First start creates a virtualenv next to this script and installs requirements.txt (torch: several minutes).
set -e
cd "$(dirname "$0")"
if [ ! -f .venv/.installed ]; then
  python3 -m venv .venv
  .venv/bin/pip install --quiet --upgrade pip
  .venv/bin/pip install --quiet -r requirements.txt
  touch .venv/.installed
fi
exec .venv/bin/python -m module.main "$@"
