#!/bin/sh
# Module entrypoint (meta.json "entrypoint"). viam-server runs it with the module socket path as $1.
#
# The virtualenv lives in $VIAM_MODULE_DATA (set by viam-server, kept across module versions), not next to
# this script: every `viam module reload-local` unpacks into a new versioned directory, so a local venv meant
# reinstalling CPU torch (~10 min on sfhack4) on each redeploy. The install marker is keyed on a hash of
# requirements.txt + the torch install method, so it reinstalls only when dependencies change.
set -e
cd "$(dirname "$0")"
VENV="${VIAM_MODULE_DATA:-$(pwd)}/venv"
TORCH_INSTALL="torch --index-url https://download.pytorch.org/whl/cpu"
if command -v sha256sum >/dev/null 2>&1; then HASH_CMD="sha256sum"; else HASH_CMD="shasum -a 256"; fi
DEPS_HASH=$( (cat requirements.txt; echo "$TORCH_INSTALL") | $HASH_CMD | cut -c1-16)
MARKER="$VENV/.installed-$DEPS_HASH"
if [ ! -f "$MARKER" ]; then
  [ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
  "$VENV/bin/pip" install --quiet --upgrade pip
  # CPU-only torch first: the default Linux wheel pulls CUDA libraries (>2 GB), too slow for module startup
  "$VENV/bin/pip" install --quiet $TORCH_INSTALL
  "$VENV/bin/pip" install --quiet -r requirements.txt
  rm -f "$VENV"/.installed-*
  touch "$MARKER"
fi
exec "$VENV/bin/python" -m module.main "$@"
