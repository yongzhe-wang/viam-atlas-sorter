#!/bin/sh
# Module entrypoint (meta.json "entrypoint"). viam-server runs it with the module socket path as $1.
#
# Everything slow happens here, once, and is kept in $VIAM_MODULE_DATA (set by viam-server, kept across module
# versions), so redeploys start fast:
#   venv/   Python deps, CPU-only torch. Reinstalled only when requirements.txt changes (marker keyed on its hash).
#   cache/  model caches (HF_HOME, TORCH_HOME, XDG_CACHE_HOME), so the ~600 MB CLIP weights are downloaded here
#           during install, not on the first command. Fetched again only when the CLIP model line changes.
set -e
cd "$(dirname "$0")"
BASE="${VIAM_MODULE_DATA:-$(pwd)}"
VENV="$BASE/venv"
CACHE="$BASE/cache"
mkdir -p "$CACHE"
export HF_HOME="$CACHE/huggingface" TORCH_HOME="$CACHE/torch" XDG_CACHE_HOME="$CACHE"
# torch AND torchvision from the same CPU index: open_clip pulls torchvision from PyPI otherwise, built for a
# different torch, and the first CLIP import dies with "operator torchvision::nms does not exist" (seen on sfhack4)
TORCH_INSTALL="torch torchvision --index-url https://download.pytorch.org/whl/cpu"
if command -v sha256sum >/dev/null 2>&1; then HASH_CMD="sha256sum"; else HASH_CMD="shasum -a 256"; fi
DEPS_HASH=$( (cat requirements.txt; echo "$TORCH_INSTALL") | $HASH_CMD | cut -c1-16)
MARKER="$VENV/.installed-$DEPS_HASH"
if [ ! -f "$MARKER" ]; then
  [ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
  "$VENV/bin/pip" install --quiet --upgrade pip
  # CPU-only torch first: the default Linux wheel pulls CUDA libraries (>2 GB), too slow for module startup.
  # Remove any mismatched pair left by an earlier install so pip can't keep a PyPI torchvision as "already satisfied".
  "$VENV/bin/pip" uninstall --quiet -y torch torchvision >/dev/null 2>&1 || true
  "$VENV/bin/pip" install --quiet $TORCH_INSTALL
  "$VENV/bin/pip" install --quiet -r requirements.txt
  # Prove the stack loads before marking the venv good: a torch/torchvision mismatch only shows up on import
  # ("operator torchvision::nms does not exist"). No marker on failure, so the next start installs again.
  if ! "$VENV/bin/python" -c "import torch, torchvision, open_clip; torchvision.ops.nms(torch.zeros((1, 4)), torch.zeros(1), 0.5)"; then
    echo "run.sh: torch / torchvision / open_clip do not load together after install; venv not marked installed:" >&2
    "$VENV/bin/pip" list 2>/dev/null | grep -iE '^(torch|torchvision|open[-_]clip)' >&2 || true
    exit 1
  fi
  rm -f "$VENV"/.installed-*
  touch "$MARKER"
fi
# CLIP weights: keyed on the model/pretrained line in sorter/clip_embed.py, so a model change downloads again
MODEL_KEY=$(grep -E '^MODEL, PRETRAINED *=' sorter/clip_embed.py | $HASH_CMD | cut -c1-16)
WEIGHTS_MARKER="$CACHE/.clip-weights-$MODEL_KEY"
if [ ! -f "$WEIGHTS_MARKER" ]; then
  "$VENV/bin/python" -c "from sorter import clip_embed; clip_embed.embed_text('download the weights')"
  rm -f "$CACHE"/.clip-weights-*
  touch "$WEIGHTS_MARKER"
fi
exec "$VENV/bin/python" -m module.main "$@"
