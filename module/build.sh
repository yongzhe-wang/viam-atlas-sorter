#!/bin/sh
# Package the module for `viam module upload`: $DIST/archive.tar.gz (default module/dist) with run.sh at its root
# (meta.json "build"). Includes this machine's calibration (data/workspace.json, data/jaw_calib.json) only if
# present: those are specific to one camera mount, so another team should build after calibrating their own.
#
# The code comes from the last commit (git archive HEAD), never from the working tree: VIAM CENTERAL "only build
# from a fully committed state" (an archive once carried other agents' half-done edits). BUILD_INFO.json in the
# archive names the commit, and the module reports it in status. ALLOW_DIRTY=1 packs the working tree instead and
# lists the uncommitted files in BUILD_INFO.json. Data that is not in git (object library, calibration) always
# comes from this checkout's data/.
set -e
cd "$(dirname "$0")/.."
DIST="${DIST:-module/dist}"
STAGE="$DIST/stage"
CODE="module sorter scripts/pick_place_auto.py scripts/jaw_calib.py meta.json requirements.txt"
commit=$(git rev-parse HEAD)
dirty=$(git status --porcelain -- $CODE 2>/dev/null || true)
rm -rf "$DIST" && mkdir -p "$STAGE/module" "$STAGE/scripts" "$STAGE/data"
if [ "${ALLOW_DIRTY:-0}" = 1 ]; then
  SRC=.
  if [ -n "$dirty" ]; then
    echo "WARNING (ALLOW_DIRTY=1): uncommitted changes go into this archive:"
    echo "$dirty"
  fi
else
  SRC="$DIST/src"
  mkdir -p "$SRC"
  git archive HEAD $CODE | tar -x -C "$SRC"
  echo "building commit ${commit}"
  if [ -n "$dirty" ]; then
    echo "note: these uncommitted changes are NOT in this archive (commit them, or ALLOW_DIRTY=1):"
    echo "$dirty"
  fi
fi
cp "$SRC/module/run.sh" "$STAGE/run.sh"
# viam-server reads the entrypoint from meta.json inside local tarballs (reload-local fails without it)
cp "$SRC/meta.json" "$STAGE/meta.json"
# Every module/*.py: listing files by name shipped a tarball without live_map.py and the module died on import
cp "$SRC"/module/*.py "$STAGE/module/"
cp "$SRC/requirements.txt" "$STAGE/requirements.txt"
cp -R "$SRC/sorter" "$STAGE/sorter"
cp "$SRC/scripts/pick_place_auto.py" "$SRC/scripts/jaw_calib.py" "$STAGE/scripts/"
rm -rf "$DIST/src"
for f in data/workspace.json data/jaw_calib.json; do
  if [ -f "$f" ]; then cp "$f" "$STAGE/data/"; fi
done
# Object library (views + their crop images): the module seeds it into $VIAM_MODULE_DATA so it can tell
# look-alikes apart without Atlas. Built from this machine's data/views.jsonl (not in git).
# BUILD_INFO.json: which commit this is, so a running module can say exactly what it is.
STAGE="$STAGE" COMMIT="$commit" DIRTY="$dirty" ALLOW_DIRTY="${ALLOW_DIRTY:-0}" python3 - <<'EOF'
import json, os, pathlib, shutil, time
src, stage = pathlib.Path("data"), pathlib.Path(os.environ["STAGE"])
dst = stage / "library"
views = src / "views.jsonl"
if not views.exists():
    print("no data/views.jsonl: the module ships without an object library")
else:
    (dst / "img").mkdir(parents=True, exist_ok=True)
    n = imgs = 0
    for line in views.read_text().splitlines():
        if not line.strip():
            continue
        n += 1
        path = (json.loads(line).get("jpeg") or {}).get("path")
        if path and (src / path).exists():
            shutil.copy2(src / path, dst / path)
            imgs += 1
    shutil.copy2(views, dst / "views.jsonl")
    print(f"bundled library: {n} views, {imgs} images")
dirty = [l for l in os.environ["DIRTY"].splitlines() if l.strip()] if os.environ["ALLOW_DIRTY"] == "1" else []
(stage / "BUILD_INFO.json").write_text(json.dumps({
    "commit": os.environ["COMMIT"], "dirty": dirty, "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}))
EOF
find "$STAGE" -name __pycache__ -prune -exec rm -rf {} +
chmod +x "$STAGE/run.sh"
# COPYFILE_DISABLE: macOS tar otherwise adds an AppleDouble ._<name> file next to every file (they shipped to the robot)
COPYFILE_DISABLE=1 tar -czf "$DIST/archive.tar.gz" -C "$STAGE" .
echo "built $DIST/archive.tar.gz"
# Unpack it and check the module can start from the copy (a file missing from the tarball fails here, not on the robot)
"${PYTHON:-python3}" module/check_archive.py "$DIST/archive.tar.gz"
