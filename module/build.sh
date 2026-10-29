#!/bin/sh
# Package the module for `viam module upload`: module/dist/archive.tar.gz with run.sh at its root (meta.json "build").
# Includes this machine's calibration (data/workspace.json, data/jaw_calib.json) only if present: those are
# specific to one camera mount, so another team should build after calibrating their own.
set -e
cd "$(dirname "$0")/.."
rm -rf module/dist && mkdir -p module/dist/stage/module module/dist/stage/scripts module/dist/stage/data
cp module/run.sh module/dist/stage/run.sh
# viam-server reads the entrypoint from meta.json inside local tarballs (reload-local fails without it)
cp meta.json module/dist/stage/meta.json
cp module/main.py module/sorter_service.py module/dist/stage/module/
cp requirements.txt module/dist/stage/requirements.txt
cp -R sorter module/dist/stage/sorter
cp scripts/pick_place_auto.py scripts/jaw_calib.py module/dist/stage/scripts/
for f in data/workspace.json data/jaw_calib.json; do
  if [ -f "$f" ]; then cp "$f" module/dist/stage/data/; fi
done
find module/dist/stage -name __pycache__ -prune -exec rm -rf {} +
chmod +x module/dist/stage/run.sh
tar -czf module/dist/archive.tar.gz -C module/dist/stage .
echo "built module/dist/archive.tar.gz"
