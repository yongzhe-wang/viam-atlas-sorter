#!/bin/bash
# Run a Python program forever, restarting it when it dies.
# WHY: the Viam SDK's native layer can abort the whole process during a reconnect (exit 134, SIGABRT), which
# Python cannot catch. run_trials.py and sorter/worker.py resume from data/arm_state.json and the Store.
# Usage: scripts/supervise.sh scripts/run_trials.py 10      |   scripts/supervise.sh -m sorter.worker
# Env: PY (python to use, default python3), VIAM_* and MONGODB_URI as usual. Ctrl-C twice to stop.
PY=${PY:-python3}
export RUN_ID=${RUN_ID:-$(date +%Y%m%d-%H%M%S)}  # same run id across restarts
cd "$(dirname "$0")/.." || exit 1
for attempt in $(seq 1 20); do
  "$PY" -u "$@"
  code=$?
  # 0 = finished normally; 130 = Ctrl-C
  if [ $code -eq 0 ] || [ $code -eq 130 ]; then exit $code; fi
  echo "supervise: exited with $code (restart $attempt/20 in 5 s)"
  sleep 5
done
echo "supervise: giving up after 20 restarts"
exit 1
