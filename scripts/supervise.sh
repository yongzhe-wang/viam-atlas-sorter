#!/bin/bash
# Run a Python program forever, restarting it when it dies or hangs.
# WHY: the Viam SDK's native layer can abort the whole process during a reconnect (exit 134, SIGABRT), which
# Python cannot catch; and a trial once hung > 10 min inside one SDK call. The child touches $ALIVE_FILE every
# 5 s (pick_place_auto.Rig._alive); if it goes stale for WATCHDOG_S the child is killed and restarted.
# run_trials.py and sorter/worker.py resume from data/arm_state.json and the Store.
# Usage: scripts/supervise.sh scripts/run_trials.py 10      |   scripts/supervise.sh -m sorter.worker
# Env: PY (python, default python3), WATCHDOG_S (default 120), VIAM_* and MONGODB_URI as usual.
PY=${PY:-python3}
WATCHDOG_S=${WATCHDOG_S:-120}
export RUN_ID=${RUN_ID:-$(date +%Y%m%d-%H%M%S)}  # same run id across restarts
cd "$(dirname "$0")/.." || exit 1
pid=""
trap 'if [ -n "$pid" ]; then kill "$pid" 2>/dev/null; fi; exit 130' INT TERM
mtime() { stat -f %m "$1" 2>/dev/null || stat -c %Y "$1" 2>/dev/null; }
for attempt in $(seq 1 20); do
  ALIVE_FILE="$(mktemp -t alive.XXXXXX)"
  export ALIVE_FILE
  "$PY" -u "$@" &
  pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    sleep 10
    now=$(date +%s)
    mt=$(mtime "$ALIVE_FILE")
    mt=${mt:-$now}
    if [ $((now - mt)) -gt "$WATCHDOG_S" ]; then
      echo "supervise: no sign of life for $((now - mt)) s, killing $pid"
      kill -9 "$pid" 2>/dev/null
    fi
  done
  wait "$pid"
  code=$?
  pid=""
  rm -f "$ALIVE_FILE"
  # 0 = finished normally; 130 = Ctrl-C
  if [ $code -eq 0 ] || [ $code -eq 130 ]; then exit $code; fi
  echo "supervise: exited with $code (restart $attempt/20 in 5 s)"
  sleep 5
done
echo "supervise: giving up after 20 restarts"
exit 1
