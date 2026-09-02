#!/usr/bin/env bash
# Wait until all 6 methods in rerun_baselines60_717f reach 90 videos and no
# generation process is running, then launch the 9.1 experiments.
set -uo pipefail
B=/mnt/cpfs/epic-user/hujiahui-20260612
ROOT=$B/kv-quant-result/Self-Forcing/rerun_baselines60_717f
METHODS=(RTN_INT2 RTN_INT4 KIVI_INT2 KIVI_INT4 QUAROT_KV_INT2 QUAROT_KV_INT4)
LOG=$ROOT/logs/run_91.log

echo "[wait] $(date) waiting for rerun to finish..." | tee "$LOG"
while true; do
  gen=$(pgrep -c -f "01_generate.py" 2>/dev/null || echo 0)
  all90=1
  for m in "${METHODS[@]}"; do
    n=$(find "$ROOT/videos/$m" -name 'prompt_*.mp4' 2>/dev/null | wc -l | tr -d ' ')
    if (( n < 90 )); then all90=0; fi
  done
  if (( gen == 0 )) && (( all90 == 1 )); then
    echo "[wait] $(date) rerun done (all 90), launching 9.1" | tee -a "$LOG"
    break
  fi
  sleep 120
done

# launch 9.1 (run_study then ablation) detached, logged
nohup bash "$B/Tempokv/scripts/paper_experiments/run_91.sh" all >> "$LOG" 2>&1 < /dev/null &
echo "[wait] run_91.sh launched pid=$!" | tee -a "$LOG"
