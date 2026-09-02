#!/usr/bin/env bash
# Wait until run_study (from run_91.sh study) is done, then launch
# predictor_ablation on all 4 GPUs.  Logged to rerun_baselines60_717f/logs.
set -uo pipefail
B=/mnt/cpfs/epic-user/hujiahui-20260612
LOG=$B/kv-quant-result/Self-Forcing/rerun_baselines60_717f/logs/run_91_study.log
A_LOG=$B/kv-quant-result/Self-Forcing/rerun_baselines60_717f/logs/run_91_ablation.log

echo "[after_study] $(date) waiting for run_study to finish..." | tee -a "$A_LOG"
# run_study logs "输出：...observation/{decompose.json,...}" at the very end.
while ! grep -q "输出：.*observation" "$LOG" 2>/dev/null; do
  sleep 30
  # abort if the study process died without finishing
  if ! pgrep -f "run_91.sh study" >/dev/null 2>&1; then
    echo "[after_study] $(date) run_study process gone; check $LOG" | tee -a "$A_LOG"
    sleep 5
  fi
done
echo "[after_study] $(date) run_study done, launching predictor_ablation" | tee -a "$A_LOG"
nohup bash "$B/Tempokv/scripts/paper_experiments/run_91.sh" ablation >> "$A_LOG" 2>&1 < /dev/null &
echo "[after_study] ablation launched pid=$!" | tee -a "$A_LOG"
