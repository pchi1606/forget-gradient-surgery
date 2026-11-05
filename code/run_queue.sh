#!/bin/bash
# Queue runner: executes tasks sequentially, logging to a file.
# Each line of the task file: "<kind> <backbone> <dataset> <seed> [frac]"
#   kind=main -> BACKBONE=<backbone> python3 experiment.py <dataset> <seed>
#   kind=ff   -> BACKBONE=<backbone> FORGET_FRAC=<frac> python3 experiment_ff.py <dataset> <seed>
# Usage: run_queue.sh <taskfile> <logfile>
cd ~/workspace/graph-unlearning
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
TASKFILE="$1"
LOGFILE="$2"
while read -r kind bb ds seed frac; do
  [[ -z "$kind" || "$kind" == \#* ]] && continue
  echo "=== $(date +%H:%M:%S) START $kind bb=$bb ds=$ds seed=$seed frac=$frac ===" >> "$LOGFILE"
  if [[ "$kind" == "main" ]]; then
    BACKBONE="$bb" python3 experiment.py "$ds" "$seed" >> "$LOGFILE" 2>&1
  elif [[ "$kind" == "ff" ]]; then
    BACKBONE="$bb" FORGET_FRAC="$frac" python3 experiment_ff.py "$ds" "$seed" >> "$LOGFILE" 2>&1
  elif [[ "$kind" == "noproj" ]]; then
    # fgs_noproj addon on an already-written per-seed JSON (idempotent)
    BACKBONE="$bb" python3 experiment_addon3.py "results/per_seed/${bb}_${ds}_seed${seed}.json" >> "$LOGFILE" 2>&1
  fi
  echo "=== $(date +%H:%M:%S) END $kind bb=$bb ds=$ds seed=$seed frac=$frac rc=$? ===" >> "$LOGFILE"
done < "$TASKFILE"
echo "QUEUE_DONE $(date +%H:%M:%S)" >> "$LOGFILE"
