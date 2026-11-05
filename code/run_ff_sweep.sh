#!/usr/bin/env bash
# M3 fraction sweep: 5 seeds x 3 fracs x 3 datasets = 45 runs.
# Reruns with M4-fixed code (disjoint stop_neg).
# Usage: bash run_ff_sweep.sh
set -uo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

run_one() {
  local ds=$1 seed=$2 frac=$3
  echo "=== [$(date +%H:%M:%S)] ff $ds seed $seed frac $frac ==="
  FORGET_FRAC=$frac BACKBONE=gcn python3 experiment_ff.py "$ds" "$seed" 2>&1 | tail -3
}

export -f run_one
CONFIGS=()
for ds in cora citeseer pubmed; do
  for seed in 42 123 2024 2025 999; do
    for frac in 0.01 0.10 0.20; do
      CONFIGS+=("$ds $seed $frac")
    done
  done
done

printf "%s\n" "${CONFIGS[@]}" | xargs -P 2 -n 3 bash -c 'run_one "$0" "$1" "$2"'
echo "ff sweep complete."
