#!/usr/bin/env bash
# GCN sweep driver: 3 datasets x 10 seeds = 30 runs.
# Mirrors the sweep behind the manuscript's results.
# Each run: BACKBONE=gcn python3 experiment.py <dataset> <seed>
# Seeds 7/42/123 use the legacy filename convention (<ds>_seed<N>.json);
# the seven gcn_-prefixed seeds use gcn_<ds>_seed<N>.json.
# All were run under the identical protocol (see README).
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for ds in cora citeseer pubmed; do
  for seed in 7 42 123 2024 2025 31337 31338 5555 7777 999; do
    echo "=== [$(date +%H:%M:%S)] gcn $ds seed $seed ==="
    BACKBONE=gcn python3 experiment.py "$ds" "$seed"
  done
done
# No-projection ablation (fgs_noproj) for the paired Table 3 comparison:
for ds in cora citeseer pubmed; do
  for seed in 7 42 123 2024 2025 31337 31338 5555 7777 999; do
    echo "=== [$(date +%H:%M:%S)] fgs_noproj $ds seed $seed ==="
    python3 experiment_addon3.py "$ds" "$seed"
  done
done
echo "sweep complete: results in results/per_seed/"
