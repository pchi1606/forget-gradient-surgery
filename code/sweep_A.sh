#!/bin/bash
# Sweep A: GCN new seeds (full new protocol) + SAGE backbone (cora, citeseer).
cd ~/workspace/graph-unlearning
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for ds in cora citeseer pubmed; do
  for s in 2024 31337; do
    echo "=== A: gcn $ds seed $s ==="
    BACKBONE=gcn python3 experiment.py $ds $s || exit 1
  done
done
for ds in cora citeseer; do
  for s in 42 7 123 2024 31337; do
    echo "=== A: sage $ds seed $s ==="
    BACKBONE=sage python3 experiment.py $ds $s || exit 1
  done
done
echo "SWEEP_A_DONE"
