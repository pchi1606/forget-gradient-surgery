#!/bin/bash
# Sweep B: GCN add-on (new methods on existing seeds) + SAGE pubmed +
# forget-fraction sensitivity (GCN backbone, 1%/10%/20%).
cd ~/workspace/graph-unlearning
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for ds in cora citeseer pubmed; do
  for s in 42 7 123; do
    echo "=== B: addon gcn $ds seed $s ==="
    python3 experiment_addon.py $ds $s || exit 1
  done
done
for s in 42 7 123 2024 31337; do
  echo "=== B: sage pubmed seed $s ==="
  BACKBONE=sage python3 experiment.py pubmed $s || exit 1
done
for fr in 0.01 0.10 0.20; do
  for ds in cora citeseer pubmed; do
    for s in 42 7 123; do
      echo "=== B: ff gcn $ds seed $s frac $fr ==="
      BACKBONE=gcn FORGET_FRAC=$fr python3 experiment_ff.py $ds $s || exit 1
    done
  done
done
echo "SWEEP_B_DONE"
