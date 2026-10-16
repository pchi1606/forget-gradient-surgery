#!/usr/bin/env bash
# Author: Prabhu Rangaraju
# One-command reproduction of the FGS graph-unlearning experiment results.
#
# Usage:  bash reproduce.sh
# Run from this directory. Requires Python 3.10+ and pip.
#
# What it does:
#   1. installs Python dependencies (requirements.txt: numpy, scipy)
#   2. fetches the Planetoid datasets (skipped if already bundled)
#   3. runs the GCN sweep: 30 runs (Cora/CiteSeer/PubMed x 10 seeds),
#      each training a 2-layer GCN link predictor, forgetting 5% of
#      train edges, and timing retrain vs FGS (KS-stopped) vs
#      attack-calibrated FGS; plus the no-projection ablation
#      (experiment_addon3.py) for the paired comparison
#   4. verifies headline manuscript numbers against expected/ and
#      reports PASS/FAIL
#
# Expected runtime: ~7 hours single-threaded on a modern CPU
# (pure NumPy/SciPy; no GPU needed or used).
# All stochastic steps use fixed seeds; reruns are deterministic.
set -euo pipefail
cd "$(dirname "$0")"

echo "================================================================"
echo " STEP 1/4: dependencies"
echo "================================================================"
pip install -r requirements.txt || echo "WARNING: pip install failed; continuing with installed dependencies."

echo "================================================================"
echo " STEP 2/4: fetch Planetoid datasets (skipped if present)"
echo "================================================================"
bash fetch_data.sh

echo "================================================================"
echo " STEP 3/4: GCN sweep (30 runs + no-projection ablation)"
echo "================================================================"
bash run_sweep.sh

echo "================================================================"
echo " STEP 4/4: verify headline numbers"
echo "================================================================"
python3 scripts/verify_reproduction.py
