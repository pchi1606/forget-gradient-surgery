#!/usr/bin/env bash
# One-command verification of the FGS reproduction package.
#
#   bash reproduce.sh
#
# Step 1 recomputes every headline number from the archived per-seed
# JSONs (scripts/verify_reproduction.py) and compares them against
# expected/headline_numbers.json. Step 2 is a smoke end-to-end re-run of
# one configuration (Cora, seed 42: FGS with KS stopping + retrain)
# through the real pipeline code, checked against the archived values.
# The Planetoid data files are fetched from their public mirror first
# (the package does not redistribute the raw datasets; see README.md).
#
# Exits nonzero if any check fails. Total runtime is a few minutes.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
echo "== environment =="
"$PY" -c "import sys, numpy, scipy; print('python', sys.version.split()[0]); print('numpy', numpy.__version__); print('scipy', scipy.__version__)"

echo "== step 1: verify headline numbers from archived per-seed results =="
"$PY" scripts/verify_reproduction.py

echo "== step 2: fetch Planetoid data (public mirror) =="
"$PY" scripts/fetch_data.py cora

echo "== step 3: smoke end-to-end re-run (Cora, seed 42, FGS KS + retrain) =="
PYTHONPATH="scripts:code" "$PY" scripts/smoke_rerun.py

echo "REPRODUCE: OK"
