# Author: Prabhu Rangaraju
"""Full 30-config re-run of the FROZEN pipeline (with M4 disjointness fix and
the utu baseline) on one machine, into results/per_seed_rerun/.

Purpose (test-31 fix round):
  - same-machine wall-clock timings for every method (the stored per-seed
    JSONs mix fgs_attack/fgs_noproj timings from the overnight M4 rerun
    machine with retrain/fgs/gnndelete timings from the original runs);
  - determinism check: attack/util must reproduce the stored values
    bit-for-bit (pure NumPy, seeded RNGs);
  - utu values for all 30 configs (already covered by compute_utu.py;
    kept here for consistency).
Does NOT touch results/per_seed/; comparison happens after.
"""
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import experiment

OUT = os.path.join(HERE, "results", "per_seed_rerun")
os.makedirs(OUT, exist_ok=True)
experiment.RES_DIR = OUT

SEEDS = [7, 42, 123, 999, 2024, 2025, 5555, 7777, 31337, 31338]
CONFIGS = ([("cora", s) for s in SEEDS] + [("citeseer", s) for s in SEEDS] +
           [("pubmed", s) for s in SEEDS])

if __name__ == "__main__":
    only = sys.argv[1:]
    for ds, seed in CONFIGS:
        if only and f"{ds}:{seed}" not in only:
            continue
        print(f"[full-rerun] {ds} seed={seed}", flush=True)
        experiment.run(ds, seed)
        shutil.rmtree(os.path.join(HERE, "results", "cache"), ignore_errors=True)
