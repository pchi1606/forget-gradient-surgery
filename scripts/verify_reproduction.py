#!/usr/bin/env python3
# Author: Prabhu Rangaraju
"""Verify the FGS reproduction: check headline manuscript numbers against
the regenerated results/per_seed/*.json files.

Checks:
  1. all 30 per-seed JSONs present (3 datasets x 10 seeds)
  2. every file contains fgs_attack (attack-calibrated variant) and fgs_noproj
  3. KS-stopped FGS / retrain time ratios, median over 10 seeds
  4. ACCURACY HEADLINE: CiteSeer fgs attack gap to retrain ~ +0.037
  5. ACCURACY HEADLINE: Cora fgs_noproj utility gap to retrain ~ -0.048
  6. ACCURACY HEADLINE: paired fgs - fgs_attack attack gap on CiteSeer ~ +0.028

Exits 0 with 'ALL CHECKS PASSED' on success, 1 otherwise.
"""
import glob
import json
import os
import statistics
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PER_SEED = os.path.join(BASE, "results", "per_seed")
SEEDS = [7, 42, 123, 2024, 2025, 31337, 31338, 5555, 7777, 999]
DATASETS = ["cora", "citeseer", "pubmed"]

# Filename convention: seeds 7/42/123 use <ds>_seed<N>.json (legacy),
# others use gcn_<ds>_seed<N>.json. Both are valid.
def seed_path(ds, s):
    if s in (7, 42, 123):
        return os.path.join(PER_SEED, f"{ds}_seed{s}.json")
    return os.path.join(PER_SEED, f"gcn_{ds}_seed{s}.json")

failures = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def main():
    files = {}
    for ds in DATASETS:
        for s in SEEDS:
            p = seed_path(ds, s)
            if os.path.isfile(p):
                files[(ds, s)] = json.load(open(p))
    check("all 30 per-seed JSONs present", len(files) == 30,
          f"found {len(files)}/30")

    missing_attack = [k for k, d in files.items()
                      if "fgs_attack" not in d.get("methods", {})]
    check("fgs_attack present in every file", not missing_attack,
          f"missing in {missing_attack}" if missing_attack else "30/30")

    missing_noproj = [k for k, d in files.items()
                      if "fgs_noproj" not in d.get("methods", {})]
    check("fgs_noproj present in every file", not missing_noproj,
          f"missing in {missing_noproj}" if missing_noproj else "30/30")

    # KS-stopped FGS / retrain medians per dataset (timing)
    # Expected from the 30-run sweep (original hardware for fgs/retrain).
    expected_medians = {"cora": 0.394, "citeseer": 0.396, "pubmed": 0.370}
    for ds in DATASETS:
        ratios = [files[(ds, s)]["methods"]["fgs"]["s"]
                  / files[(ds, s)]["methods"]["retrain"]["s"] for s in SEEDS
                  if (ds, s) in files]
        med = statistics.median(ratios)
        exp = expected_medians[ds]
        check(f"KS-stopped FGS/retrain median [{ds}] ~{exp}",
              abs(med - exp) <= 0.05, f"got {med:.3f}")

    # ACCURACY HEADLINE 1: CiteSeer fgs attack gap to retrain
    # Manuscript Table 3: +0.037 (mean over 10 seeds)
    cs_gaps = []
    for s in SEEDS:
        d = files.get(("citeseer", s))
        if d:
            m = d["methods"]
            cs_gaps.append(m["fgs"]["attack"] - m["retrain"]["attack"])
    if cs_gaps:
        mean_gap = statistics.mean(cs_gaps)
        check("ACCURACY: CiteSeer fgs attack gap ~ +0.037",
              abs(mean_gap - 0.037) <= 0.015, f"got {mean_gap:+.3f}")

    # ACCURACY HEADLINE 2: Cora fgs_noproj utility gap to retrain
    # Manuscript Table 3: -0.048 (mean over 10 seeds)
    cora_gaps = []
    for s in SEEDS:
        d = files.get(("cora", s))
        if d:
            m = d["methods"]
            cora_gaps.append(m["fgs_noproj"]["util"] - m["retrain"]["util"])
    if cora_gaps:
        mean_gap = statistics.mean(cora_gaps)
        check("ACCURACY: Cora fgs_noproj utility gap ~ -0.048",
              abs(mean_gap - (-0.048)) <= 0.015, f"got {mean_gap:+.3f}")

    # ACCURACY HEADLINE 3: paired fgs - fgs_attack attack gap on CiteSeer
    # Manuscript: +0.028 (the attack-calibrated gain)
    paired_diffs = []
    for s in SEEDS:
        d = files.get(("citeseer", s))
        if d:
            m = d["methods"]
            paired_diffs.append(m["fgs"]["attack"] - m["fgs_attack"]["attack"])
    if paired_diffs:
        mean_diff = statistics.mean(paired_diffs)
        check("ACCURACY: CiteSeer paired fgs - fgs_attack ~ +0.028",
              abs(mean_diff - 0.028) <= 0.015, f"got {mean_diff:+.3f}")

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {failures}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
