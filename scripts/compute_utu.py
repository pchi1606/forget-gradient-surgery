#!/usr/bin/env python3
"""Unlink-only (UtU) baseline, computed from the harness.

UtU is the structural-deletion-only control: take the base model exactly
as trained (no parameter update of any kind) and score it on the
post-deletion graph (adjacency built from the remain edges). Its attack
gap to retrain (Table 4) is also iterate 0 of the archived fixed-step
trajectories, which scripts/verify_reproduction.py checks against
results/fixedstep.json without any re-run; this script recomputes the
same quantity from the live pipeline code in code/.

Usage:
    python3 scripts/compute_utu.py --dataset cora --seed 42
    python3 scripts/compute_utu.py --all   # 30 headline configs (slow)

Prints the UtU attack/utility AUCs, the archived retrain reference, and
the resulting gaps.
"""
import argparse
import sys

import numpy as np

from pipeline import archived_per_seed, attack_auc, build_config, util_auc


def utu_for(dataset, seed):
    cfg = build_config(dataset, seed)
    arch = archived_per_seed(dataset, seed)
    a = attack_auc(cfg, cfg["base"], cfg["remain_e"])
    u = util_auc(cfg, cfg["base"], cfg["remain_e"])
    ra, ru = arch["methods"]["retrain"]["attack"], arch["methods"]["retrain"]["util"]
    print(f"[{dataset} seed={seed}] UtU attack {a:.4f} util {u:.4f} | "
          f"retrain attack {ra:.4f} util {ru:.4f} | "
          f"gaps attack {a - ra:+.4f} util {u - ru:+.4f}")
    return a - ra, u - ru


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cora")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    if args.all:
        seeds = (42, 7, 123, 2024, 2025, 31337, 31338, 5555, 7777, 999)
        for ds in ("cora", "citeseer", "pubmed"):
            ga, gu = [], []
            for s in seeds:
                a, u = utu_for(ds, s)
                ga.append(a)
                gu.append(u)
            print(f"UtU {ds}: attack gap {np.mean(ga):+.4f} +/- "
                  f"{np.std(ga, ddof=1):.4f}; util gap {np.mean(gu):+.4f} "
                  f"+/- {np.std(gu, ddof=1):.4f} (n=10)")
    else:
        utu_for(args.dataset, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
