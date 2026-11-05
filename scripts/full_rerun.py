#!/usr/bin/env python3
"""Frozen-pipeline re-run entry point.

Re-runs specified (dataset, seed, method) configurations through the
frozen harness in code/ (via scripts/pipeline.py's exact rng replay of
code/experiment.py) and reports each method's attack/utility AUC next to
the archived per-seed values in results/per_seed/.

This is the script credited in the manuscript with (a) the seed-pooling
exchangeability check -- re-running the three earlier-batch seeds per
dataset (7, 42, 123) under the frozen pipeline and confirming every
method's attack/utility reproduces the archived values within
3 x 10^-3 (floating-point noise) -- and (b) the same-machine timing
ratios of Table 7 (single-machine re-run on the same ten seeds).

Usage:
    python3 scripts/full_rerun.py --dataset cora --seeds 42 7 \
        --methods retrain fgs
    python3 scripts/full_rerun.py --all-headline   # slow: everything

Methods: retrain, finetune, grad_ascent, fgs, fgs_margin, fgs_attack,
fgs_noproj, gnndelete. Requires the Planetoid data files (see
scripts/fetch_data.py; reproduce.sh fetches them automatically).
Timings printed are this machine's wall-clock and will differ from the
archived ones; the AUC agreement is the check that matters.
"""
import argparse
import sys
import time

import numpy as np

from pipeline import (EPOCHS, HID, LR, FGS_ETA, archived_per_seed,
                      attack_auc, build_config, draw_retain_negatives,
                      draw_stop_negatives, util_auc)

TOL = 3e-3


def run_config(dataset, seed, methods):
    from unlearn import (unlearn_fgs, unlearn_finetune, unlearn_grad_ascent,
                         unlearn_retrain)
    cfg = build_config(dataset, seed)
    M = cfg["M"]
    arch = archived_per_seed(dataset, seed)
    print(f"[{dataset} seed={seed}] base replay: archived base_attack_auc "
          f"{arch['base_attack_auc']:.4f}")
    out = {}
    need_rn = any(m in methods for m in
                  ("fgs", "fgs_margin", "fgs_attack", "fgs_noproj"))
    rn = draw_retain_negatives(cfg) if need_rn else None
    for method in methods:
        t0 = time.time()
        if method == "retrain":
            p, _ = unlearn_retrain(M, cfg["remain_e"], cfg["n"], cfg["X"],
                                   cfg["train_neg"], epochs=EPOCHS, hid=HID,
                                   lr=LR, rng=np.random.default_rng(seed + 999))
        elif method == "finetune":
            p, _ = unlearn_finetune(M, cfg["base"], cfg["remain_e"], cfg["n"],
                                    cfg["X"], cfg["train_neg"], epochs=40,
                                    lr=LR)
        elif method == "grad_ascent":
            p, _ = unlearn_grad_ascent(M, cfg["base"], cfg["forget_e"],
                                       cfg["remain_e"], cfg["n"], cfg["X"],
                                       steps=15, lr=0.005)
        elif method in ("fgs", "fgs_margin"):
            rule = "ks" if method == "fgs" else "ks_margin"
            p, _, _, _ = unlearn_fgs(M, cfg["base"], cfg["forget_e"],
                                     cfg["remain_e"], cfg["n"], cfg["X"],
                                     cfg["remain_e"], rn, eta=FGS_ETA,
                                     val_edges=cfg["val_e"], stop_rule=rule)
        elif method in ("fgs_attack", "fgs_noproj"):
            stop_neg = draw_stop_negatives(cfg, seed)
            p, _, _, _ = unlearn_fgs(M, cfg["base"], cfg["forget_e"],
                                     cfg["remain_e"], cfg["n"], cfg["X"],
                                     cfg["remain_e"], rn, eta=FGS_ETA,
                                     val_edges=cfg["val_e"], max_steps=80,
                                     stop_rule="attack", stop_neg=stop_neg,
                                     use_projection=(method == "fgs_attack"))
        elif method == "gnndelete":
            from gnndelete import unlearn_gnndelete, scores_with_operator
            from experiment import sample_non_edges
            rng = cfg["rng"]
            k_ni = min(2048, len(cfg["remain_e"]))
            ni_pos = [cfg["remain_e"][i] for i in
                      rng.choice(len(cfg["remain_e"]), k_ni, replace=False)]
            ni_neg = sample_non_edges(rng, cfg["n"], cfg["edge_set"], k_ni,
                                      cfg["labeled"])
            phi, aff, _ = unlearn_gnndelete(M, "gcn", cfg["base"],
                                            cfg["forget_e"], cfg["remain_e"],
                                            cfg["n"], cfg["X"], ni_pos, ni_neg,
                                            epochs=40, lr=0.01, seed=seed)
            from pipeline import auc_score
            ya = np.array([1] * len(cfg["forget_e"])
                          + [0] * len(cfg["attack_neg"]))
            yt = np.array([1] * len(cfg["test_e"]) + [0] * len(cfg["test_neg"]))
            a = auc_score(ya, scores_with_operator(
                M, "gcn", cfg["base"], phi, aff, cfg["remain_e"], cfg["n"],
                cfg["X"], cfg["forget_e"] + cfg["attack_neg"]))
            u = auc_score(yt, scores_with_operator(
                M, "gcn", cfg["base"], phi, aff, cfg["remain_e"], cfg["n"],
                cfg["X"], cfg["test_e"] + cfg["test_neg"]))
            out[method] = (a, u, time.time() - t0)
            continue
        else:
            raise ValueError(method)
        out[method] = (attack_auc(cfg, p, cfg["remain_e"]),
                       util_auc(cfg, p, cfg["remain_e"]), time.time() - t0)
    ok = True
    for method, (a, u, s) in out.items():
        aa, au = arch["methods"][method]["attack"], arch["methods"][method]["util"]
        da, du = abs(a - aa), abs(u - au)
        flag = "OK " if max(da, du) <= TOL else "DIFF"
        if max(da, du) > TOL:
            ok = False
        print(f"  {flag} {method:<11} attack {a:.4f} (archived {aa:.4f}, "
              f"|d| {da:.4f})  util {u:.4f} (archived {au:.4f}, |d| {du:.4f})"
              f"  wall {s:.1f}s")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cora")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--methods", nargs="+",
                    default=["retrain", "fgs"])
    ap.add_argument("--all-headline", action="store_true",
                    help="all datasets x 10 seeds x all methods (hours)")
    args = ap.parse_args()
    if args.all_headline:
        jobs = [(ds, s, ["retrain", "finetune", "grad_ascent", "fgs",
                          "fgs_margin", "fgs_attack", "fgs_noproj",
                          "gnndelete"])
                for ds in ("cora", "citeseer", "pubmed")
                for s in (42, 7, 123, 2024, 2025, 31337, 31338, 5555,
                          7777, 999)]
    else:
        jobs = [(args.dataset, s, args.methods) for s in args.seeds]
    ok = True
    for ds, s, ms in jobs:
        ok = run_config(ds, s, ms) and ok
    print("EXCHANGEABILITY: all re-run values within 3e-3 of archived."
          if ok else "EXCHANGEABILITY: some values differ by > 3e-3 "
                      "(see DIFF lines).")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
