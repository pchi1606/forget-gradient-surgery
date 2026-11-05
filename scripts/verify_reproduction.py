#!/usr/bin/env python3
"""Recompute every headline number of the FGS paper from the raw per-seed
JSON outputs under results/ and compare against expected/headline_numbers.json.

Coverage:
  * Table 4 attack/utility gaps (mean +/- sample SD, ddof=1) for
    finetune, grad_ascent (NegGrad), gnndelete, fgs, fgs_margin,
    fgs_attack, fgs_noproj on Cora / CiteSeer / PubMed (n=10 GCN seeds),
    plus the unlink-only (UtU) row, recomputed from results/fixedstep.json
    iterate 0, and the fgs hybrid CiteSeer attack-gap mean from
    results/analysis_v2.json's hybrid summary (itself recomputed there
    from the archived trajectories).
  * GNNDelete lambda sweep (Table 5): the lambda=1.0 column, recomputed
    from the Table 4 GNNDelete gaps (the manuscript notes this column is
    an independent re-run reproducing Table 4 within 0.001). The other
    lambda columns and the best-point utility values were not archived
    as raw outputs in the experiment tree; they are reported as
    NOT ARCHIVED and counted separately -- never as passes.
  * Fraction sensitivity (Section 6.2 text values) from
    results/per_seed_ff (KS-stopped FGS, n=5 seeds per fraction).
  * Fixed-step ablation (Table 8) from results/fixedstep.json.
  * MEGU and ETR aggregates (run records of 2026-11-04) from
    results/megu/*.json and results/etr/*.json.

Exit code is nonzero if any recomputable check mismatches beyond the
tolerance stored in the expected file (rounding of the printed 3-decimal
values). Pure stdlib + numpy. Run from anywhere:
    python3 scripts/verify_reproduction.py
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOL = None  # set from expected file

passed, failed, not_archived = [], [], []


def check(name, got, want, tol=None):
    tol = TOL if tol is None else tol
    if abs(got - want) <= tol:
        passed.append(name)
        print(f"  PASS {name}: recomputed {got:+.4f} vs expected {want:+.4f}")
    else:
        failed.append(name)
        print(f"  FAIL {name}: recomputed {got:+.4f} vs expected {want:+.4f} "
              f"(|diff| {abs(got - want):.4f} > tol {tol})")


def check_pair(name, gaps, want_pair):
    check(name + " mean", float(np.mean(gaps)), want_pair[0])
    check(name + " sd", float(np.std(gaps, ddof=1)), want_pair[1])


def load_main():
    """{(dataset, seed): per-seed dict} for the GCN n=10 headline runs."""
    out = {}
    for path in glob.glob(os.path.join(ROOT, "results", "per_seed", "*.json")):
        with open(path) as f:
            d = json.load(f)
        if d.get("backbone", "gcn") != "gcn":
            continue
        out[(d["dataset"], int(d["seed"]))] = d
    return out


def main():
    global TOL
    with open(os.path.join(ROOT, "expected", "headline_numbers.json")) as f:
        exp = json.load(f)
    TOL = float(exp.get("tolerance", 0.0016))
    seeds = exp["seeds"]
    datasets = ["cora", "citeseer", "pubmed"]
    per_seed = load_main()

    print("== Table 4 gaps (n=10 seeds per dataset) ==")
    for method, want in exp["table4_gaps"].items():
        if method == "utu" or not isinstance(want, dict):
            continue  # utu handled below; scalar entries are not method rows
        for ds in datasets:
            gaps_a, gaps_u = [], []
            for s in seeds:
                d = per_seed.get((ds, s))
                if d is None or method not in d["methods"]:
                    failed.append(f"table4 {method} {ds} seed{s} missing")
                    print(f"  FAIL table4 {method} {ds}: seed {s} missing")
                    continue
                r = d["methods"]["retrain"]
                gaps_a.append(d["methods"][method]["attack"] - r["attack"])
                gaps_u.append(d["methods"][method]["util"] - r["util"])
            if len(gaps_a) == len(seeds):
                check_pair(f"table4 {method} {ds} attack_gap", gaps_a,
                           want[ds]["attack"])
                check_pair(f"table4 {method} {ds} util_gap", gaps_u,
                           want[ds]["util"])

    print("== Table 4 UtU (unlink-only) from fixedstep.json iterate 0 ==")
    with open(os.path.join(ROOT, "results", "fixedstep.json")) as f:
        fixed = json.load(f)
    for ds in datasets:
        cfgs = [c for c in fixed["configs"] if c["dataset"] == ds]
        gaps_a = [c["fixed"]["0"]["attack_gap"] for c in cfgs]
        gaps_u = [c["fixed"]["0"]["util_gap"] for c in cfgs]
        check_pair(f"table4 utu {ds} attack_gap", gaps_a,
                   exp["table4_gaps"]["utu"][ds]["attack"])
        check_pair(f"table4 utu {ds} util_gap", gaps_u,
                   exp["table4_gaps"]["utu"][ds]["util"])

    print("== fgs hybrid CiteSeer attack-gap mean (analysis_v2 hybrid) ==")
    with open(os.path.join(ROOT, "results", "analysis_v2.json")) as f:
        a2 = json.load(f)
    citeseer_hybrid = a2["hybrid"]["summary"]["citeseer_comparison"]
    check("hybrid citeseer attack_gap mean",
          float(np.mean([c["hybrid_gap"] for c in citeseer_hybrid])),
          exp["table4_gaps"]["fgs_hybrid_citeseer_attack_gap_mean"])

    print("== GNNDelete lambda sweep (Table 5) ==")
    lam = exp["lambda_sweep_table5"]
    for ds in datasets:
        gaps = []
        for s in seeds:
            d = per_seed[(ds, s)]
            gaps.append(d["methods"]["gnndelete"]["attack"]
                        - d["methods"]["retrain"]["attack"])
        check_pair(f"lambda {ds} lam=1.0 attack_gap (Table 5 re-run)",
                   gaps, lam["attack_gap"][ds]["1.0"])
    for ds in datasets:
        for lmb, pair in lam["attack_gap"][ds].items():
            if lmb != "1.0":
                not_archived.append(f"lambda {ds} lam={lmb} attack_gap "
                                    f"(printed {pair[0]:+.3f} +/- {pair[1]:.3f})")
        bp = lam["best_points"][ds]
        not_archived.append(f"lambda {ds} best point lam={bp['lambda']} "
                            f"attack/util (printed {bp['attack'][0]:+.3f} / "
                            f"{bp['util'][0]:+.3f})")
    print(f"  NOT ARCHIVED: {len(not_archived)} lambda-sweep values have no raw "
          "outputs in the experiment tree (see expected file note); they are "
          "reported, not passed.")

    print("== Fraction sensitivity (per_seed_ff, fgs, n=5) ==")
    ff = {}
    for path in glob.glob(os.path.join(ROOT, "results", "per_seed_ff", "*.json")):
        with open(path) as f:
            d = json.load(f)
        ff.setdefault((d["dataset"], float(d["forget_frac"])), []).append(d)
    fr = exp["fraction_sensitivity"]
    for ds, by_ff in fr["fgs_attack_gap_mean"].items():
        for ffs, want in by_ff.items():
            rows = ff[(ds, float(ffs))]
            gaps = [r["methods"]["fgs"]["attack"] - r["methods"]["retrain"]["attack"]
                    for r in rows]
            check(f"fraction {ds} ff={ffs} fgs attack_gap mean",
                  float(np.mean(gaps)), want)
    for ds, by_ff in fr.get("fgs_attack_gap_sd", {}).items():
        for ffs, want in by_ff.items():
            rows = ff[(ds, float(ffs))]
            gaps = [r["methods"]["fgs"]["attack"] - r["methods"]["retrain"]["attack"]
                    for r in rows]
            check(f"fraction {ds} ff={ffs} fgs attack_gap sd",
                  float(np.std(gaps, ddof=1)), want)
    for ds, by_ff in fr.get("fgs_util_gap_mean", {}).items():
        for ffs, want in by_ff.items():
            rows = ff[(ds, float(ffs))]
            gaps = [r["methods"]["fgs"]["util"] - r["methods"]["retrain"]["util"]
                    for r in rows]
            check(f"fraction {ds} ff={ffs} fgs util_gap mean",
                  float(np.mean(gaps)), want)
    for ds, by_ff in fr.get("fgs_util_gap_sd", {}).items():
        for ffs, want in by_ff.items():
            rows = ff[(ds, float(ffs))]
            gaps = [r["methods"]["fgs"]["util"] - r["methods"]["retrain"]["util"]
                    for r in rows]
            check(f"fraction {ds} ff={ffs} fgs util_gap sd",
                  float(np.std(gaps, ddof=1)), want)
    lo, hi = fr["pubmed_attack_gap_range_all_fractions"]
    for ffs in ["0.01", "0.10", "0.20"]:
        rows = ff[("pubmed", float(ffs))]
        ga = float(np.mean([r["methods"]["fgs"]["attack"]
                            - r["methods"]["retrain"]["attack"] for r in rows]))
        gu = float(np.mean([r["methods"]["fgs"]["util"]
                            - r["methods"]["retrain"]["util"] for r in rows]))
        name = f"fraction pubmed ff={ffs} ranges"
        if lo - TOL <= ga <= hi + TOL and abs(gu) <= fr[
                "pubmed_util_gap_abs_max_all_fractions"] + TOL:
            passed.append(name)
            print(f"  PASS {name}: attack mean {ga:+.4f} in [{lo:+.3f},{hi:+.3f}], "
                  f"|util mean| {abs(gu):.4f}")
        else:
            failed.append(name)
            print(f"  FAIL {name}: attack mean {ga:+.4f}, util mean {gu:+.4f}")

    print("== Fixed-step ablation (Table 8, fixedstep.json) ==")
    summ = fixed["fixed_step_summary"]
    for step, want in exp["fixed_step_table8"]["attack_diff"].items():
        check(f"fixedstep step={step} attack mean_diff",
              float(summ[step]["attack"]["mean_diff"]), want)
    for step, want in exp["fixed_step_table8"]["util_diff"].items():
        check(f"fixedstep step={step} util mean_diff",
              float(summ[step]["util"]["mean_diff"]), want)

    print("== MEGU / ETR aggregates ==")
    for label, sub in [("megu", "megu"), ("etr", "etr")]:
        for ds in datasets:
            ga, gu = [], []
            for path in sorted(glob.glob(os.path.join(
                    ROOT, "results", sub, f"fgs_{label}_{ds}_seed*.json"))):
                with open(path) as f:
                    d = json.load(f)
                ga.append(d["attack_gap_vs_retrain"])
                gu.append(d["util_gap_vs_retrain"])
            if len(ga) != 10:
                failed.append(f"{label} {ds} count")
                print(f"  FAIL {label} {ds}: {len(ga)} per-seed files, want 10")
                continue
            check_pair(f"{label} {ds} attack_gap", ga,
                       exp[f"{label}_aggregates"][ds]["attack"])
            check_pair(f"{label} {ds} util_gap", gu,
                       exp[f"{label}_aggregates"][ds]["util"])

    print()
    print(f"CHECKS PASSED: {len(passed)}")
    print(f"CHECKS FAILED: {len(failed)}")
    print(f"NOT ARCHIVED (reported, not passed): {len(not_archived)}")
    for name in not_archived:
        print(f"  - {name}")
    if failed:
        print("FAILED CHECKS:")
        for name in failed:
            print(f"  - {name}")
        return 1
    print("ALL RECOMPUTABLE HEADLINE NUMBERS MATCH.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
