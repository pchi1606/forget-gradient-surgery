"""Aggregate per-seed JSONs into tables with uncertainty + paired statistics.

Reads results/per_seed/*.json (legacy gcn + backbone-prefixed) and
results/per_seed_ff/*.json. Prints markdown tables and writes
results/analysis.json.

Statistics per (backbone, dataset, method): mean +/- std (ddof=1) of
attack AUC, utility AUC, seconds; attack gap vs retrain computed
paired-per-seed; paired t-test and Wilcoxon signed-rank p-values for the
gap vs 0; Cohen's dz (paired effect size) for the gap.
"""
import glob
import json
import os

import numpy as np
from scipy.stats import ttest_rel, wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
PER_SEED = os.path.join(HERE, "results", "per_seed")
PER_FF = os.path.join(HERE, "results", "per_seed_ff")

METHODS = ["retrain", "finetune", "grad_ascent", "fgs", "fgs_margin",
           "fgs_attack", "fgs_noproj", "gnndelete"]


def load_main():
    """{(backbone, dataset, seed): record} for the 5% protocol."""
    recs = {}
    for path in glob.glob(os.path.join(PER_SEED, "*.json")):
        d = json.load(open(path))
        bb = d.get("backbone", "gcn")
        key = (bb, d["dataset"], d["seed"])
        recs[key] = d
    return recs


def load_ff():
    recs = {}
    for path in glob.glob(os.path.join(PER_FF, "*.json")):
        d = json.load(open(path))
        key = (d["backbone"], d["dataset"], d["seed"], d["forget_frac"])
        recs[key] = d
    return recs


def paired_stats(gaps):
    """gaps: array of per-seed paired differences. Returns dict."""
    g = np.asarray(gaps, dtype=float)
    n = len(g)
    out = {"n": n, "mean": float(g.mean()),
           "std": float(g.std(ddof=1)) if n > 1 else 0.0}
    if n > 1 and out["std"] > 0:
        t = ttest_rel(g, np.zeros(n))
        out["t_p"] = float(t.pvalue)
        try:
            w = wilcoxon(g)
            out["wilcoxon_p"] = float(w.pvalue)
        except Exception as e:  # e.g. zero-dispersion edge cases
            out["wilcoxon_p"] = None
            out["wilcoxon_note"] = str(e)[:80]
        out["cohen_dz"] = float(g.mean() / g.std(ddof=1))
    else:
        out["t_p"] = out["wilcoxon_p"] = out["cohen_dz"] = None
    return out


def main():
    recs = load_main()
    ff = load_ff()
    analysis = {"main": {}, "forget_frac": {}}

    combos = sorted(set((bb, ds) for (bb, ds, s) in recs))
    for (bb, ds) in combos:
        seeds = sorted(s for (b, d, s) in recs if b == bb and d == ds)
        block = {"seeds": seeds, "methods": {}}
        ret_atk = np.array([recs[(bb, ds, s)]["methods"]["retrain"]["attack"]
                            for s in seeds])
        for m in METHODS:
            have = [s for s in seeds if m in recs[(bb, ds, s)]["methods"]]
            atk = np.array([recs[(bb, ds, s)]["methods"][m]["attack"]
                            for s in have])
            utl = np.array([recs[(bb, ds, s)]["methods"][m]["util"]
                            for s in have])
            sec = np.array([recs[(bb, ds, s)]["methods"][m]["s"]
                            for s in have])
            gap = atk - np.array(
                [recs[(bb, ds, s)]["methods"]["retrain"]["attack"]
                 for s in have])
            ugap = utl - np.array(
                [recs[(bb, ds, s)]["methods"]["retrain"]["util"]
                 for s in have])
            info = {
                "n": len(have),
                "attack_mean": float(atk.mean()),
                "attack_std": float(atk.std(ddof=1)) if len(have) > 1 else 0.0,
                "util_mean": float(utl.mean()),
                "util_std": float(utl.std(ddof=1)) if len(have) > 1 else 0.0,
                "s_mean": float(sec.mean()),
                "attack_gap": paired_stats(gap),
                "util_gap": paired_stats(ugap),
            }
            if m in ("fgs", "fgs_margin", "fgs_attack"):
                steps = [recs[(bb, ds, s)]["methods"][m].get("steps")
                         for s in have]
                info["steps"] = steps
            block["methods"][m] = info
        # head-to-head: fgs vs fgs_margin, fgs vs fgs_attack,
        # fgs_attack vs fgs_noproj (paired attack)
        for other, key, base in (("fgs_margin", "fgs_vs_margin_attack", "fgs"),
                                 ("fgs_attack", "fgs_vs_attack_attack", "fgs"),
                                 ("fgs_noproj", "fgs_attack_vs_noproj_attack",
                                  "fgs_attack")):
            if base in block["methods"] and other in block["methods"]:
                a = np.array([recs[(bb, ds, s)]["methods"][base]["attack"]
                              for s in seeds
                              if other in recs[(bb, ds, s)]["methods"]
                              and base in recs[(bb, ds, s)]["methods"]])
                b = np.array([recs[(bb, ds, s)]["methods"][other]["attack"]
                              for s in seeds
                              if other in recs[(bb, ds, s)]["methods"]
                              and base in recs[(bb, ds, s)]["methods"]])
                if len(a) == len(b) and len(a) > 1:
                    block[key] = paired_stats(a - b)
        analysis["main"][f"{bb}/{ds}"] = block

    # forget-fraction sensitivity: mean attack gap vs retrain per frac
    ff_combos = sorted(set((bb, ds) for (bb, ds, s, f) in ff))
    for (bb, ds) in ff_combos:
        fracs = sorted(set(f for (b, d, s, f) in ff if b == bb and d == ds))
        fblock = {}
        for fr in fracs:
            seeds = sorted(s for (b, d, s, f) in ff
                           if b == bb and d == ds and f == fr)
            mrow = {}
            for m in METHODS:
                have = [s for s in seeds
                        if m in ff[(bb, ds, s, fr)]["methods"]]
                if not have:
                    continue
                gap = np.array([
                    ff[(bb, ds, s, fr)]["methods"][m]["attack"]
                    - ff[(bb, ds, s, fr)]["methods"]["retrain"]["attack"]
                    for s in have])
                ugap = np.array([
                    ff[(bb, ds, s, fr)]["methods"][m]["util"]
                    - ff[(bb, ds, s, fr)]["methods"]["retrain"]["util"]
                    for s in have])
                mrow[m] = {"n": len(have),
                           "attack_gap": paired_stats(gap),
                           "util_gap": paired_stats(ugap)}
            fblock[str(fr)] = mrow
        analysis["forget_frac"][f"{bb}/{ds}"] = fblock

    with open(os.path.join(HERE, "results", "analysis.json"), "w") as f:
        json.dump(analysis, f, indent=1)

    # ---- print markdown ----
    for key, block in analysis["main"].items():
        print(f"\n## {key} (retrain n={block['methods']['retrain']['n']} seeds: "
              f"{block['seeds']})")
        print("| n | method | attack | util | t(s) | attack gap vs retrain "
              "| util gap vs retrain |")
        print("|---|---|---|---|---|---|---|")
        for m in METHODS:
            if m not in block["methods"]:
                continue
            i = block["methods"][m]
            ag = i["attack_gap"]
            ug = i["util_gap"]
            def fmt(st):
                p = st["t_p"]
                ps = f"p={p:.3f}" if p is not None else "p=n/a"
                dz = st["cohen_dz"]
                dzs = f"dz={dz:.2f}" if dz is not None else ""
                return (f"{st['mean']:+.3f}±{st['std']:.3f} {ps} {dzs}")
            print(f"| {i['n']} | {m} | {i['attack_mean']:.3f}±{i['attack_std']:.3f} | "
                  f"{i['util_mean']:.3f}±{i['util_std']:.3f} | "
                  f"{i['s_mean']:.1f} | {fmt(ag)} | {fmt(ug)} |")
        for key, label in (("fgs_vs_margin_attack", "fgs - fgs_margin"),
                           ("fgs_vs_attack_attack", "fgs - fgs_attack"),
                           ("fgs_attack_vs_noproj_attack",
                            "fgs_attack - fgs_noproj")):
            if key in block:
                st = block[key]
                print(f"{label} attack (paired): {st['mean']:+.4f} "
                      f"p_t={st['t_p']:.3f} p_w={st['wilcoxon_p']} "
                      f"dz={st['cohen_dz']:.2f}")

    for key, fblock in analysis["forget_frac"].items():
        print(f"\n## forget-fraction sensitivity: {key}")
        print("| frac | method | attack gap vs retrain | util gap vs retrain |")
        print("|---|---|---|---|")
        for fr, mrow in fblock.items():
            for m in METHODS:
                if m not in mrow:
                    continue
                ag = mrow[m]["attack_gap"]
                ug = mrow[m]["util_gap"]
                print(f"| {fr} | {m} | {ag['mean']:+.3f}±{ag['std']:.3f} "
                      f"(n={ag['n']}) | {ug['mean']:+.3f}±{ug['std']:.3f} |")


if __name__ == "__main__":
    main()
