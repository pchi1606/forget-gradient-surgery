"""Task E — consolidated analysis: results/analysis_v2.json.

Contents:
  (i)   n=10 aggregates (mean +/- sample SD, paired gaps vs retrain with
        paired t-test p and Cohen's dz) for all 8 methods on the 5% runs,
        reusing analyze.py's paired_stats.
  (ii)  the fixed-step stopping summary (from results/fixedstep.json).
  (iii) the hybrid stopping summary (from results/fixedstep.json).
  (iv)  1%/10%/20% forget-fraction table (from results/per_seed_ff/).

Every aggregate lists its source JSON files so each number traces back.
Writes results/analysis_v2.json and prints markdown tables.

Usage: python3 analysis_v2.py
"""
import glob
import json
import os

import numpy as np

from analyze import paired_stats

HERE = os.path.dirname(os.path.abspath(__file__))
PER_SEED = os.path.join(HERE, "results", "per_seed")
PER_FF = os.path.join(HERE, "results", "per_seed_ff")
OUT = os.path.join(HERE, "results", "analysis_v2.json")

METHODS = ["retrain", "finetune", "grad_ascent", "fgs", "fgs_margin",
           "fgs_attack", "fgs_noproj", "gnndelete"]
FF_METHODS = ["retrain", "finetune", "grad_ascent", "fgs", "fgs_margin",
              "fgs_attack", "gnndelete"]  # experiment_ff.py has no fgs_noproj
DATASETS = ("cora", "citeseer", "pubmed")
SEEDS_10 = (42, 7, 123, 2024, 31337, 999, 2025, 5555, 7777, 31338)


def rel(path):
    return os.path.relpath(path, HERE)


def load_main_gcn():
    """{(dataset, seed): (path, record)} for gcn 5% runs."""
    recs = {}
    for path in glob.glob(os.path.join(PER_SEED, "*.json")):
        d = json.load(open(path))
        if d.get("backbone", "gcn") != "gcn":
            continue
        recs[(d["dataset"], d["seed"])] = (rel(path), d)
    return recs


def load_ff_gcn():
    recs = {}
    for path in glob.glob(os.path.join(PER_FF, "*.json")):
        d = json.load(open(path))
        if d.get("backbone", "gcn") != "gcn":
            continue
        recs[(d["dataset"], d["seed"], d["forget_frac"])] = (rel(path), d)
    return recs


def main():
    recs = load_main_gcn()
    ff = load_ff_gcn()
    analysis = {"main_n10": {}, "fixed_step": None, "hybrid": None,
                "forget_frac": {}, "provenance": {}}

    # ---- (i) n=10 aggregates ----
    for ds in DATASETS:
        seeds = sorted(s for (d, s) in recs if d == ds)
        block = {"seeds": seeds, "methods": {}, "sources": []}
        for s in seeds:
            block["sources"].append(recs[(ds, s)][0])
        for m in METHODS:
            have = [s for s in seeds if m in recs[(ds, s)][1]["methods"]]
            src = [recs[(ds, s)][0] for s in have]
            atk = np.array([recs[(ds, s)][1]["methods"][m]["attack"]
                            for s in have])
            utl = np.array([recs[(ds, s)][1]["methods"][m]["util"]
                            for s in have])
            sec = np.array([recs[(ds, s)][1]["methods"][m]["s"]
                            for s in have])
            gap = atk - np.array(
                [recs[(ds, s)][1]["methods"]["retrain"]["attack"]
                 for s in have])
            ugap = utl - np.array(
                [recs[(ds, s)][1]["methods"]["retrain"]["util"]
                 for s in have])
            info = {
                "n": len(have),
                "attack_mean": float(atk.mean()),
                "attack_std": (float(atk.std(ddof=1)) if len(have) > 1
                               else 0.0),
                "util_mean": float(utl.mean()),
                "util_std": (float(utl.std(ddof=1)) if len(have) > 1
                             else 0.0),
                "s_mean": float(sec.mean()),
                "attack_gap": paired_stats(gap),
                "util_gap": paired_stats(ugap),
                "sources": src,
            }
            if m in ("fgs", "fgs_margin", "fgs_attack", "fgs_noproj"):
                info["steps"] = [recs[(ds, s)][1]["methods"][m].get("steps")
                                 for s in have]
            block["methods"][m] = info
        # head-to-head paired attack comparisons (as in analyze.py)
        for other, key, base in (
                ("fgs_margin", "fgs_vs_margin_attack", "fgs"),
                ("fgs_attack", "fgs_vs_attack_attack", "fgs"),
                ("fgs_noproj", "fgs_attack_vs_noproj_attack", "fgs_attack")):
            a = np.array([recs[(ds, s)][1]["methods"][base]["attack"]
                          for s in seeds
                          if other in recs[(ds, s)][1]["methods"]
                          and base in recs[(ds, s)][1]["methods"]])
            b = np.array([recs[(ds, s)][1]["methods"][other]["attack"]
                          for s in seeds
                          if other in recs[(ds, s)][1]["methods"]
                          and base in recs[(ds, s)][1]["methods"]])
            if len(a) == len(b) and len(a) > 1:
                block[key] = paired_stats(a - b)
        analysis["main_n10"][f"gcn/{ds}"] = block
    analysis["provenance"]["main_n10"] = {
        ds: [recs[(ds, s)][0] for s in sorted(s for (d, s) in recs if d == ds)]
        for ds in DATASETS}

    # ---- (ii)+(iii) fixed-step + hybrid (computed in fixedstep.json) ----
    fs = json.load(open(os.path.join(HERE, "results", "fixedstep.json")))
    analysis["fixed_step"] = {
        "source": "results/fixedstep.json",
        "summary": fs["fixed_step_summary"],
        "validation": fs["validation"],
    }
    analysis["hybrid"] = {"source": "results/fixedstep.json#hybrid",
                          "summary": fs["hybrid"]}
    analysis["provenance"]["fixed_step"] = {
        "fixedstep_json": "results/fixedstep.json",
        "iterate_npzs": sorted(
            rel(p) for p in glob.glob(os.path.join(
                HERE, "results", "fixedstep_iterates", "*.npz"))),
        "per_seed_jsons": [c["json_file"] for c in fs["configs"]],
    }

    # ---- (iv) forget-fraction table ----
    for ds in DATASETS:
        fracs = sorted(set(f for (d, s, f) in ff if d == ds))
        fblock = {"sources": []}
        for fr in fracs:
            seeds = sorted(s for (d, s, f) in ff if d == ds and f == fr)
            mrow = {}
            for m in FF_METHODS:
                have = [s for s in seeds
                        if m in ff[(ds, s, fr)][1]["methods"]]
                if not have:
                    continue
                src = [ff[(ds, s, fr)][0] for s in have]
                fblock["sources"].extend(src)
                gap = np.array([
                    ff[(ds, s, fr)][1]["methods"][m]["attack"]
                    - ff[(ds, s, fr)][1]["methods"]["retrain"]["attack"]
                    for s in have])
                ugap = np.array([
                    ff[(ds, s, fr)][1]["methods"][m]["util"]
                    - ff[(ds, s, fr)][1]["methods"]["retrain"]["util"]
                    for s in have])
                mrow[m] = {"n": len(have),
                           "attack_gap": paired_stats(gap),
                           "util_gap": paired_stats(ugap),
                           "sources": src}
            fblock[str(fr)] = mrow
        fblock["sources"] = sorted(set(fblock["sources"]))
        analysis["forget_frac"][f"gcn/{ds}"] = fblock

    with open(OUT, "w") as f:
        json.dump(analysis, f, indent=1)

    # ---- print markdown ----
    for key, block in analysis["main_n10"].items():
        print(f"\n## {key} n=10 (seeds: {block['seeds']})")
        print("| n | method | attack | util | t(s) | attack gap vs retrain "
              "| util gap vs retrain |")
        print("|---|---|---|---|---|---|---|")
        for m in METHODS:
            if m not in block["methods"]:
                continue
            i = block["methods"][m]
            ag, ug = i["attack_gap"], i["util_gap"]

            def fmt(st):
                p = st["t_p"]
                ps = f"p={p:.3f}" if p is not None else "p=n/a"
                dz = st["cohen_dz"]
                dzs = f"dz={dz:.2f}" if dz is not None else ""
                return f"{st['mean']:+.3f}±{st['std']:.3f} {ps} {dzs}"
            print(f"| {i['n']} | {m} | {i['attack_mean']:.3f}±{i['attack_std']:.3f} | "
                  f"{i['util_mean']:.3f}±{i['util_std']:.3f} | "
                  f"{i['s_mean']:.1f} | {fmt(ag)} | {fmt(ug)} |")
        for k, label in (("fgs_vs_margin_attack", "fgs - fgs_margin"),
                         ("fgs_vs_attack_attack", "fgs - fgs_attack"),
                         ("fgs_attack_vs_noproj_attack",
                          "fgs_attack - fgs_noproj")):
            if k in block:
                st = block[k]
                print(f"{label} attack (paired): {st['mean']:+.4f} "
                      f"p_t={st['t_p']:.3f} dz={st['cohen_dz']:.2f}")

    print("\n## forget fractions (attack gap vs retrain)")
    for key, fblock in analysis["forget_frac"].items():
        print(f"\n### {key}")
        for fr in sorted(k for k in fblock if k != "sources"):
            row = " | ".join(
                f"{m}: {fblock[fr][m]['attack_gap']['mean']:+.3f}"
                f" (n={fblock[fr][m]['n']})"
                for m in FF_METHODS if m in fblock[fr])
            print(f"  frac {fr}: {row}")

    print(f"\nwrote {rel(OUT)}")


if __name__ == "__main__":
    main()
