"""Assemble results/fixedstep.json from per-stream recs files.

Usage: python3 assemble_fixedstep.py <recs1.json> <recs2.json> ...
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiment_fixedstep as EF

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "fixedstep.json")


def main():
    recs = []
    for p in sys.argv[1:]:
        recs.extend(json.load(open(p)))
    # canonical order: dataset, seed
    recs.sort(key=lambda r: (r["dataset"], r["seed"]))
    devs = [r["attack_deviation_vs_published"] for r in recs]
    result = {
        "description": ("Fixed-step stopping ablation for FGS (Task A). "
                        "Re-runs the 40-step projected KS trajectory with a "
                        "documented fresh retain-negative anchor "
                        "(seed*1000+5) and scores attack/utility AUC at every "
                        "iterate 0..40. Covers all GCN five-percent configs: "
                        "%d configs (3 datasets x 10 seeds)." % len(recs)),
        "backbone": "gcn",
        "n_configs": len(recs),
        "fixed_steps": list(EF.FIXED_STEPS),
        "rn_anchor_stream": "np.random.default_rng(seed*1000+5)",
        "configs": recs,
        "fixed_step_summary": EF.summarize(recs),
        "validation": {
            "check": ("re-run KS-selected attack vs published per-seed fgs "
                      "attack (anchor-resampling noise bound ~0.005)"),
            "tolerance": EF.ATTACK_TOL,
            "max_attack_deviation": float(max(devs)),
            "pass": bool(max(devs) <= EF.ATTACK_TOL),
            "per_config_deviation": {
                f"{r['dataset']}_seed{r['seed']}":
                    round(r["attack_deviation_vs_published"], 5)
                for r in recs},
        },
    }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=1)
    print(f"wrote {OUT}")
    print(f"validation: max deviation {max(devs):.4f} "
          f"(tol {EF.ATTACK_TOL}) -> "
          f"{'PASS' if max(devs) <= EF.ATTACK_TOL else 'FAIL'}")
    for s in EF.FIXED_STEPS:
        sm = result["fixed_step_summary"][str(s)]
        print(f"  fixed step {s:>2}: attack diff "
              f"{sm['attack']['mean_diff']:+.4f} p={sm['attack']['t_p']:.3f} | "
              f"util diff {sm['util']['mean_diff']:+.4f} "
              f"p={sm['util']['t_p']:.3f}")


if __name__ == "__main__":
    main()
