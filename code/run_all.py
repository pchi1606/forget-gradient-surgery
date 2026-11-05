"""Full sweep: 3 datasets x 3 seeds. Aggregates per-seed JSONs into a
summary CSV and prints markdown tables. Skips seeds already done."""
import csv
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results", "per_seed")

DATASETS = ["cora", "citeseer", "pubmed"]
SEEDS = [42, 7, 123]


def main():
    for ds in DATASETS:
        for s in SEEDS:
            path = os.path.join(RES, f"{ds}_seed{s}.json")
            if os.path.exists(path):
                print(f"skip {ds} seed {s} (exists)")
                continue
            print(f"=== running {ds} seed {s} ===", flush=True)
            r = subprocess.run([sys.executable, os.path.join(HERE, "experiment.py"),
                                ds, str(s)], capture_output=True, text=True)
            tail = "\n".join(r.stdout.strip().split("\n")[-8:])
            print(tail)
            if r.returncode != 0:
                print("STDERR:", r.stderr[-2000:])
                return

    rows = []
    for path in sorted(glob.glob(os.path.join(RES, "*.json"))):
        d = json.load(open(path))
        for m, v in d["methods"].items():
            rows.append({"dataset": d["dataset"], "seed": d["seed"], "method": m,
                         "attack": round(v["attack"], 4), "util": round(v["util"], 4),
                         "s": v["s"],
                         "base_attack": d["base_attack_auc"],
                         "base_test": d["base_test_auc"]})
    with open(os.path.join(HERE, "results", "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # markdown tables
    for ds in DATASETS:
        print(f"\n## {ds}")
        print("| seed | method | attack AUC | util AUC | time (s) |")
        print("|---|---|---|---|---|")
        for r in rows:
            if r["dataset"] == ds:
                print(f"| {r['seed']} | {r['method']} | {r['attack']:.3f} | "
                      f"{r['util']:.3f} | {r['s']} |")
        print("\nmean over seeds:")
        for m in ["retrain", "finetune", "grad_ascent", "fgs"]:
            sub = [r for r in rows if r["dataset"] == ds and r["method"] == m]
            a = sum(r["attack"] for r in sub) / len(sub)
            u = sum(r["util"] for r in sub) / len(sub)
            t = sum(r["s"] for r in sub) / len(sub)
            print(f"  {m:<11} attack={a:.3f} util={u:.3f} t={t:.1f}s")


if __name__ == "__main__":
    main()
