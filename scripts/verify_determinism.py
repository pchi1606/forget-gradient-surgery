# Author: Prabhu Rangaraju
"""Determinism verification: re-run the FROZEN pipeline for selected configs
into results/rerun_verify/ and compare attack/util of every method against the
stored JSONs. This verifies (a) gnndelete values were produced by this code,
(b) the 3 legacy seeds reproduce under the frozen pipeline (seed-pooling
exchangeability for fix #3)."""
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import experiment

OUT = os.path.join(HERE, "results", "rerun_verify")
os.makedirs(OUT, exist_ok=True)
experiment.RES_DIR = OUT

# (stored_filename, dataset, seed)
CONFIGS = [
    ("citeseer_seed7.json", "citeseer", 7),
    ("citeseer_seed42.json", "citeseer", 42),
    ("citeseer_seed123.json", "citeseer", 123),
    ("gcn_citeseer_seed2024.json", "citeseer", 2024),
    ("gcn_cora_seed999.json", "cora", 999),
    ("gcn_pubmed_seed2024.json", "pubmed", 2024),
]

report = []
for stored, ds, seed in CONFIGS:
    print(f"[rerun] {ds} seed={seed}", flush=True)
    res = experiment.run(ds, seed)
    # drop the big cache npz to keep the package lean
    shutil.rmtree(os.path.join(HERE, "results", "cache"), ignore_errors=True)
    old = json.load(open(os.path.join(HERE, "results", "per_seed", stored)))
    worst = 0.0
    for m, v in res["methods"].items():
        o = old["methods"][m]
        d = max(abs(v["attack"] - o["attack"]), abs(v["util"] - o["util"]))
        worst = max(worst, d)
        if d > 1e-9:
            report.append(f"MISMATCH {stored} {m}: attack {o['attack']} -> {v['attack']}, "
                          f"util {o['util']} -> {v['util']}")
    report.append(f"OK {stored}: max |diff| over methods = {worst:.2e}")
    print(f"  max|diff|={worst:.2e}", flush=True)

with open(os.path.join(OUT, "REPORT.txt"), "w") as f:
    f.write("\n".join(report) + "\n")
print("\n".join(report))
