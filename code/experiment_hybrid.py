"""Task B — Cheap hybrid stopping variant (fgs_hybrid) for FGS.

Uses the SAME 40-step trajectories saved by experiment_fixedstep.py
(results/fixedstep_iterates/*.npz) -- NO new gradient runs.

Per config:
  1. Take the 5 iterates with the lowest KS (from the trajectory's own
     diag["ks"], iterates 0..39 -- the same selection space the KS rule
     uses).
  2. Compute the attack-calibrated key
       |AUC(forget vs stop_neg) - AUC(val vs stop_neg)|
     on ONLY those 5 iterates, select the minimizer.
  3. stop_neg = 4 pooled hard-negative samples from the documented seed
     stream np.random.default_rng(seed*100003 + 11), disjoint construction
     mirroring experiment.py (this is the SAME stop_neg the published
     fgs_attack used).
  4. Record attack/util/steps at the selected iterate and the wall-clock
     of the selection step (the 5-iterate scoring only).

Appends a "hybrid" block to results/fixedstep.json and compares hybrid vs
fgs (KS, published) vs fgs_attack (published per-seed JSONs; missing for
CiteSeer seed 2024) -- on CiteSeer especially: does it close the CiteSeer
gap at ~fgs cost?

Usage: python3 experiment_hybrid.py
"""
import json
import os
import time

import numpy as np
from scipy.stats import rankdata

from data import load_planetoid
import gcn as M

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "results", "cache")
PER_SEED = os.path.join(HERE, "results", "per_seed")
ITER_DIR = os.path.join(HERE, "results", "fixedstep_iterates")
FIXEDSTEP_JSON = os.path.join(HERE, "results", "fixedstep.json")

LEGACY_SEEDS = (42, 7, 123)
N_TOP_KS = 5
STOP_SEED_MULT = 100003
STOP_SEED_ADD = 11


def auc_score(y, s):
    r = rankdata(s)
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def hard_negatives(rng, forget_edges, labels, edge_set, by_label, seen=None):
    """One non-edge per forget edge, matched on the sorted label pair.

    `seen`: optional set shared across calls to enforce disjointness
    (e.g. stopping-rule negatives disjoint from evaluation attack
    negatives). When None, a fresh per-call set is used.
    """
    out = []
    if seen is None:
        seen = set()
    for (u, v) in forget_edges:
        key = tuple(sorted((int(labels[u]), int(labels[v]))))
        cands = by_label.get(key)
        if not cands:
            continue
        while True:
            a = cands[rng.integers(0, len(cands))]
            b = cands[rng.integers(0, len(cands))]
            if a == b:
                continue
            e = (a, b) if a < b else (b, a)
            if e in edge_set or e in seen:
                continue
            seen.add(e)
            out.append(e)
            break
    return out


def _pairs(edgelist):
    u = np.array([e[0] for e in edgelist], dtype=np.int64)
    v = np.array([e[1] for e in edgelist], dtype=np.int64)
    return u, v


def json_path(dataset, seed):
    pre = "" if seed in LEGACY_SEEDS else "gcn_"
    return os.path.join(PER_SEED, f"{pre}{dataset}_seed{seed}.json")


def run_config(cfg):
    dataset, seed = cfg["dataset"], cfg["seed"]
    npz = np.load(os.path.join(
        ITER_DIR, f"{dataset}_seed{seed}_iterates.npz"))
    ks = np.array(npz["ks"])                      # 40 values, iterates 0..39
    top5 = np.argsort(ks)[:N_TOP_KS]              # 5 lowest-KS iterates

    # rebuild label-pair buckets (same construction as experiment.py)
    Xsp, labels, edges = load_planetoid(dataset)
    n = Xsp.shape[0]
    X = Xsp.toarray().astype(np.float32)
    edge_set = set(edges)
    labeled = np.where(labels >= 0)[0]
    by_label = {}
    lab = labels[labeled]
    for i, nd in enumerate(labeled):
        by_label.setdefault(int(lab[i]), []).append(int(nd))
    pair_buckets = {}
    keys = sorted(by_label)
    for ii, c1 in enumerate(keys):
        for c2 in keys[ii:]:
            members = (by_label[c1] + by_label[c2] if c1 != c2
                       else by_label[c1])
            pair_buckets[(c1, c2)] = members

    z = np.load(os.path.join(
        CACHE_DIR, cfg["cache_file"]))
    forget_e = [tuple(e) for e in z["forget_e"]]
    remain_e = [tuple(e) for e in z["remain_e"]]
    val_e = [tuple(e) for e in z["val_e"]]
    attack_neg = [tuple(e) for e in z["attack_neg"]]

    # attack-calibrated stop negatives: documented stream, same as fgs_attack.
    # M4 fix: disjoint from the evaluation attack_neg (shared seen set),
    # mirroring the fixed experiment.py construction.
    rng_stop = np.random.default_rng(seed * STOP_SEED_MULT + STOP_SEED_ADD)
    stop_seen = set(attack_neg)
    stop_neg = []
    for _ in range(4):
        stop_neg += hard_negatives(rng_stop, forget_e, labels, edge_set,
                                   pair_buckets, seen=stop_seen)

    fu, fv = _pairs(forget_e)
    vu, vv = _pairs(val_e)
    hu, hv = _pairs(stop_neg)
    y_sur = np.array([1] * len(fu) + [0] * len(hu))
    y_ref = np.array([1] * len(vu) + [0] * len(hu))

    An = M.norm_adj(remain_e, n)
    t0 = time.time()
    keys_sel = []
    for i in top5:
        prm = {k: npz[f"iter{int(i):02d}_{k}"] for k in ("W1", "W2")}
        Z, _ = M.forward(prm, An, X)
        sf = M.pair_scores(Z, fu, fv)
        sv = M.pair_scores(Z, vu, vv)
        sh = M.pair_scores(Z, hu, hv)
        surr = auc_score(y_sur, np.concatenate([sf, sh]))
        ref = auc_score(y_ref, np.concatenate([sv, sh]))
        keys_sel.append(abs(surr - ref))
    sel_seconds = time.time() - t0
    best_local = int(np.argmin(keys_sel))
    best_iter = int(top5[best_local])

    attacks = list(np.array(npz["attack"], dtype=float))
    utils = list(np.array(npz["util"], dtype=float))
    ret_atk, ret_utl = cfg["retrain_attack"], cfg["retrain_util"]
    pub = json.load(open(json_path(dataset, seed)))
    fgs_attack_pub = pub["methods"].get("fgs_attack")

    rec = {
        "dataset": dataset, "seed": seed,
        "stop_neg_stream": f"np.random.default_rng({seed}*100003+11), "
                           "4x hard_negatives (same construction as "
                           "experiment.py fgs_attack)",
        "n_stop_neg": len(stop_neg),
        "top5_ks_iterates": [int(i) for i in top5],
        "top5_keys": [round(float(k), 5) for k in keys_sel],
        "selected_step": best_iter,
        "attack": attacks[best_iter], "util": utils[best_iter],
        "attack_gap": attacks[best_iter] - ret_atk,
        "util_gap": utils[best_iter] - ret_utl,
        "selection_seconds": round(sel_seconds, 2),
        "published_fgs_ks_gap": (pub["methods"]["fgs"]["attack"]
                                - ret_atk),
        "published_fgs_ks_steps": pub["methods"]["fgs"].get("steps"),
        "published_fgs_ks_seconds": pub["methods"]["fgs"]["s"],
        "published_fgs_attackrule_gap": (
            None if fgs_attack_pub is None
            else fgs_attack_pub["attack"] - ret_atk),
        "published_fgs_attackrule_seconds": (
            None if fgs_attack_pub is None else fgs_attack_pub["s"]),
        "published_fgs_attackrule_steps": (
            None if fgs_attack_pub is None else fgs_attack_pub.get("steps")),
    }
    print(f"[hybrid {dataset} seed={seed}] top5={rec['top5_ks_iterates']} "
          f"-> step {best_iter} attack={rec['attack']:.4f} "
          f"gap={rec['attack_gap']:+.4f} sel_t={sel_seconds:.1f}s",
          flush=True)
    return rec


def main():
    fs = json.load(open(FIXEDSTEP_JSON))
    recs = [run_config(c) for c in fs["configs"]]

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return float(np.mean(xs)) if xs else None

    # CiteSeer comparison: hybrid vs fgs(KS) vs fgs_attack, per seed
    cs = [r for r in recs if r["dataset"] == "citeseer"]
    citeseer_table = [{
        "seed": r["seed"],
        "hybrid_gap": round(r["attack_gap"], 4),
        "fgs_ks_gap": round(r["published_fgs_ks_gap"], 4),
        "fgs_attack_gap": (None if r["published_fgs_attackrule_gap"]
                           is None else
                           round(r["published_fgs_attackrule_gap"], 4)),
    } for r in cs]
    hybrid = {
        "description": ("fgs_hybrid: 40-step projected KS trajectory (same "
                        "trajectories as Task A, no new gradient runs); "
                        "attack-calibrated key evaluated on the 5 lowest-KS "
                        "iterates only."),
        "configs": recs,
        "mean_selection_seconds": round(
            float(np.mean([r["selection_seconds"] for r in recs])), 2),
        "mean_attack_gap_hybrid": mean([r["attack_gap"] for r in recs]),
        "mean_attack_gap_fgs_ks": mean(
            [r["published_fgs_ks_gap"] for r in recs]),
        "mean_attack_gap_fgs_attackrule": mean(
            [r["published_fgs_attackrule_gap"] for r in recs]),
        "mean_util_gap_hybrid": mean([r["util_gap"] for r in recs]),
        "citeseer_comparison": citeseer_table,
    }
    fs["hybrid"] = hybrid
    with open(FIXEDSTEP_JSON, "w") as f:
        json.dump(fs, f, indent=1)
    print(f"\nappended 'hybrid' to {FIXEDSTEP_JSON}")
    print("CiteSeer attack gaps (hybrid | fgs KS | fgs_attack):")
    for row in citeseer_table:
        print(f"  seed {row['seed']}: {row['hybrid_gap']:+.4f} | "
              f"{row['fgs_ks_gap']:+.4f} | {row['fgs_attack_gap']}")
    print(f"overall mean attack gap: hybrid {hybrid['mean_attack_gap_hybrid']:+.4f} "
          f"vs fgs_ks {hybrid['mean_attack_gap_fgs_ks']:+.4f} "
          f"vs fgs_attackrule {hybrid['mean_attack_gap_fgs_attackrule']}")
    return hybrid


if __name__ == "__main__":
    main()
