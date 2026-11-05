"""Universal add-on: add `fgs_attack` (and any missing new method) to every
per_seed JSON that lacks it. Backbone-aware (gcn|sage), idempotent.

For each results/per_seed/<backbone>_<dataset>_seed<seed>.json (also legacy
<dataset>_seed<seed>.json = gcn):
  - load the matching cache (backbone-prefixed, legacy fallback)
  - rebuild the deterministic retain anchor (seed*31+7) and the held-out
    stop negatives (seed*100003+11, 4x)
  - run unlearn_fgs(stop_rule="attack", max_steps=80) and record
    attack/util/time/steps
Skips files that already contain fgs_attack.
"""
import glob
import json
import os

import numpy as np
from scipy.stats import rankdata

from data import load_planetoid
from unlearn import unlearn_fgs
from experiment import sample_non_edges, hard_negatives

HERE = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.join(HERE, "results", "per_seed")
CACHE_DIR = os.path.join(HERE, "results", "cache")
FGS_ETA = 0.05


def auc_score(y, s):
    r = rankdata(s)
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def process(path):
    res = json.load(open(path))
    if "fgs_attack" in res["methods"]:
        print(f"[skip] {os.path.basename(path)} already has fgs_attack")
        return
    backbone = res.get("backbone", "gcn")
    dataset = res["dataset"]
    seed = res["seed"]
    if backbone == "gcn":
        import gcn as M
        pkeys = ("W1", "W2")
    elif backbone == "sage":
        import sage as M
        pkeys = ("W1n", "W1s", "W2n", "W2s")
    else:
        raise ValueError(backbone)

    Xsp, labels, edges = load_planetoid(dataset)
    n = Xsp.shape[0]
    X = Xsp.toarray().astype(np.float32)
    labeled = np.where(labels >= 0)[0]
    by_label = {}
    lab = labels[labeled]
    for i, nd in enumerate(labeled):
        by_label.setdefault(int(lab[i]), []).append(int(nd))
    pair_buckets = {}
    keys = sorted(by_label)
    for ii, c1 in enumerate(keys):
        for c2 in keys[ii:]:
            members = by_label[c1] + by_label[c2] if c1 != c2 else by_label[c1]
            pair_buckets[(c1, c2)] = members
    edge_set = set(edges)

    p1 = os.path.join(CACHE_DIR, f"{backbone}_{dataset}_seed{seed}_base.npz")
    p2 = os.path.join(CACHE_DIR, f"{dataset}_seed{seed}_base.npz")
    z = np.load(p1 if os.path.exists(p1) else p2)
    base = {k: z[k] for k in pkeys}
    val_e = [tuple(e) for e in z["val_e"]]
    test_e = [tuple(e) for e in z["test_e"]]
    test_neg = [tuple(e) for e in z["test_neg"]]
    forget_e = [tuple(e) for e in z["forget_e"]]
    remain_e = [tuple(e) for e in z["remain_e"]]
    attack_neg = [tuple(e) for e in z["attack_neg"]]

    rng_a = np.random.default_rng(seed * 31 + 7)
    rn = sample_non_edges(rng_a, n, edge_set, len(remain_e), labeled)
    rng_stop = np.random.default_rng(seed * 100003 + 11)
    stop_neg = []
    for _ in range(4):
        stop_neg += hard_negatives(rng_stop, forget_e, labels, edge_set,
                                  pair_buckets)

    y_attack = np.array([1] * len(forget_e) + [0] * len(attack_neg))
    y_test = np.array([1] * len(test_e) + [0] * len(test_neg))

    print(f"[addon2 {backbone} {dataset} seed={seed}] fgs_attack "
          f"(forget={len(forget_e)})...", flush=True)
    p, t, n_steps, diag = unlearn_fgs(
        M, base, forget_e, remain_e, n, X, remain_e, rn, eta=FGS_ETA,
        val_edges=val_e, max_steps=80, stop_rule="attack", stop_neg=stop_neg)
    res["methods"]["fgs_attack"] = {
        "attack": auc_score(y_attack, M.scores_for(p, remain_e, n, X,
                                                   forget_e + attack_neg)),
        "util": auc_score(y_test, M.scores_for(p, remain_e, n, X,
                                               test_e + test_neg)),
        "s": round(t, 1), "steps": n_steps}
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    d = res["methods"]["fgs_attack"]
    print(f"  -> attack={d['attack']:.3f} util={d['util']:.3f} "
          f"steps={d['steps']} t={d['s']}s", flush=True)


if __name__ == "__main__":
    import sys
    targets = sys.argv[1:]
    if not targets:
        targets = sorted(glob.glob(os.path.join(RES_DIR, "*.json")))
    for path in targets:
        process(path)
