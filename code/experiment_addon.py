"""Add-on driver: run the NEW methods (fgs_margin, fgs_attack, gnndelete) on
the EXISTING
GCN base runs (seeds 42/7/123, cache written by the original protocol) and
append results to the existing per-seed JSONs.

The original methods (retrain/finetune/grad_ascent/fgs) are untouched: the
fgs code path is identical, so those numbers are reused as-is.
"""
import json
import os
import time

import numpy as np
from scipy.stats import rankdata

from data import load_planetoid
from unlearn import unlearn_fgs
from gnndelete import unlearn_gnndelete, scores_with_operator
import gcn as M
from experiment import sample_non_edges, hard_negatives

HERE = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.join(HERE, "results", "per_seed")
CACHE_DIR = os.path.join(HERE, "results", "cache")

BACKBONE = "gcn"
FGS_ETA = float(os.environ.get("FGS_ETA", "0.05"))
GD_EPOCHS = 40
GD_LR = 0.01
GD_NI = 2048


def auc_score(y, s):
    r = rankdata(s)
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def run(dataset, seed):
    rng = np.random.default_rng(seed)
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

    z = np.load(os.path.join(CACHE_DIR, f"{dataset}_seed{seed}_base.npz"))
    base = {k: z[k] for k in ("W1", "W2")}
    train_e = [tuple(e) for e in z["train_e"]]
    val_e = [tuple(e) for e in z["val_e"]]
    test_e = [tuple(e) for e in z["test_e"]]
    test_neg = [tuple(e) for e in z["test_neg"]]
    forget_e = [tuple(e) for e in z["forget_e"]]
    remain_e = [tuple(e) for e in z["remain_e"]]
    attack_neg = [tuple(e) for e in z["attack_neg"]]

    # retain anchor negatives: resampled deterministically (documented; the
    # original anchor used the in-flight rng state which is not recoverable)
    rng_a = np.random.default_rng(seed * 31 + 7)
    rn = sample_non_edges(rng_a, n, edge_set, len(remain_e), labeled)

    y_attack = np.array([1] * len(forget_e) + [0] * len(attack_neg))
    y_test = np.array([1] * len(test_e) + [0] * len(test_neg))

    path = os.path.join(RES_DIR, f"{dataset}_seed{seed}.json")
    res = json.load(open(path))

    print(f"[addon gcn {dataset} seed={seed}] fgs_margin...")
    p, t, n_steps, diag = unlearn_fgs(M, base, forget_e, remain_e, n, X,
                                      remain_e, rn, eta=FGS_ETA,
                                      val_edges=val_e, stop_rule="ks_margin")
    res["methods"]["fgs_margin"] = {
        "attack": auc_score(y_attack, M.scores_for(p, remain_e, n, X,
                                                   forget_e + attack_neg)),
        "util": auc_score(y_test, M.scores_for(p, remain_e, n, X,
                                               test_e + test_neg)),
        "s": round(t, 1), "steps": n_steps,
        "fallback": bool(diag.get("fallback", False))}

    print(f"[addon gcn {dataset} seed={seed}] fgs_attack...")
    rng_stop = np.random.default_rng(seed * 100003 + 11)
    stop_neg = []
    for _ in range(4):  # 4x forget size: stable surrogate/reference AUCs
        stop_neg += hard_negatives(rng_stop, forget_e, labels, edge_set,
                                  pair_buckets)
    p, t, n_steps, diag = unlearn_fgs(M, base, forget_e, remain_e, n, X,
                                      remain_e, rn, eta=FGS_ETA,
                                      val_edges=val_e, max_steps=80,
                                      stop_rule="attack", stop_neg=stop_neg)
    res["methods"]["fgs_attack"] = {
        "attack": auc_score(y_attack, M.scores_for(p, remain_e, n, X,
                                                   forget_e + attack_neg)),
        "util": auc_score(y_test, M.scores_for(p, remain_e, n, X,
                                               test_e + test_neg)),
        "s": round(t, 1), "steps": n_steps}

    print(f"[addon gcn {dataset} seed={seed}] gnndelete...")
    rng_b = np.random.default_rng(seed * 131 + 17)
    k_ni = min(GD_NI, len(remain_e))
    ni_pos = [remain_e[i] for i in rng_b.choice(len(remain_e), k_ni, replace=False)]
    ni_neg = sample_non_edges(rng_b, n, edge_set, k_ni, labeled)
    phi, aff, t = unlearn_gnndelete(M, BACKBONE, base, forget_e, remain_e,
                                    n, X, ni_pos, ni_neg, epochs=GD_EPOCHS,
                                    lr=GD_LR, seed=seed)
    res["methods"]["gnndelete"] = {
        "attack": auc_score(y_attack,
                            scores_with_operator(M, BACKBONE, base, phi, aff,
                                                 remain_e, n, X,
                                                 forget_e + attack_neg)),
        "util": auc_score(y_test,
                          scores_with_operator(M, BACKBONE, base, phi, aff,
                                               remain_e, n, X,
                                               test_e + test_neg)),
        "s": round(t, 1)}

    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    for m in ("fgs_margin", "fgs_attack", "gnndelete"):
        d = res["methods"][m]
        print(f"    {m:<11} attack={d['attack']:.3f} util={d['util']:.3f} t={d['s']}s")
    return res


if __name__ == "__main__":
    import sys
    run(sys.argv[1], int(sys.argv[2]))
