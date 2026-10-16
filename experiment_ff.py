"""Forget-fraction sensitivity driver.

Reuses the cached base model + splits for (backbone, dataset, seed) and
reruns the full unlearning protocol at a different forget fraction.

Env: BACKBONE=gcn|sage (default gcn), FORGET_FRAC in {0.01, 0.10, 0.20}.
The forget set is drawn with a fresh rng seeded by
seed*1000 + round(1000*frac) -- deterministic and documented; it is NOT the
same forget set as the 5% runs (which used the in-flight rng).

Output: results/per_seed_ff/<backbone>_<dataset>_seed<seed>_ff<frac>.json
"""
import json
import os
import time

import numpy as np
from scipy.stats import rankdata

from data import load_planetoid
from unlearn import (unlearn_fgs, unlearn_finetune, unlearn_grad_ascent,
                     unlearn_retrain)
from gnndelete import unlearn_gnndelete, scores_with_operator
from experiment import sample_non_edges, hard_negatives

HERE = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.join(HERE, "results", "per_seed_ff")
CACHE_DIR = os.path.join(HERE, "results", "cache")

BACKBONE = os.environ.get("BACKBONE", "gcn")
FORGET_FRAC = float(os.environ.get("FORGET_FRAC", "0.10"))
if BACKBONE == "gcn":
    import gcn as M
    PKEYS = ("W1", "W2")
elif BACKBONE == "sage":
    import sage as M
    PKEYS = ("W1n", "W1s", "W2n", "W2s")
else:
    raise ValueError(f"unknown BACKBONE={BACKBONE}")

EPOCHS = 150
HID = 32
LR = 0.01
FT_EPOCHS = 40
GA_STEPS = 15
GA_LR = 0.005
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
    t_all = time.time()
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

    # cache: new naming first, legacy (gcn, no prefix) fallback
    p1 = os.path.join(CACHE_DIR, f"{BACKBONE}_{dataset}_seed{seed}_base.npz")
    p2 = os.path.join(CACHE_DIR, f"{dataset}_seed{seed}_base.npz")
    z = np.load(p1 if os.path.exists(p1) else p2)
    base = {k: z[k] for k in PKEYS}
    train_e = [tuple(e) for e in z["train_e"]]
    val_e = [tuple(e) for e in z["val_e"]]
    test_e = [tuple(e) for e in z["test_e"]]
    train_neg = [tuple(e) for e in z["train_neg"]]
    test_neg = [tuple(e) for e in z["test_neg"]]

    rng = np.random.default_rng(seed * 1000 + round(1000 * FORGET_FRAC))
    labeled_train = [e for e in train_e
                     if labels[e[0]] >= 0 and labels[e[1]] >= 0]
    k_forget = max(10, int(FORGET_FRAC * len(train_e)))
    forget_e = [labeled_train[i] for i in rng.choice(len(labeled_train),
                                                     k_forget, replace=False)]
    forget_set = set(forget_e)
    remain_e = [e for e in train_e if e not in forget_set]
    remain_neg = [e for e in train_neg]
    attack_neg = hard_negatives(rng, forget_e, labels, edge_set, pair_buckets)
    rn = sample_non_edges(rng, n, edge_set, len(remain_e), labeled)
    print(f"[ff {BACKBONE} {dataset} seed={seed} frac={FORGET_FRAC}] "
          f"forget={len(forget_e)} remain={len(remain_e)}")

    y_attack = np.array([1] * len(forget_e) + [0] * len(attack_neg))
    y_test = np.array([1] * len(test_e) + [0] * len(test_neg))

    def attack_auc(params, adj_edges):
        return auc_score(y_attack, M.scores_for(params, adj_edges, n, X,
                                                forget_e + attack_neg))

    def util_auc(params, adj_edges):
        return auc_score(y_test, M.scores_for(params, adj_edges, n, X,
                                              test_e + test_neg))

    res = {"backbone": BACKBONE, "dataset": dataset, "seed": seed,
           "forget_frac": FORGET_FRAC, "n_nodes": n, "n_edges": len(edges),
           "n_train": len(train_e), "n_forget": len(forget_e), "methods": {}}

    print("  retrain (gold)...")
    p, t = unlearn_retrain(M, remain_e, n, X, remain_neg, epochs=EPOCHS,
                           hid=HID, lr=LR,
                           rng=np.random.default_rng(seed + 999))
    res["methods"]["retrain"] = {"attack": attack_auc(p, remain_e),
                                 "util": util_auc(p, remain_e), "s": round(t, 1)}
    print("  finetune...")
    p, t = unlearn_finetune(M, base, remain_e, n, X, remain_neg,
                            epochs=FT_EPOCHS, lr=LR)
    res["methods"]["finetune"] = {"attack": attack_auc(p, remain_e),
                                  "util": util_auc(p, remain_e), "s": round(t, 1)}
    print("  grad_ascent...")
    p, t = unlearn_grad_ascent(M, base, forget_e, remain_e, n, X,
                               steps=GA_STEPS, lr=GA_LR)
    res["methods"]["grad_ascent"] = {"attack": attack_auc(p, remain_e),
                                     "util": util_auc(p, remain_e), "s": round(t, 1)}
    print("  fgs (KS stopping)...")
    p, t, n_steps, diag = unlearn_fgs(M, base, forget_e, remain_e, n, X,
                                      remain_e, rn, eta=FGS_ETA,
                                      val_edges=val_e, stop_rule="ks")
    res["methods"]["fgs"] = {"attack": attack_auc(p, remain_e),
                             "util": util_auc(p, remain_e), "s": round(t, 1),
                             "steps": n_steps}
    print("  fgs_margin (KS+margin stopping)...")
    p, t, n_steps, diag = unlearn_fgs(M, base, forget_e, remain_e, n, X,
                                      remain_e, rn, eta=FGS_ETA,
                                      val_edges=val_e, stop_rule="ks_margin")
    res["methods"]["fgs_margin"] = {
        "attack": attack_auc(p, remain_e), "util": util_auc(p, remain_e),
        "s": round(t, 1), "steps": n_steps,
        "fallback": bool(diag.get("fallback", False))}
    print("  fgs_attack (attack-calibrated stopping)...")
    rng_stop = np.random.default_rng(seed * 100003 + 11)
    # M4 fix: stop_neg disjoint from evaluation attack_neg (shared seen set).
    stop_seen = set(attack_neg)
    stop_neg = []
    for _ in range(4):
        stop_neg += hard_negatives(rng_stop, forget_e, labels, edge_set,
                                  pair_buckets, seen=stop_seen)
    p, t, n_steps, diag = unlearn_fgs(M, base, forget_e, remain_e, n, X,
                                      remain_e, rn, eta=FGS_ETA,
                                      val_edges=val_e, max_steps=80,
                                      stop_rule="attack", stop_neg=stop_neg)
    res["methods"]["fgs_attack"] = {
        "attack": attack_auc(p, remain_e), "util": util_auc(p, remain_e),
        "s": round(t, 1), "steps": n_steps}
    print("  gnndelete...")
    k_ni = min(GD_NI, len(remain_e))
    ni_pos = [remain_e[i] for i in rng.choice(len(remain_e), k_ni, replace=False)]
    ni_neg = sample_non_edges(rng, n, edge_set, k_ni, labeled)
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

    os.makedirs(RES_DIR, exist_ok=True)
    path = os.path.join(
        RES_DIR, f"{BACKBONE}_{dataset}_seed{seed}_ff{FORGET_FRAC:.2f}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    print(f"  total {time.time() - t_all:.1f}s -> {path}")
    for m, d in res["methods"].items():
        print(f"    {m:<11} attack={d['attack']:.3f} util={d['util']:.3f} t={d['s']}s")
    return res


if __name__ == "__main__":
    import sys
    run(sys.argv[1], int(sys.argv[2]))
