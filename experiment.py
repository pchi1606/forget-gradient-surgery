# Author: Prabhu Rangaraju
"""Per-(backbone, dataset, seed) experiment driver.

Protocol (Lethe-style):
  1. Load a public Planetoid graph; split edges 70/15/15 (train/val/test);
     sample equal-sized negative sets per split (labeled nodes only).
  2. Train a 2-layer link predictor (BACKBONE=gcn|GraphSAGE, env BACKBONE)
     on train edges (base model).
  3. Forget set: 5% of train edges (both endpoints labeled).
  4. Unlearn with: retrain (gold), finetune, grad_ascent, fgs (novel, KS
     stopping), fgs_margin (novel, KS+margin robust stopping),
     fgs_attack (novel, attack-calibrated robust stopping), gnndelete
     (GNNDelete-style deletion-operator baseline, see gnndelete.py),
     utu (unlink-only baseline: base params on post-deletion adjacency).
  5. Metrics per method:
       attack AUC : model scores separate FORGOTTEN edges from HARD
                    non-edges (label-pair matched). Lower = better forgotten;
                    retrain's value is the reference for "forgotten".
       utility AUC: link-prediction AUC on held-out test edges.
       seconds    : wall-clock unlearning time (retrain = full retraining).
Results are written to results/per_seed/<backbone>_<dataset>_seed<seed>.json
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

HERE = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.join(HERE, "results", "per_seed")

BACKBONE = os.environ.get("BACKBONE", "gcn")
if BACKBONE == "gcn":
    import gcn as M
elif BACKBONE == "sage":
    import sage as M
else:
    raise ValueError(f"unknown BACKBONE={BACKBONE}")

EPOCHS = 150
HID = 32
LR = 0.01
FT_EPOCHS = 40
GA_STEPS = 15
GA_LR = 0.005
FGS_ETA = float(os.environ.get("FGS_ETA", "0.05"))
FGS_RETAIN = 512
FORGET_FRAC = 0.05
GD_EPOCHS = 40
GD_LR = 0.01
GD_NI = 2048  # NI sample size per class for the GNNDelete-style baseline


def auc_score(y, s):
    r = rankdata(s)
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def sample_non_edges(rng, n, edge_set, k, allowed):
    allowed = np.asarray(allowed)
    out, seen = [], set()
    while len(out) < k:
        a = allowed[rng.integers(0, len(allowed))]
        b = allowed[rng.integers(0, len(allowed))]
        if a == b:
            continue
        e = (int(a), int(b)) if a < b else (int(b), int(a))
        if e in edge_set or e in seen:
            continue
        seen.add(e)
        out.append(e)
    return out


def hard_negatives(rng, forget_edges, labels, edge_set, by_label, seen=None):
    """One non-edge per forget edge, matched on the sorted label pair.

    `seen`: optional set shared across calls. When provided, sampled
    negatives are drawn disjoint from it (and added to it), enforcing
    disjointness between e.g. evaluation attack negatives and stopping-rule
    negatives. When None, a fresh per-call set is used (within-call
    dedup only).
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


def run(dataset, seed):
    rng = np.random.default_rng(seed)
    t_all = time.time()
    Xsp, labels, edges = load_planetoid(dataset)
    n = Xsp.shape[0]
    X = Xsp.toarray().astype(np.float32)
    labeled = np.where(labels >= 0)[0]
    by_label = {}
    lab = labels[labeled]
    for i, nd in enumerate(labeled):
        by_label.setdefault(int(lab[i]), []).append(int(nd))
    # label-pair buckets for hard negatives
    pair_buckets = {}
    keys = sorted(by_label)
    for ii, c1 in enumerate(keys):
        for c2 in keys[ii:]:
            members = by_label[c1] + by_label[c2] if c1 != c2 else by_label[c1]
            pair_buckets[(c1, c2)] = members

    perm = rng.permutation(len(edges))
    edges = [edges[i] for i in perm]
    n_train = int(0.70 * len(edges))
    n_val = int(0.15 * len(edges))
    train_e = edges[:n_train]
    val_e = edges[n_train:n_train + n_val]
    test_e = edges[n_train + n_val:]
    edge_set = set(edges)

    train_neg = sample_non_edges(rng, n, edge_set, len(train_e), labeled)
    val_neg = sample_non_edges(rng, n, edge_set, len(val_e), labeled)
    test_neg = sample_non_edges(rng, n, edge_set, len(test_e), labeled)

    print(f"[{BACKBONE} {dataset} seed={seed}] train={len(train_e)} "
          f"val={len(val_e)} test={len(test_e)}")

    print("  training base model...")
    base, t_base = M.train(train_e, n, X, train_neg, EPOCHS, lr=LR, hid=HID,
                           rng=rng, verbose=False)
    base_test = auc_score(
        np.array([1] * len(test_e) + [0] * len(test_neg)),
        M.scores_for(base, train_e, n, X, test_e + test_neg))
    print(f"  base: {t_base:.1f}s  test AUC={base_test:.3f}")

    # forget set: 5% of train edges, both endpoints labeled
    labeled_train = [e for e in train_e if labels[e[0]] >= 0 and labels[e[1]] >= 0]
    k_forget = max(10, int(FORGET_FRAC * len(train_e)))
    forget_e = [labeled_train[i] for i in rng.choice(len(labeled_train), k_forget,
                                                     replace=False)]
    forget_set = set(forget_e)
    remain_e = [e for e in train_e if e not in forget_set]
    remain_neg = [e for e in train_neg]  # negatives unaffected by edge deletion
    attack_neg = hard_negatives(rng, forget_e, labels, edge_set, pair_buckets)
    print(f"  forget={len(forget_e)} remain={len(remain_e)} attack_neg={len(attack_neg)}")
    # persist everything needed for fast unlearning-variant iteration
    os.makedirs(os.path.join(HERE, "results", "cache"), exist_ok=True)
    np.savez(os.path.join(HERE, "results", "cache",
                          f"{BACKBONE}_{dataset}_seed{seed}_base.npz"),
             **{k: v for k, v in base.items()},
             train_e=np.array(train_e), val_e=np.array(val_e),
             test_e=np.array(test_e), train_neg=np.array(train_neg),
             test_neg=np.array(test_neg), forget_e=np.array(forget_e),
             remain_e=np.array(remain_e), attack_neg=np.array(attack_neg))

    y_attack = np.array([1] * len(forget_e) + [0] * len(attack_neg))
    y_test = np.array([1] * len(test_e) + [0] * len(test_neg))

    def attack_auc(params, adj_edges):
        return auc_score(y_attack, M.scores_for(params, adj_edges, n, X,
                                                forget_e + attack_neg))

    def util_auc(params, adj_edges):
        return auc_score(y_test, M.scores_for(params, adj_edges, n, X,
                                              test_e + test_neg))

    res = {"backbone": BACKBONE, "dataset": dataset, "seed": seed,
           "n_nodes": n, "n_edges": len(edges),
           "n_train": len(train_e), "n_forget": len(forget_e),
           "base_train_s": round(t_base, 1), "base_test_auc": round(base_test, 4),
           "base_attack_auc": round(attack_auc(base, train_e), 4),
           "methods": {}}

    print("  retrain (gold)...")
    p, t = unlearn_retrain(M, remain_e, n, X, remain_neg, epochs=EPOCHS,
                           hid=HID, lr=LR, rng=np.random.default_rng(seed + 999))
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
    rp = remain_e  # full retained edge set: deterministic, stable anchor
    rn = sample_non_edges(rng, n, edge_set, len(rp), labeled)
    print("  fgs (novel, KS stopping)...")
    p, t, n_steps, diag = unlearn_fgs(M, base, forget_e, remain_e, n, X, rp,
                                      rn, eta=FGS_ETA, val_edges=val_e,
                                      stop_rule="ks")
    res["methods"]["fgs"] = {"attack": attack_auc(p, remain_e),
                             "util": util_auc(p, remain_e), "s": round(t, 1),
                             "steps": n_steps,
                             "diag_ks": [round(v, 4) for v in diag["ks"]],
                             "diag_mean_f": [round(v, 4) for v in diag["mean_f"]],
                             "diag_mean_v": [round(v, 4) for v in diag["mean_v"]]}
    print("  fgs_margin (novel, KS+margin robust stopping)...")
    p, t, n_steps, diag = unlearn_fgs(M, base, forget_e, remain_e, n, X, rp,
                                      rn, eta=FGS_ETA, val_edges=val_e,
                                      stop_rule="ks_margin")
    res["methods"]["fgs_margin"] = {
        "attack": attack_auc(p, remain_e), "util": util_auc(p, remain_e),
        "s": round(t, 1), "steps": n_steps,
        "fallback": bool(diag.get("fallback", False))}
    print("  fgs_attack (novel, attack-calibrated stopping)...")
    rng_stop = np.random.default_rng(seed * 100003 + 11)
    # M4 fix: stop_neg must be disjoint from the evaluation attack_neg
    # (and internally disjoint across the 4 batches) — share one seen set.
    stop_seen = set(attack_neg)
    stop_neg = []
    for _ in range(4):  # 4x forget size: stable surrogate/reference AUCs
        stop_neg += hard_negatives(rng_stop, forget_e, labels, edge_set,
                                  pair_buckets, seen=stop_seen)
    p, t, n_steps, diag = unlearn_fgs(M, base, forget_e, remain_e, n, X, rp,
                                      rn, eta=FGS_ETA, val_edges=val_e,
                                      max_steps=80, stop_rule="attack",
                                      stop_neg=stop_neg)
    res["methods"]["fgs_attack"] = {
        "attack": attack_auc(p, remain_e), "util": util_auc(p, remain_e),
        "s": round(t, 1), "steps": n_steps}
    print("  gnndelete (GNNDelete-style deletion operator)...")
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

    # UtU (Unlink to Unlearn): no parameter change at all — the base model
    # (trained WITH the forget edges) evaluated on the post-deletion
    # adjacency. Pure structural deletion; tests whether unlinking alone
    # suffices for parametric indistinguishability from retraining.
    print("  utu (unlink-only: base params, post-deletion adjacency)...")
    res["methods"]["utu"] = {"attack": attack_auc(base, remain_e),
                             "util": util_auc(base, remain_e), "s": 0.0}

    os.makedirs(RES_DIR, exist_ok=True)
    path = os.path.join(RES_DIR, f"{BACKBONE}_{dataset}_seed{seed}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    print(f"  total {time.time() - t_all:.1f}s -> {path}")
    for m, d in res["methods"].items():
        print(f"    {m:<11} attack={d['attack']:.3f} util={d['util']:.3f} t={d['s']}s")
    return res


if __name__ == "__main__":
    import sys
    run(sys.argv[1], int(sys.argv[2]))
