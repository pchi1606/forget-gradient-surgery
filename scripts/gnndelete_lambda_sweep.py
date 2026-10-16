# Author: Prabhu Rangaraju
"""GNNDelete lambda sensitivity sweep (reviewer request, loop-2 fixes).

Replicates experiment.py's exact rng call sequence (permutation, negative
sampling, base training, forget-set/attack-negative/hard-negative sampling)
so that per-seed eval sets are IDENTICAL to the stored per_seed JSONs; the
lambda=1.0 point is validated against the stored gnndelete attack/util AUCs
before any other lambda is trusted. Only the GNNDelete training (lambda)
varies.

Usage: python3 scripts/gnndelete_lambda_sweep.py [out.json]
Writes results/gnndelete_lambda_sweep.json and per-seed records.
"""
import json
import os
import sys
import time

import numpy as np
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import gcn as M
from data import load_planetoid
from gnndelete import unlearn_gnndelete, scores_with_operator

EPOCHS = 150
HID = 32
LR = 0.01
FORGET_FRAC = 0.05
GD_EPOCHS = 40
GD_LR = 0.01
GD_NI = 2048
LAMS = (0.25, 0.5, 1.0, 2.0, 4.0)
DATASETS = ("cora", "citeseer", "pubmed")
SEEDS = (42, 7, 123, 2024, 31337, 999, 2025, 5555, 7777, 31338)


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


def hard_negatives(rng, forget_edges, labels, edge_set, pair_buckets):
    out, seen = [], set()
    for (u, v) in forget_edges:
        key = tuple(sorted((int(labels[u]), int(labels[v]))))
        cands = pair_buckets.get(key)
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


def seed_path(dataset, seed):
    p = os.path.join(HERE, "results", "per_seed",
                     f"gcn_{dataset}_seed{seed}.json")
    if not os.path.exists(p):
        p = os.path.join(HERE, "results", "per_seed",
                         f"{dataset}_seed{seed}.json")
    return p


def run_seed(dataset, seed, lams=LAMS):
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

    base, _ = M.train(train_e, n, X, train_neg, EPOCHS, lr=LR, hid=HID,
                      rng=rng, verbose=False)

    labeled_train = [e for e in train_e
                     if labels[e[0]] >= 0 and labels[e[1]] >= 0]
    k_forget = max(10, int(FORGET_FRAC * len(train_e)))
    forget_e = [labeled_train[i] for i in rng.choice(len(labeled_train),
                                                    k_forget, replace=False)]
    forget_set = set(forget_e)
    remain_e = [e for e in train_e if e not in forget_set]
    attack_neg = hard_negatives(rng, forget_e, labels, edge_set, pair_buckets)

    y_attack = np.array([1] * len(forget_e) + [0] * len(attack_neg))
    y_test = np.array([1] * len(test_e) + [0] * len(test_neg))

    # rng sequence to the gnndelete step (rn sampling has no other draws in
    # between; finetune/grad_ascent/fgs do not consume rng)
    rp = remain_e
    rn = sample_non_edges(rng, n, edge_set, len(rp), labeled)
    k_ni = min(GD_NI, len(remain_e))
    ni_pos = [remain_e[i] for i in rng.choice(len(remain_e), k_ni,
                                              replace=False)]
    ni_neg = sample_non_edges(rng, n, edge_set, k_ni, labeled)

    stored = json.load(open(seed_path(dataset, seed)))["methods"]["gnndelete"]

    rec = {"dataset": dataset, "seed": seed,
           "lam1_vs_stored": None, "lams": {}}
    for lam in lams:
        phi, aff, t = unlearn_gnndelete(M, "gcn", base, forget_e, remain_e,
                                        n, X, ni_pos, ni_neg, epochs=GD_EPOCHS,
                                        lr=GD_LR, lam=lam, seed=seed)
        atk = auc_score(y_attack, scores_with_operator(
            M, "gcn", base, phi, aff, remain_e, n, X, forget_e + attack_neg))
        utl = auc_score(y_test, scores_with_operator(
            M, "gcn", base, phi, aff, remain_e, n, X, test_e + test_neg))
        rec["lams"][str(lam)] = {"attack": atk, "util": utl,
                                 "s": round(t, 1)}
        if lam == 1.0:
            rec["lam1_vs_stored"] = {
                "attack_diff": abs(atk - stored["attack"]),
                "util_diff": abs(utl - stored["util"])}
    return rec


def main():
    out = os.path.join(HERE, "results", "gnndelete_lambda_sweep.json")
    if len(sys.argv) > 1:
        out = sys.argv[1]
    recs = []
    for ds in DATASETS:
        for seed in SEEDS:
            t0 = time.time()
            rec = run_seed(ds, seed)
            recs.append(rec)
            d1 = rec["lam1_vs_stored"]
            print(f"[{ds} seed={seed}] {time.time()-t0:.0f}s "
                  f"lam1 attack diff={d1['attack_diff']:.2e} "
                  f"util diff={d1['util_diff']:.2e}", flush=True)
            json.dump(recs, open(out, "w"), indent=1)
    json.dump(recs, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
