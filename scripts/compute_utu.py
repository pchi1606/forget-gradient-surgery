# Author: Prabhu Rangaraju
"""UtU (Unlink to Unlearn) baseline: base model evaluated on the post-deletion
adjacency remain_e. Replicates the data-split and base-training code path of
experiment.run() verbatim (same RNG sequence), so splits/base params match the
stored per-seed JSONs; writes results/utu/<dataset>_seed<seed>.json."""
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import numpy as np
from experiment import (BACKBONE, EPOCHS, HID, LR, FORGET_FRAC, auc_score,
                        sample_non_edges, hard_negatives)
from data import load_planetoid
import gcn as M

OUT = os.path.join(HERE, "results", "utu")
os.makedirs(OUT, exist_ok=True)

CONFIGS = [("cora", s) for s in [7, 42, 123, 999, 2024, 2025, 5555, 7777, 31337, 31338]] + \
          [("citeseer", s) for s in [7, 42, 123, 999, 2024, 2025, 5555, 7777, 31337, 31338]] + \
          [("pubmed", s) for s in [7, 42, 123, 999, 2024, 2025, 5555, 7777, 31337, 31338]]


def run_utu(dataset, seed):
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

    base, t_base = M.train(train_e, n, X, train_neg, EPOCHS, lr=LR, hid=HID,
                           rng=rng, verbose=False)

    labeled_train = [e for e in train_e if labels[e[0]] >= 0 and labels[e[1]] >= 0]
    k_forget = max(10, int(FORGET_FRAC * len(train_e)))
    forget_e = [labeled_train[i] for i in rng.choice(len(labeled_train), k_forget,
                                                     replace=False)]
    remain_e = [e for e in train_e if e not in set(forget_e)]
    attack_neg = hard_negatives(rng, forget_e, labels, edge_set, pair_buckets)

    y_attack = np.array([1] * len(forget_e) + [0] * len(attack_neg))
    y_test = np.array([1] * len(test_e) + [0] * len(test_neg))

    # UtU: base (trained WITH forget edges) evaluated on post-deletion adjacency
    utu_attack = auc_score(y_attack, M.scores_for(base, remain_e, n, X,
                                                  forget_e + attack_neg))
    utu_util = auc_score(y_test, M.scores_for(base, remain_e, n, X,
                                               test_e + test_neg))
    base_test = auc_score(y_test, M.scores_for(base, train_e, n, X,
                                                test_e + test_neg))
    # sanity: base_test must match the stored base_test_auc (same code path)
    return {"dataset": dataset, "seed": seed,
            "utu_attack": float(utu_attack), "utu_util": float(utu_util),
            "base_test_auc": float(base_test), "s": round(t_base, 1)}


if __name__ == "__main__":
    only = sys.argv[1:]  # optional: "citeseer" or "citeseer:7"
    for ds, seed in CONFIGS:
        if only and not any(o in (ds, f"{ds}:{seed}") for o in only):
            continue
        print(f"[utu] {ds} seed={seed}", flush=True)
        r = run_utu(ds, seed)
        # sanity check vs stored base_test_auc
        for pat in (f"gcn_{ds}_seed{seed}.json", f"{ds}_seed{seed}.json"):
            p = os.path.join(HERE, "results", "per_seed", pat)
            if os.path.exists(p):
                old = json.load(open(p))
                d = abs(r["base_test_auc"] - old["base_test_auc"])
                print(f"   base_test_auc check: stored={old['base_test_auc']} "
                      f"recomputed={r['base_test_auc']:.4f} |diff|={d:.2e}", flush=True)
                break
        with open(os.path.join(OUT, f"{ds}_seed{seed}.json"), "w") as f:
            json.dump(r, f, indent=1)
