"""Shared harness for the reproduction-package scripts.

Rebuilds one (dataset, seed) configuration exactly as code/experiment.py
does -- same rng stream, same splits, same negative sampling, same base
training -- so the scripts in this directory run the frozen pipeline code
in code/ rather than reimplementing it. The rng replay order mirrors
experiment.py's run(): permutation, train/val/test negatives, base-model
training, forget-set choice, attack negatives; later consumers (the
retain negatives for FGS, the attack-rule stop negatives) are replayed by
the callers that need them, in the same order experiment.py draws them.
"""
import os
import sys

import numpy as np
from scipy.stats import rankdata

CODE_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "code")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

EPOCHS = 150
HID = 32
LR = 0.01
FGS_ETA = 0.05
FORGET_FRAC = 0.05


def auc_score(y, s):
    r = rankdata(s)
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def build_config(dataset, seed, train_base=True):
    """Replay experiment.py's construction for one configuration.

    Returns a dict with the split, the forget/retain sets, the negative
    sets, the trained base model (if train_base), and the rng positioned
    exactly where experiment.py's is after the attack negatives are drawn
    (i.e. just before the retain negatives for FGS are sampled).
    """
    from data import load_planetoid
    from experiment import sample_non_edges, hard_negatives
    import gcn as M

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

    base, t_base = (None, None)
    if train_base:
        base, t_base = M.train(train_e, n, X, train_neg, EPOCHS, lr=LR,
                               hid=HID, rng=rng, verbose=False)
    else:
        # experiment.py's base training consumes rng draws (init_params);
        # replay the same draws so downstream sampling is identical.
        M.init_params(X.shape[1], HID, rng)

    labeled_train = [e for e in train_e
                     if labels[e[0]] >= 0 and labels[e[1]] >= 0]
    k_forget = max(10, int(FORGET_FRAC * len(train_e)))
    forget_e = [labeled_train[i] for i in
                rng.choice(len(labeled_train), k_forget, replace=False)]
    forget_set = set(forget_e)
    remain_e = [e for e in train_e if e not in forget_set]
    attack_neg = hard_negatives(rng, forget_e, labels, edge_set, pair_buckets)

    return {
        "M": M, "rng": rng, "X": X, "labels": labels, "n": n,
        "labeled": labeled, "edge_set": edge_set,
        "pair_buckets": pair_buckets,
        "train_e": train_e, "val_e": val_e, "test_e": test_e,
        "train_neg": train_neg, "val_neg": val_neg, "test_neg": test_neg,
        "forget_e": forget_e, "remain_e": remain_e,
        "attack_neg": attack_neg, "base": base, "base_train_s": t_base,
    }


def attack_auc(cfg, params, adj_edges):
    y = np.array([1] * len(cfg["forget_e"]) + [0] * len(cfg["attack_neg"]))
    return auc_score(y, cfg["M"].scores_for(
        params, adj_edges, cfg["n"], cfg["X"],
        cfg["forget_e"] + cfg["attack_neg"]))


def util_auc(cfg, params, adj_edges):
    y = np.array([1] * len(cfg["test_e"]) + [0] * len(cfg["test_neg"]))
    return auc_score(y, cfg["M"].scores_for(
        params, adj_edges, cfg["n"], cfg["X"],
        cfg["test_e"] + cfg["test_neg"]))


def draw_retain_negatives(cfg):
    """The rn draw experiment.py makes just before running FGS."""
    from experiment import sample_non_edges
    return sample_non_edges(cfg["rng"], cfg["n"], cfg["edge_set"],
                            len(cfg["remain_e"]), cfg["labeled"])


def draw_stop_negatives(cfg, seed):
    """The attack rule's stop negatives (experiment.py's M4 construction):
    4x hard negatives from rng seed*100003+11, disjoint from attack_neg."""
    from experiment import hard_negatives
    rng_stop = np.random.default_rng(seed * 100003 + 11)
    seen = set(cfg["attack_neg"])
    out = []
    for _ in range(4):
        out += hard_negatives(rng_stop, cfg["forget_e"], cfg["labels"],
                              cfg["edge_set"], cfg["pair_buckets"], seen=seen)
    return out


def archived_per_seed(dataset, seed, root=None):
    """Load the archived per-seed JSON for a GCN configuration."""
    import json
    if root is None:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name in (f"gcn_{dataset}_seed{seed}.json", f"{dataset}_seed{seed}.json"):
        path = os.path.join(root, "results", "per_seed", name)
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    raise FileNotFoundError(f"no archived per-seed JSON for {dataset} seed {seed}")
