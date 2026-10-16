# Author: Prabhu Rangaraju
"""Mild adaptive-auditor test (fix #9): per-seed shadow retrain on CiteSeer.

The attack-calibrated stopping rule (fgs_attack) selects the iterate whose
attack ranking matches validation edges -- the same link-inference auditor
used for evaluation (a circularity risk). Here we calibrate stopping to a
DIFFERENT auditor: a shadow retrain model (same protocol, fresh init seed).
For each CiteSeer seed: run the 80-step projected trajectory, keep all
iterates, pick the iterate minimizing KS(forget-edge scores under theta_t,
forget-edge scores under the shadow retrain). Compare the shadow-chosen
iterate's attack gap to the attack-calibrated choice.

Writes results/shadow_auditor/citeseer_seed<seed>.json."""
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import numpy as np
from scipy.stats import ks_2samp
from experiment import (EPOCHS, HID, LR, FGS_ETA, FORGET_FRAC, auc_score,
                        sample_non_edges, hard_negatives)
from unlearn import unlearn_fgs, unlearn_retrain
from data import load_planetoid
import gcn as M

OUT = os.path.join(HERE, "results", "shadow_auditor")
os.makedirs(OUT, exist_ok=True)

SEEDS = [7, 42, 123, 999, 2024, 2025, 5555, 7777, 31337, 31338]


def pairs(es):
    u = np.array([e[0] for e in es], dtype=np.int64)
    v = np.array([e[1] for e in es], dtype=np.int64)
    return u, v


def run_shadow(dataset, seed):
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

    labeled_train = [e for e in train_e if labels[e[0]] >= 0 and labels[e[1]] >= 0]
    k_forget = max(10, int(FORGET_FRAC * len(train_e)))
    forget_e = [labeled_train[i] for i in rng.choice(len(labeled_train), k_forget,
                                                     replace=False)]
    remain_e = [e for e in train_e if e not in set(forget_e)]
    attack_neg = hard_negatives(rng, forget_e, labels, edge_set, pair_buckets)
    remain_neg = [e for e in train_neg]

    # shadow retrain: same protocol as gold retrain, fresh init stream
    shadow, _ = unlearn_retrain(M, remain_e, n, X, remain_neg, epochs=EPOCHS,
                                hid=HID, lr=LR,
                                rng=np.random.default_rng(seed + 555))

    # Faithful replicate of experiment.py's rng stream (reconciles with the
    # main per-seed runs): rn from the shared rng after the attack-negative
    # draw; stop_neg with the shared seen-set seeded by the evaluation
    # attack_neg (M4 fix).
    rn = sample_non_edges(rng, n, edge_set, len(remain_e), labeled)
    rng_stop = np.random.default_rng(seed * 100003 + 11)
    stop_seen = set(attack_neg)
    stop_neg = []
    for _ in range(4):
        stop_neg += hard_negatives(rng_stop, forget_e, labels, edge_set,
                                   pair_buckets, seen=stop_seen)

    # 80-step attack-calibrated trajectory, keep all iterates
    p_atk, t, n_steps, diag = unlearn_fgs(
        M, base, forget_e, remain_e, n, X, remain_e, rn, eta=FGS_ETA,
        val_edges=val_e, max_steps=80, stop_rule="attack", stop_neg=stop_neg,
        keep_iterates=True)
    iterates = diag["iterates"]
    chosen_attack = int(diag["chosen_step"])
    assert len(iterates) == 81

    # shadow-KS auditor: KS(forget scores under theta_t, forget scores under shadow)
    fu, fv = pairs(forget_e)
    s_shadow = M.pair_scores(*_emb(M, shadow, remain_e, n, X), fu, fv)
    ks_vals = []
    for prm in iterates:
        s_t = M.pair_scores(*_emb(M, prm, remain_e, n, X), fu, fv)
        ks_vals.append(float(ks_2samp(s_t, s_shadow).statistic))
    chosen_shadow = int(np.argmin(ks_vals))

    # evaluation under the FIXED attack auditor (same as the paper)
    y_attack = np.array([1] * len(forget_e) + [0] * len(attack_neg))
    au, av = pairs(attack_neg)
    atk_all = np.concatenate([fu, au])
    atk_vv = np.concatenate([fv, av])
    gold = json.load(open(os.path.join(
        HERE, "results", "per_seed",
        next(p for p in (f"gcn_{dataset}_seed{seed}.json", f"{dataset}_seed{seed}.json")
             if os.path.exists(os.path.join(HERE, "results", "per_seed", p))))))
    ret_atk = gold["methods"]["retrain"]["attack"]
    atk_shadow = float(auc_score(y_attack, M.scores_for(iterates[chosen_shadow], remain_e, n, X,
                                                        forget_e + attack_neg)))
    atk_attack = float(auc_score(y_attack, M.scores_for(iterates[chosen_attack], remain_e, n, X,
                                                        forget_e + attack_neg)))
    return {"dataset": dataset, "seed": seed,
            "chosen_attack_step": chosen_attack,
            "chosen_shadow_step": chosen_shadow,
            "ks_at_shadow_choice": ks_vals[chosen_shadow],
            "ks_at_attack_choice": ks_vals[chosen_attack],
            "attack_gap_shadow_choice": atk_shadow - ret_atk,
            "attack_gap_attack_choice": atk_attack - ret_atk}


def _emb(M_, prm, adj_edges, n, X):
    Z, _ = M_.forward(prm, M_.norm_adj(adj_edges, n), X)
    return Z,


if __name__ == "__main__":
    for seed in SEEDS:
        if len(sys.argv) > 1 and str(seed) not in sys.argv[1:]:
            continue
        print(f"[shadow] citeseer seed={seed}", flush=True)
        r = run_shadow("citeseer", seed)
        print("   attack_step=%d shadow_step=%d atk_gap(shadow)=%+.3f atk_gap(attack)=%+.3f"
              % (r["chosen_attack_step"], r["chosen_shadow_step"],
                 r["attack_gap_shadow_choice"], r["attack_gap_attack_choice"]), flush=True)
        with open(os.path.join(OUT, f"citeseer_seed{seed}.json"), "w") as f:
            json.dump(r, f, indent=1)
