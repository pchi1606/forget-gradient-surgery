"""Cheng et al. (GNNDelete, ICLR 2023) AUROC(E_d) statistic computed for
the FGS harness's GNNDelete NumPy port and for the harness retrain model,
under the FGS protocol — for the port-validation record
(battery/gnndelete-port-validation-2026-11-04.md).

Cheng et al.'s AUROC(E_d) (their Appendix C.1): deleted edges labeled 0,
an equal number of sampled remaining edges labeled 1, AUC over the
model's predicted probabilities -> P(score(remaining) > score(deleted)).
Higher = deleted edges score systematically below remaining edges.

Protocol notes (all disclosed in the validation record):
  * Base/retrain/port use the harness's own functions and constants
    (experiment.py: EPOCHS=150, HID=32, LR=0.01, GD_EPOCHS=40, GD_LR=0.01,
    GD_NI=2048, lam=1.0 default) and the harness's persisted caches.
  * Retrain is re-run with the harness's exact call
    (rng = default_rng(seed + 999), remain_neg = train_neg from cache).
  * The port's NI sample uses a fresh default_rng(seed) (the harness's
    original NI draw came from its sequential rng stream and is not
    recoverable from the cache alone); seed passed to the port = run seed,
    as in the harness.
  * The remaining-edge comparison sample for AUROC(E_d) is drawn once
    per (dataset, seed) with default_rng(20261104) and shared by both
    models (paired comparison).
Usage: python gnndelete_cheng_metric.py <dataset> <seed> [<seed> ...]
Writes results/gnndelete_validation/<ds>_seed<seed>.json (new files).
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FGS_ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, FGS_ROOT)

import gcn as M
from data import load_planetoid
from experiment import (auc_score, sample_non_edges, EPOCHS, HID, LR,
                        GD_EPOCHS, GD_LR, GD_NI)
from unlearn import unlearn_retrain
from gnndelete import unlearn_gnndelete, scores_with_operator

OUT_DIR = os.path.join(FGS_ROOT, "results", "gnndelete_validation")


def cheng_auroc_ed(score_fn, forget_e, remain_e, rng_cmp):
    k = len(forget_e)
    idx = rng_cmp.choice(len(remain_e), k, replace=False)
    rem_sample = [remain_e[i] for i in idx]
    pairs = rem_sample + forget_e
    y = np.array([1] * k + [0] * k)
    return auc_score(y, score_fn(pairs))


def run(dataset, seed):
    stem = f"{dataset}_seed{seed}"
    cache = np.load(os.path.join(FGS_ROOT, "results", "cache",
                                 f"{stem}_base.npz"))
    base = {"W1": cache["W1"], "W2": cache["W2"]}
    train_neg = [tuple(x) for x in cache["train_neg"].tolist()]
    forget_e = [tuple(x) for x in cache["forget_e"].tolist()]
    remain_e = [tuple(x) for x in cache["remain_e"].tolist()]

    Xsp, labels, edges = load_planetoid(dataset)
    X = Xsp.toarray().astype(np.float32)
    n = Xsp.shape[0]
    edge_set = set(edges)
    labeled = np.where(labels >= 0)[0]

    # --- harness retrain (exact harness call) ---
    rt, _t = unlearn_retrain(M, remain_e, n, X, train_neg, epochs=EPOCHS,
                             hid=HID, lr=LR,
                             rng=np.random.default_rng(seed + 999))

    # --- harness GNNDelete port (harness call, fresh NI rng) ---
    rng = np.random.default_rng(seed)
    k_ni = min(GD_NI, len(remain_e))
    ni_pos = [remain_e[i] for i in rng.choice(len(remain_e), k_ni,
                                              replace=False)]
    ni_neg = sample_non_edges(rng, n, edge_set, k_ni, labeled)
    phi, aff, _t2 = unlearn_gnndelete(M, "gcn", base, forget_e, remain_e,
                                      n, X, ni_pos, ni_neg,
                                      epochs=GD_EPOCHS, lr=GD_LR, seed=seed)

    rng_cmp = np.random.default_rng(20261104)
    ed_rt = cheng_auroc_ed(
        lambda pairs: M.scores_for(rt, remain_e, n, X, pairs),
        forget_e, remain_e, rng_cmp)
    rng_cmp = np.random.default_rng(20261104)  # same sample for the port
    ed_port = cheng_auroc_ed(
        lambda pairs: scores_with_operator(M, "gcn", base, phi, aff,
                                           remain_e, n, X, pairs),
        forget_e, remain_e, rng_cmp)

    ref = json.load(open(os.path.join(FGS_ROOT, "results", "per_seed",
                                      f"{stem}.json")))
    out = {
        "dataset": dataset, "seed": seed, "n_forget": len(forget_e),
        "cheng_auroc_ed_retrain": round(ed_rt, 4),
        "cheng_auroc_ed_port_lam1.0": round(ed_port, 4),
        "cheng_auroc_ed_delta_port_minus_retrain": round(ed_port - ed_rt, 4),
        "fgs_attack_auc_retrain_recorded": round(
            ref["methods"]["retrain"]["attack"], 4),
        "fgs_attack_auc_port_recorded": round(
            ref["methods"]["gnndelete"]["attack"], 4),
        "fgs_util_auc_retrain_recorded": round(
            ref["methods"]["retrain"]["util"], 4),
        "fgs_util_auc_port_recorded": round(
            ref["methods"]["gnndelete"]["util"], 4),
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f"{stem}.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out), flush=True)
    return out


if __name__ == "__main__":
    ds = sys.argv[1]
    for s in [int(a) for a in sys.argv[2:]]:
        run(ds, s)
