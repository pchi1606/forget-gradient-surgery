# Author: Prabhu Rangaraju
"""Unlearning methods for GNN link prediction.

Every function takes a model module `M` (gcn or sage) exposing:
    forward, backward_ds, norm_adj, pair_scores, train, Adam, scores_for

Baselines
---------
retrain      : fresh model trained with BCE on the remaining edges (gold std).
finetune     : base model continued with BCE on the remaining edges.
grad_ascent  : base model, gradient ASCENT on BCE-with-y=0 over the forget
               edges (NegGrad-style naive baseline). Uses the corrected
               (post-deletion) adjacency like every other method.
gnndelete    : GNNDelete-style deletion-operator baseline (see gnndelete.py).

Novel method: FGS (Forget Gradient Surgery)
-------------------------------------------
Goal: suppress the model's scores on forgotten edges while leaving its
scores on retained data (the pos-vs-neg margin) unchanged to first order --
without retraining, new parameters, Hessians, or sharding.

Let s_uv(theta) be the link logit for pair (u,v), E_f the forget edges,
E_r a sample of retained edges and N_r a sample of retained non-edges,
all scored under the POST-DELETION adjacency A_r.

For t = 1..K:
  g_f = grad_theta mean_{(u,v) in E_f} s_uv        (direction that RAISES
                                                    forget scores)
  g_r = grad_theta [ mean_{E_r} s - mean_{N_r} s ]  (direction that raises
                                                    the retained margin;
                                                    E_r = ALL remaining
                                                    train edges, N_r =
                                                    equal-sized non-edge
                                                    sample: deterministic,
                                                    stable anchor)
  g   = g_f - ((g_f . g_r) / ||g_r||^2) g_r         (forget direction with
                                                    the margin-harming
                                                    component removed)
  theta <- theta - eta * g / ||g||                  (normalized descent:
                                                    LOWER forget scores)

Stopping (adaptive, no per-dataset K tuning): run up to `max_steps`
projected steps and keep the iterate minimizing a stopping criterion
computed on VALIDATION edges (never trained on, absent from the adjacency
-- exactly the status forget edges have under retraining).

stop_rule="ks" (original):
    minimize the two-sample KS statistic between forget-edge scores and
    validation-edge scores.
stop_rule="ks_margin" (mean-gated KS; tried as a fix for premature KS
    alignment):
    eligible iterates are those with mean(forget scores) <= mean(val scores)
    (central-tendency indistinguishability); among eligible iterates keep
    the one minimizing KS. If no iterate is eligible, keep the iterate with
    the smallest mean forget score (maximum forgetting). This is NOT
    dataset-specific: it is the same rule on every dataset/seed, and it can
    only delay stopping relative to the first eligible iterate, never
    advance it past an ineligible KS minimum.
stop_rule="attack" (attack-calibrated; the recommended rule):
    minimize |AUC(forget vs stop_neg) - AUC(val vs stop_neg)|, where
    stop_neg is a held-out hard-negative sample disjoint from the evaluation
    attack negatives. Rationale: the evaluation attack measures the
    forget-vs-nonedge RANKING, while KS measures marginal-distribution
    match; on graphs where the KS minimum and the attack-matching point
    disagree (e.g. CiteSeer seeds 7/123), calibrating the ranking of forget
    edges to the ranking of unseen (validation) edges recovers
    retraining-level attack AUC. Uses only the forget request, validation
    edges, and freshly sampled non-edges -- never the evaluation set.

Relation to prior work (honest novelty statement):
- The orthogonal-projection primitive comes from multi-task learning
  (PCGrad, Yu et al. 2020) and has been used for unlearning in diffusion
  models; to our knowledge it has never been formulated for GRAPH EDGE
  unlearning.
- vs GNNDelete: no added "deletion operator" parameters, no DEC/NI output
  losses; FGS edits existing weights directly in a few projected steps.
- vs GIF/CEU/IDEA (influence functions): no Hessian, no convexity or
  convergence assumptions, no one-step Newton approximation.
- vs MEGU: no dual prediction/unlearning modules, no mutual-evolution
  training loop.
- vs Projector/ETR: not a closed-form weight projection derived from
  linear-GNN theory nor Fisher-ratio dampening; it is iterative,
  applies to nonlinear GNNs, and is explicitly margin-preserving.
- vs shard methods (GraphEraser/GUIDE): no partitioning, single model.
- vs naive gradient ascent (NegGrad): ascent is the UNPROJECTED special
  case of FGS; we show empirically that the projection is what makes
  forgetting actually happen.
"""
import time

import numpy as np


def _pairs(edgelist):
    u = np.array([e[0] for e in edgelist], dtype=np.int64)
    v = np.array([e[1] for e in edgelist], dtype=np.int64)
    return u, v


def unlearn_retrain(M, remain_edges, n, X, neg_edges, epochs=150, hid=32,
                    lr=0.01, rng=None):
    return M.train(remain_edges, n, X, neg_edges, epochs, params=None,
                   lr=lr, hid=hid, rng=rng)


def unlearn_finetune(M, base_params, remain_edges, n, X, neg_edges,
                     epochs=40, lr=0.01):
    return M.train(remain_edges, n, X, neg_edges, epochs, params=base_params,
                   lr=lr)


def unlearn_grad_ascent(M, base_params, forget_edges, remain_edges, n, X,
                        steps=15, lr=0.005):
    params = {k: v.copy() for k, v in base_params.items()}
    An = M.norm_adj(remain_edges, n)
    fu, fv = _pairs(forget_edges)
    m = len(fu)
    y = np.zeros(m, dtype=np.float32)
    opt = M.Adam(params, lr)
    t0 = time.time()
    for _ in range(steps):
        Z, cache = M.forward(params, An, X)
        s = M.pair_scores(Z, fu, fv)
        p = 1.0 / (1.0 + np.exp(-s)).astype(np.float32)
        grads = M.backward_ds(params, cache, fu, fv, (p - y) / m)
        opt.step(params, grads, ascend=True)
    return params, time.time() - t0


def unlearn_fgs(M, base_params, forget_edges, remain_edges, n, X,
                retain_pos, retain_neg, steps=20, eta=0.05,
                val_edges=None, tol=0.02, max_steps=40, stop_rule="ks",
                stop_neg=None, use_projection=True, keep_iterates=False):
    """Forget Gradient Surgery.

    Returns (params, seconds, n_steps_done, diagnostics). diagnostics holds
    per-iterate trajectories and the chosen stop step.

    stop_rule selects the iterate:
      "ks"        : min KS(forget scores, val-edge scores).
      "ks_margin" : min KS among iterates with mean(forget) <= mean(val);
                    falls back to the max-forgetting iterate if none qualify.
      "attack"    : min |AUC(forget vs stop_neg) - AUC(val vs stop_neg)|.
                    `stop_neg` is a held-out hard-negative sample (disjoint
                    from the evaluation attack negatives); the rule calibrates
                    the forget-vs-nonedge ranking to the unseen-edge
                    (validation) ranking, which is the retraining behavior the
                    membership-style attack measures. Requires val_edges and
                    stop_neg.

    use_projection=False disables the orthogonal projection (g = g_f);
    this is the no-projection ablation isolating the projection's causal
    contribution (it is NOT the same as grad_ascent, which ascends BCE).

    keep_iterates=True stores a deep copy of the parameters at every
    iterate -- INCLUDING iterate 0 before any gradient step, plus the
    final iterate after the last update -- into diag["iterates"]. Used by
    the fixed-step stopping ablation (experiment_fixedstep.py). It does
    not change the trajectory, the stopping selection, or any other
    behavior.
    """
    from scipy.stats import ks_2samp, rankdata
    assert stop_rule in ("ks", "ks_margin", "attack")
    if stop_rule == "attack":
        assert val_edges is not None and stop_neg is not None

    def _auc(y, s):
        r = rankdata(s)
        n1 = int(y.sum())
        n0 = len(y) - n1
        return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

    params = {k: v.copy() for k, v in base_params.items()}
    An = M.norm_adj(remain_edges, n)  # post-deletion message passing
    fu, fv = _pairs(forget_edges)
    ru, rv = _pairs(retain_pos)
    nu, nv = _pairs(retain_neg)
    m, rp, rn = len(fu), len(ru), len(nu)
    ds_f = np.full(m, 1.0 / m, dtype=np.float32)
    ds_r = np.concatenate([np.full(rp, 1.0 / rp),
                           np.full(rn, -1.0 / rn)]).astype(np.float32)
    cru = np.concatenate([ru, nu])
    crv = np.concatenate([rv, nv])
    vu, vv = _pairs(val_edges) if val_edges else (None, None)
    if stop_rule == "attack":
        hu, hv = _pairs(stop_neg)
        y_sur = np.array([1] * m + [0] * len(hu))
        y_ref = np.array([1] * len(vu) + [0] * len(hu))
    t0 = time.time()
    diag = {"ks": [], "mean_f": [], "mean_v": [],
            "surr": [], "ref": [], "rule": stop_rule}
    if keep_iterates:
        diag["iterates"] = []
    best = (float("inf"), {k: v.copy() for k, v in params.items()}, 0)
    best_mean = (float("inf"), None, 0)  # fallback: max-forgetting iterate
    n_done = 0
    for _ in range(max_steps):
        if keep_iterates:
            # iterate n_done, recorded BEFORE the gradient update;
            # first entry is iterate 0 (base params, no step taken yet)
            diag["iterates"].append(
                {k: v.copy() for k, v in params.items()})
        Z, cache = M.forward(params, An, X)
        if val_edges is not None:
            sf = M.pair_scores(Z, fu, fv)
            sv = M.pair_scores(Z, vu, vv)
            mf, mv = float(sf.mean()), float(sv.mean())
            diag["mean_f"].append(mf)
            diag["mean_v"].append(mv)
            if stop_rule == "attack":
                sh = M.pair_scores(Z, hu, hv)
                surr = _auc(y_sur, np.concatenate([sf, sh]))
                ref = _auc(y_ref, np.concatenate([sv, sh]))
                diag["surr"].append(surr)
                diag["ref"].append(ref)
                key = abs(surr - ref)
            else:
                ks = float(ks_2samp(sf, sv).statistic)
                diag["ks"].append(ks)
                if stop_rule == "ks":
                    key = ks
                else:  # ks_margin: ineligible iterates get infinite key
                    key = ks if mf <= mv else float("inf")
            if key < best[0]:
                best = (key, {k: v.copy() for k, v in params.items()}, n_done)
            if mf < best_mean[0]:
                best_mean = (mf, {k: v.copy() for k, v in params.items()},
                             n_done)
        g_f = M.backward_ds(params, cache, fu, fv, ds_f)
        g_r = M.backward_ds(params, cache, cru, crv, ds_r)
        if use_projection:
            dot = sum(float((g_f[k] * g_r[k]).sum()) for k in g_f)
            nrm_r = sum(float((g_r[k] ** 2).sum()) for k in g_r)
            g = {k: g_f[k] - (dot / max(nrm_r, 1e-12)) * g_r[k] for k in g_f}
        else:  # no-projection ablation: raw forget-score descent
            g = g_f
        nrm = float(np.sqrt(sum(float((g[k] ** 2).sum()) for k in g)))
        if nrm < 1e-12:
            break
        for k in params:
            params[k] = (params[k] - eta * g[k] / nrm).astype(np.float32)
        n_done += 1
    if keep_iterates:
        # final iterate after the last gradient update (e.g. step 40
        # when max_steps=40); appended even if the loop broke early
        diag["iterates"].append({k: v.copy() for k, v in params.items()})
    if val_edges is not None:
        if stop_rule == "ks_margin" and not np.isfinite(best[0]):
            # no iterate satisfied mean_f <= mean_v: keep max forgetting
            params = best_mean[1]
            n_done = best_mean[2]
            diag["fallback"] = True
        else:
            params = best[1]
            n_done = best[2]
            diag["fallback"] = False
    diag["chosen_step"] = n_done
    return params, time.time() - t0, n_done, diag
