# Author: Prabhu Rangaraju
"""GNNDelete-style deletion-operator baseline for edge unlearning (NumPy).

Faithful adaptation of Cheng et al., "GNNDelete: A General Strategy for
Unlearning in Graph Neural Networks" (ICLR 2023), to edge unlearning for
link prediction.

Original method (node classification): freeze the trained GNN; insert a
trainable linear "deletion operator" phi^l at each layer, applied only to
nodes in the k-hop enclosing subgraph of deleted nodes; train phi with
  (a) DEC (deleted-element consistency): deleted nodes' predictions should
      be uninformative (KL toward uniform class distribution);
  (b) NI (neighborhood influence): predictions outside the affected region
      should match the original model's predictions.

Adaptation to edge unlearning for link prediction (documented deviations):
  D1. Affected region = endpoints of the forget edges (1-hop), the edge
      analogue of the enclosing subgraph around deleted nodes.
  D2. DEC target: for a deleted edge, the "uninformative" link-prediction
      output is the non-edge class -> BCE with y=0 on forget-edge scores
      (binary analogue of KL-to-uniform).
  D3. NI target: retained pair scores under the operator must match the
      BASE model's scores -> mean-squared error (the regression analogue
      of matching class distributions).
  D4. NumPy reimplementation (official code is PyTorch/PyG); same GCN /
      GraphSAGE backbone and training protocol as FGS for a controlled
      comparison. (A torch install was attempted in this VM; the NumPy port
      is used so all methods share one pipeline - see REPORT.md.)
  D5. Operators are per-layer linear maps (hid x hid), identity-initialized,
      trained with Adam on  DEC + lambda * NI  (lambda=1, 40 epochs, lr=0.01).

Only phi parameters are trained; the base GNN weights are frozen.
"""
import time

import numpy as np


def _pairs(edgelist):
    u = np.array([e[0] for e in edgelist], dtype=np.int64)
    v = np.array([e[1] for e in edgelist], dtype=np.int64)
    return u, v


def _sigmoid(s):
    return 1.0 / (1.0 + np.exp(-s))


class _Adam:
    def __init__(self, params, lr=0.01):
        self.lr = lr
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k in params:
            g = grads[k]
            self.m[k] = 0.9 * self.m[k] + 0.1 * g
            self.v[k] = 0.999 * self.v[k] + 0.001 * g * g
            mh = self.m[k] / (1.0 - 0.9 ** self.t)
            vh = self.v[k] / (1.0 - 0.999 ** self.t)
            params[k] -= self.lr * mh / (np.sqrt(vh) + 1e-8)


def _forward_gcn(base, phi, aff, An, X):
    AX = An @ X
    H1 = np.maximum(AX @ base["W1"], 0.0)
    H1a = H1.copy()
    H1a[aff] = H1[aff] @ phi["phi1"]
    H2 = An @ H1a
    Zpre = H2 @ base["W2"]
    Z = Zpre.copy()
    Z[aff] = Zpre[aff] @ phi["phi2"]
    return Z, (AX, H1, H1a, H2, Zpre)


def _forward_sage(base, phi, aff, An, X):
    AX = An @ X
    H1 = np.maximum(AX @ base["W1n"] + X @ base["W1s"], 0.0)
    H1a = H1.copy()
    H1a[aff] = H1[aff] @ phi["phi1"]
    AH1 = An @ H1a
    Zpre = AH1 @ base["W2n"] + H1a @ base["W2s"]
    Z = Zpre.copy()
    Z[aff] = Zpre[aff] @ phi["phi2"]
    return Z, (AX, H1, H1a, AH1, Zpre)


def _forward(M, backbone, base, phi, aff, An, X):
    if backbone == "gcn":
        return _forward_gcn(base, phi, aff, An, X)
    return _forward_sage(base, phi, aff, An, X)


def _pair_scores(Z, pu, pv):
    return (Z[pu] * Z[pv]).sum(1)


def _bce_y0(s):
    # BCE with y=0, mean over pairs: -mean(log(1 - sigmoid(s)))
    return float(np.logaddexp(0, s).mean())


def unlearn_gnndelete(M, backbone, base_params, forget_edges, remain_edges,
                      n, X, ni_pos, ni_neg, epochs=40, lr=0.01, lam=1.0,
                      seed=0):
    """Train deletion operators. Returns (phi, affected_mask, seconds).

    ni_pos / ni_neg: retained pairs for the NI loss (base-model scores are
    the targets). forget_edges: DEC pairs (BCE toward non-edge).
    """
    rng = np.random.default_rng(seed)
    t0 = time.time()
    aff = np.zeros(n, dtype=bool)
    for (u, v) in forget_edges:
        aff[u] = True
        aff[v] = True
    An = M.norm_adj(remain_edges, n)  # post-deletion message passing
    hid = base_params["W2"].shape[0] if backbone == "gcn" \
        else base_params["W2n"].shape[0]
    phi = {"phi1": np.eye(hid, dtype=np.float32),
           "phi2": np.eye(hid, dtype=np.float32)}
    fu, fv = _pairs(forget_edges)
    pu_, pv_ = _pairs(ni_pos + ni_neg)
    # base-model scores on NI pairs = NI targets (frozen reference)
    Z0, _ = _forward(M, backbone, base_params,
                     {"phi1": np.eye(hid, dtype=np.float32),
                      "phi2": np.eye(hid, dtype=np.float32)},
                     np.zeros(n, dtype=bool), An, X)
    s_tgt = _pair_scores(Z0, pu_, pv_)
    opt = _Adam(phi, lr)
    for _ in range(epochs):
        Z, cache = _forward(M, backbone, base_params, phi, aff, An, X)
        s_f = _pair_scores(Z, fu, fv)
        s_n = _pair_scores(Z, pu_, pv_)
        # DEC: BCE(y=0) on forget edges; NI: MSE to base scores
        p = _sigmoid(s_f).astype(np.float32)
        ds_f = (p / max(len(fu), 1)).astype(np.float32)  # d BCE_y0 / ds
        ds_n = (2.0 * (s_n - s_tgt) / max(len(pu_), 1)).astype(np.float32)
        grads = _backward_phi_joint(M, backbone, base_params, phi, aff, An, X,
                                    cache, [(fu, fv, ds_f, 0),
                                            (pu_, pv_, ds_n, 1)])
        # scale NI grads by lam
        g = {k: grads[k][0] + lam * grads[k][1] for k in grads}
        opt.step(phi, g)
    return phi, aff, time.time() - t0


def _backward_phi_joint(M, backbone, base_params, phi, aff, An, X, cache,
                        terms):
    """terms: list of (pu, pv, ds, idx) with idx 0=DEC, 1=NI.

    Returns {k: (grad_dec_k, grad_ni_k)}.
    """
    hid = phi["phi1"].shape[0]
    n = X.shape[0]
    # recover Z (post-operator embeddings) from cache
    if backbone == "gcn":
        AX, H1, H1a, H2, Zpre = cache
        Z = Zpre.copy()
        Z[aff] = Zpre[aff] @ phi["phi2"]
    else:
        AX, H1, H1a, AH1, Zpre = cache
        Z = Zpre.copy()
        Z[aff] = Zpre[aff] @ phi["phi2"]
    gdec = {k: np.zeros_like(v) for k, v in phi.items()}
    gni = {k: np.zeros_like(v) for k, v in phi.items()}
    for (pu, pv, ds, idx) in terms:
        G = gdec if idx == 0 else gni
        dZ = np.zeros_like(Z)
        np.add.at(dZ, pu, ds[:, None] * Z[pv])
        np.add.at(dZ, pv, ds[:, None] * Z[pu])
        # layer 2 operator: Z[aff] = Zpre[aff] @ phi2
        G["phi2"] += Zpre[aff].T @ dZ[aff]
        dZpre = dZ.copy()
        dZpre[aff] = dZ[aff] @ phi["phi2"].T
        if backbone == "gcn":
            dH2 = dZpre @ base_params["W2"].T
            dH1a = An @ dH2  # An symmetric for gcn
            G["phi1"] += H1[aff].T @ dH1a[aff]
        else:
            dAH1 = dZpre @ base_params["W2n"].T
            dH1a = dZpre @ base_params["W2s"].T + An.T @ dAH1
            G["phi1"] += H1[aff].T @ dH1a[aff]
    return {k: (gdec[k], gni[k]) for k in phi}


def scores_with_operator(M, backbone, base_params, phi, aff, adj_edges,
                         n, X, pairs):
    """Link scores under the operator-modified model (post-deletion adj)."""
    An = M.norm_adj(adj_edges, n)
    Z, _ = _forward(M, backbone, base_params, phi, aff, An, X)
    pu, pv = _pairs(pairs)
    return _pair_scores(Z, pu, pv)
