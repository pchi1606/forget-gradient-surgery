# Author: Prabhu Rangaraju
"""2-layer GCN for link prediction, pure NumPy + SciPy (sparse adjacency).

Model:  Z = A_norm @ relu(A_norm @ X @ W1) @ W2
Link score for pair (u,v):  s(u,v) = z_u . z_v   (logit; prob = sigmoid(s))

The backward pass is written in a general form: given per-pair upstream
derivatives ds = dL/ds (one scalar per pair), it returns dL/dW1, dL/dW2.
This lets the same code serve BCE training (ds = p - y) and the
forget-gradient surgery (ds = constant weights on score means).
"""
import time

import numpy as np
import scipy.sparse as sp


def norm_adj(edges, n):
    """Symmetric normalized adjacency with self-loops, CSR float32."""
    if len(edges):
        u = np.array([e[0] for e in edges], dtype=np.int64)
        v = np.array([e[1] for e in edges], dtype=np.int64)
        uu = np.concatenate([u, v])
        vv = np.concatenate([v, u])
    else:
        uu = vv = np.array([], dtype=np.int64)
    A = sp.coo_matrix((np.ones(len(uu), dtype=np.float32), (uu, vv)), shape=(n, n))
    A = A.tocsr()
    A.setdiag(1.0)
    d = np.asarray(A.sum(1)).ravel()
    dinv = 1.0 / np.sqrt(d)
    return (A.multiply(dinv[:, None]).multiply(dinv[None, :])).tocsr()


def init_params(in_dim, hid, rng):
    s1 = np.sqrt(2.0 / in_dim)
    s2 = np.sqrt(2.0 / hid)
    return {
        "W1": (rng.standard_normal((in_dim, hid)) * s1).astype(np.float32),
        "W2": (rng.standard_normal((hid, hid)) * s2).astype(np.float32),
    }


def forward(params, An, X):
    AX = An @ X
    H1pre = AX @ params["W1"]
    H1 = np.maximum(H1pre, 0.0)
    H2 = An @ H1
    Z = H2 @ params["W2"]
    cache = (AX, H1pre, H1, H2, Z, An)
    return Z, cache


def pair_scores(Z, pu, pv):
    return (Z[pu] * Z[pv]).sum(1)


def backward_ds(params, cache, pu, pv, ds):
    """Gradients of L given upstream dL/ds per pair (ds shape (m,))."""
    AX, H1pre, H1, H2, Z, An = cache
    dZ = np.zeros_like(Z)
    np.add.at(dZ, pu, ds[:, None] * Z[pv])
    np.add.at(dZ, pv, ds[:, None] * Z[pu])
    dW2 = H2.T @ dZ
    dH2 = dZ @ params["W2"].T
    dH1 = An @ dH2  # An symmetric
    dH1pre = dH1 * (H1pre > 0.0)
    dW1 = AX.T @ dH1pre
    return {"W1": dW1, "W2": dW2}


class Adam:
    def __init__(self, params, lr=0.01):
        self.lr = lr
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads, ascend=False):
        self.t += 1
        for k in params:
            g = -grads[k] if ascend else grads[k]
            self.m[k] = 0.9 * self.m[k] + 0.1 * g
            self.v[k] = 0.999 * self.v[k] + 0.001 * g * g
            mh = self.m[k] / (1.0 - 0.9 ** self.t)
            vh = self.v[k] / (1.0 - 0.999 ** self.t)
            params[k] -= self.lr * mh / (np.sqrt(vh) + 1e-8)


def bce_loss(s, y):
    return float((-(y * np.logaddexp(0, -s) + (1 - y) * np.logaddexp(0, s))).mean())


def train(edges, n, X, neg_edges, epochs, params=None, lr=0.01, hid=32, rng=None,
          verbose=False):
    """BCE training on pos edges + provided negative edges. Returns (params, secs)."""
    if rng is None:
        rng = np.random.default_rng(0)
    params = init_params(X.shape[1], hid, rng) if params is None else \
        {k: v.copy() for k, v in params.items()}
    An = norm_adj(edges, n)
    pairs = edges + neg_edges
    pu = np.array([p[0] for p in pairs], dtype=np.int64)
    pv = np.array([p[1] for p in pairs], dtype=np.int64)
    y = np.array([1.0] * len(edges) + [0.0] * len(neg_edges), dtype=np.float32)
    opt = Adam(params, lr)
    t0 = time.time()
    for ep in range(epochs):
        Z, cache = forward(params, An, X)
        s = pair_scores(Z, pu, pv)
        p = 1.0 / (1.0 + np.exp(-s))
        grads = backward_ds(params, cache, pu, pv, (p - y) / len(y))
        opt.step(params, grads)
        if verbose and ep % 50 == 0:
            print(f"    epoch {ep}: bce {bce_loss(s, y):.4f}")
    return params, time.time() - t0


def scores_for(params, edges_for_adj, n, X, pairs):
    Z, _ = forward(params, norm_adj(edges_for_adj, n), X)
    pu = np.array([p[0] for p in pairs], dtype=np.int64)
    pv = np.array([p[1] for p in pairs], dtype=np.int64)
    return pair_scores(Z, pu, pv)
