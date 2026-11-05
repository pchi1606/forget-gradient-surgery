"""2-layer GraphSAGE (mean aggregator) for link prediction, pure NumPy + SciPy.

Same module API as gcn.py so experiments can swap backbones:
    norm_adj, init_params, forward, pair_scores, backward_ds, Adam,
    bce_loss, train, scores_for

Model:  H1 = relu(A_rw @ X @ W1n + X @ W1s)
        Z  = A_rw @ H1 @ W2n + H1 @ W2s
where A_rw = D^{-1}(A + I) is the row-normalized (random-walk) adjacency.
This is the mean-aggregation variant of GraphSAGE (Hamilton et al., 2017):
each layer concatenates-then-transforms the node's own representation and
the mean of its neighbors' (here as a sum of two linear maps, which is
exactly CONCAT followed by a block-structured weight matrix).

Link score for pair (u,v): s(u,v) = z_u . z_v (logit; prob = sigmoid(s))

The backward pass takes per-pair upstream derivatives ds = dL/ds and
returns dL/dW for W in {W1n, W1s, W2n, W2s}, so the same code serves BCE
training and the forget-gradient surgery.
"""
import time

import numpy as np
import scipy.sparse as sp


def norm_adj(edges, n):
    """Row-normalized adjacency with self-loops (random walk), CSR float32."""
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
    dinv = 1.0 / d
    return (A.multiply(dinv[:, None])).tocsr()


def init_params(in_dim, hid, rng):
    s = np.sqrt(2.0 / in_dim)
    s2 = np.sqrt(2.0 / hid)
    return {
        "W1n": (rng.standard_normal((in_dim, hid)) * s).astype(np.float32),
        "W1s": (rng.standard_normal((in_dim, hid)) * s).astype(np.float32),
        "W2n": (rng.standard_normal((hid, hid)) * s2).astype(np.float32),
        "W2s": (rng.standard_normal((hid, hid)) * s2).astype(np.float32),
    }


def forward(params, An, X):
    AX = An @ X
    H1pre = AX @ params["W1n"] + X @ params["W1s"]
    H1 = np.maximum(H1pre, 0.0)
    AH1 = An @ H1
    Z = AH1 @ params["W2n"] + H1 @ params["W2s"]
    cache = (AX, H1pre, H1, AH1, Z, An, X)
    return Z, cache


def pair_scores(Z, pu, pv):
    return (Z[pu] * Z[pv]).sum(1)


def backward_ds(params, cache, pu, pv, ds):
    """Gradients of L given upstream dL/ds per pair (ds shape (m,))."""
    AX, H1pre, H1, AH1, Z, An, X = cache
    dZ = np.zeros_like(Z)
    np.add.at(dZ, pu, ds[:, None] * Z[pv])
    np.add.at(dZ, pv, ds[:, None] * Z[pu])
    dW2n = AH1.T @ dZ
    dW2s = H1.T @ dZ
    dAH1 = dZ @ params["W2n"].T
    dH1 = dZ @ params["W2s"].T + An.T @ dAH1  # An not symmetric: use An.T
    dH1pre = dH1 * (H1pre > 0.0)
    dW1n = AX.T @ dH1pre
    dW1s = X.T @ dH1pre
    return {"W1n": dW1n, "W1s": dW1s, "W2n": dW2n, "W2s": dW2s}


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
