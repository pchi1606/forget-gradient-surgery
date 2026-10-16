# Author: Prabhu Rangaraju
"""Planetoid (Cora / CiteSeer / PubMed) loader for link-prediction experiments.

Data provenance: the classic Planetoid pickle files, obtained from the DGL
mirror (https://data.dgl.ai/dataset/{cora,citeseer,pubmed}.zip) because the
original kimiyoung/planetoid GitHub repository now returns 404.

Format notes (standard Planetoid quirks, handled here):
- `allx`/`ally`: features/labels for the first A nodes (train + non-test rest).
- `tx`/`ty`: features/labels for the 1000 test nodes, in the FILE ORDER of
  `ind.<ds>.test.index` (this matches the classic tkipf loader semantics:
  features[test_idx_file_order[k]] = tx[k]).
- CiteSeer quirk: the graph has 3327 nodes but only 3312 have features;
  15 nodes (ids not in test.index, in 2312..3326) get zero feature rows and
  label -1. They are excluded from label-matched negative sampling.
- The `graph` dict stores directed edges (both directions for these
  undirected citation graphs); we symmetrize, dedupe and drop self-loops.
"""
import os
import pickle

import numpy as np
import scipy.sparse as sp

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def load_planetoid(name):
    d = os.path.join(DATA_DIR, name)
    objs = {}
    for n in ["x", "y", "tx", "ty", "allx", "ally", "graph"]:
        with open(os.path.join(d, "ind.{}.{}".format(name, n)), "rb") as f:
            objs[n] = pickle.load(f, encoding="latin1")
    with open(os.path.join(d, "ind.{}.test.index".format(name))) as f:
        test_idx = np.array([int(line.strip()) for line in f])

    allx, tx = objs["allx"], objs["tx"]
    ally, ty = objs["ally"], objs["ty"]
    n_allx = allx.shape[0]
    n_nodes = len(objs["graph"])
    n_feat = allx.shape[1]

    X = sp.lil_matrix((n_nodes, n_feat))
    X[:n_allx] = allx
    X[test_idx] = tx  # tx rows are in test.index file order
    X = X.tocsr()
    # NOTE: we intentionally do NOT row-normalize features: in our tests
    # raw binary bag-of-words features gave better link-prediction AUC
    # (0.742 vs 0.731 on Cora) and better unlearning fidelity.

    labels = np.full(n_nodes, -1, dtype=np.int64)
    labels[:n_allx] = np.asarray(ally.argmax(1)).ravel()
    labels[test_idx] = np.asarray(ty.argmax(1)).ravel()

    # undirected unique edges, no self-loops
    seen = set()
    edges = []
    for u, vs in objs["graph"].items():
        for v in vs:
            if u == v:
                continue
            e = (u, v) if u < v else (v, u)
            if e not in seen:
                seen.add(e)
                edges.append(e)
    return X, labels, edges


if __name__ == "__main__":
    for ds in ["cora", "citeseer", "pubmed"]:
        X, labels, edges = load_planetoid(ds)
        n_labeled = int((labels >= 0).sum())
        print(f"{ds}: nodes={X.shape[0]} feats={X.shape[1]} edges={len(edges)} "
              f"labeled={n_labeled} classes={labels.max() + 1}")
