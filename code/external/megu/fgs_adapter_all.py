"""FGS-protocol adapter for MEGU — Phase 2 generalization (all datasets/seeds).

Same adapter as fgs_adapter.py (Phase 1, Cora-only), parameterized by
dataset and seed. Everything else is identical to Phase 1:

  * official MEGU code (xkLi-Allen/MEGU @ 03bbb35) run unmodified
    (their NodeClassifier GCN base, neighbor_select, megu_training loop,
    published hyperparameters from parameter_parser.py defaults),
  * FGS harness's persisted split/forget/attack/test sets from
    results/cache/<stem>_base.npz, where <stem> is "<ds>_seed<seed>"
    (seeds 42/7/123) or "gcn_<ds>_seed<seed>" (seeds 2024/2025/31337/
    31338/5555/7777/999) — resolved by file existence,
  * scoring: penultimate GCN layer embeddings over the REMAIN graph,
    dot-product decoder, harness auc_score construction,
  * retrain reference read from results/per_seed/<stem>.json,
  * disclosed deviations identical to Phase 1 (see
    battery/megu-feasibility-2026-11-04.md): base-model objective
    mismatch (node classifier, never edge-trained), FGS forget set
    substituted for MEGU's random deletion request, CorrectAndSmooth
    dropped for link scoring, full-batch loader shim for the base
    classifier only, MEGU's own fixed node split (random_state 100),
    procedure time = neighbor select + unlearning loop vs harness
    recorded retrain seconds.

Usage: python fgs_adapter_all.py <dataset> <seed> [<seed> ...]
       python fgs_adapter_all.py all          # all 3 datasets x 10 seeds
Writes results/megu/fgs_megu_<ds>_seed<seed>.json per run (new files).
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MEGU_ROOT = HERE
FGS_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT_DIR = os.path.join(FGS_ROOT, "results", "megu")
sys.path.insert(0, MEGU_ROOT)
sys.path.insert(0, FGS_ROOT)

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from scipy.stats import rankdata

from data import load_planetoid  # FGS loader (same Planetoid raw files)
from exp.exp_megu import ExpMEGU, sparse_mx_to_torch_sparse_tensor, normalize_adj
from torch_geometric.utils import to_scipy_sparse_matrix

SEEDS = [42, 7, 123, 2024, 2025, 31337, 31338, 5555, 7777, 999]
DATASETS = ["cora", "citeseer", "pubmed"]


def auc_score(y, s):
    """Identical to experiment.py's auc_score."""
    r = rankdata(s)
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def directed(edge_list):
    e = np.array(edge_list, dtype=np.int64)
    both = np.concatenate([e, e[:, ::-1]], axis=0)
    return torch.from_numpy(both.T.copy())


def resolve_stem(dataset, seed):
    for stem in (f"{dataset}_seed{seed}", f"gcn_{dataset}_seed{seed}"):
        if os.path.exists(os.path.join(
                FGS_ROOT, "results", "cache", f"{stem}_base.npz")):
            return stem
    raise FileNotFoundError(f"no cache for {dataset} seed {seed}")


def run_seed(dataset, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    stem = resolve_stem(dataset, seed)
    cache = np.load(os.path.join(
        FGS_ROOT, "results", "cache", f"{stem}_base.npz"))
    train_e = [tuple(x) for x in cache["train_e"].tolist()]
    test_e = [tuple(x) for x in cache["test_e"].tolist()]
    test_neg = [tuple(x) for x in cache["test_neg"].tolist()]
    forget_e = [tuple(x) for x in cache["forget_e"].tolist()]
    remain_e = [tuple(x) for x in cache["remain_e"].tolist()]
    attack_neg = [tuple(x) for x in cache["attack_neg"].tolist()]

    Xsp, labels, _edges = load_planetoid(dataset)
    X = Xsp.toarray().astype(np.float32)
    n = X.shape[0]

    data = Data(x=torch.from_numpy(X), y=torch.from_numpy(labels.copy()),
                edge_index=directed(train_e))
    data.num_nodes = n
    data.num_classes = int(labels.max()) + 1

    args = dict(
        dataset_name=dataset, target_model="GCN", unlearn_task="edge",
        unlearn_ratio=0.05, is_split=True, test_ratio=0.2, num_epochs=100,
        num_runs=1, batch_size=2048, test_batch_size=2048,
        unlearn_lr=0.05, kappa=0.01, alpha1=0.8, alpha2=0.5,
        inductive="normal", method="MEGU",
    )

    exp = ExpMEGU.__new__(ExpMEGU)
    exp.args = args
    exp.data = data
    # train_test_split() persists the split via data_store; stub the save
    # (no computational effect) since the adapter supplies data directly.
    class _StubStore:
        @staticmethod
        def save_train_test_split(tr, te):
            pass
    exp.data_store = _StubStore()
    exp.device = torch.device("cpu")
    import logging
    exp.logger = logging.getLogger("ExpMEGU-adapter")
    exp.num_feats = data.num_features
    exp.train_test_split()          # MEGU's own node split (random_state=100)
    # Adapter data fix (Phase 2, disclosed): the Planetoid raw files
    # leave 15 CiteSeer nodes unlabeled (label -1); they cannot be
    # classification targets for MEGU's CE loss. Restrict MEGU's node
    # train/test masks to labeled nodes. No-op on Cora/PubMed (no -1
    # labels, Phase-1 Cora numbers unchanged). Unlabeled nodes still
    # take part in message passing and in the label-free link scoring.
    labeled = data.y >= 0
    if not bool(labeled.all()):
        exp.data.train_mask &= labeled
        exp.data.test_mask &= labeled
    exp.determine_target_model()    # their NodeClassifier GCN

    # FGS deletion request substituted for MEGU's random one
    exp.data.x_unlearn = exp.data.x.clone()
    exp.data.edge_index_unlearn = directed(remain_e)
    exp.temp_node = np.unique(np.array(forget_e).ravel())
    exp.num_layers = 2
    exp.adj = sparse_mx_to_torch_sparse_tensor(
        normalize_adj(to_scipy_sparse_matrix(exp.data.edge_index,
                                             num_nodes=n)))

    # Environment shim (disclosed, identical to Phase 1): PyG 2.8's
    # NeighborLoader needs pyg-lib / torch-sparse, for which no wheel
    # exists for torch 2.14+cpu in this venv. Replace ONLY the base
    # classifier's mini-batch loader with a full-batch equivalent.
    # MEGU's unlearning loop itself is untouched official code.
    def _fullbatch_loader():
        d = exp.data
        idx = torch.arange(d.num_nodes)
        order = torch.cat([idx[d.train_mask], idx[~d.train_mask]])
        pos = torch.empty_like(order)
        pos[order] = idx
        b = Data(x=d.x[order], y=d.y[order],
                 edge_index=pos[d.edge_index])
        b.batch_size = int(d.train_mask.sum())
        exp.target_model.train_loader = [b]
    exp.target_model._gen_train_loader = _fullbatch_loader

    t0 = time.time()
    exp.neighbor_khop = exp.neighbor_select(exp.data.x)
    t_neighbor = time.time() - t0

    train_time, _ = exp._train_model(0)
    f1_before = exp.evaluate(0)

    t0 = time.time()
    unlearn_loop_time, f1_after = exp.megu_training()
    t_total_unlearn = time.time() - t0  # includes C&S + F1 eval overhead
    # procedure time charged to MEGU = neighbor selection + unlearning loop
    t_procedure = t_neighbor + unlearn_loop_time

    # --- FGS scoring of the unlearned model ---
    model = exp.target_model.model
    model.eval()
    ei_remain = directed(remain_e)
    with torch.no_grad():
        emb = F.relu(model.convs[0](data.x, ei_remain)).numpy()

    def link_scores(pairs):
        P = np.array(pairs)
        return np.einsum("ij,ij->i", emb[P[:, 0]], emb[P[:, 1]])

    y_attack = np.array([1] * len(forget_e) + [0] * len(attack_neg))
    attack = auc_score(y_attack, link_scores(forget_e + attack_neg))
    y_test = np.array([1] * len(test_e) + [0] * len(test_neg))
    util = auc_score(y_test, link_scores(test_e + test_neg))

    ref = json.load(open(os.path.join(
        FGS_ROOT, "results", "per_seed", f"{stem}.json")))
    rt = ref["methods"]["retrain"]

    out = {
        "method": "MEGU (official xkLi-Allen/MEGU @ 03bbb35, FGS adapter)",
        "dataset": dataset, "seed": seed, "n_forget": len(forget_e),
        "attack_auc": round(attack, 4), "util_auc": round(util, 4),
        "retrain_attack_auc": round(rt["attack"], 4),
        "retrain_util_auc": round(rt["util"], 4),
        "attack_gap_vs_retrain": round(attack - rt["attack"], 4),
        "util_gap_vs_retrain": round(util - rt["util"], 4),
        "unlearn_loop_s": round(unlearn_loop_time, 1),
        "neighbor_select_s": round(t_neighbor, 1),
        "procedure_s": round(t_procedure, 1),
        "retrain_s": rt["s"],
        "time_ratio_procedure_vs_retrain": round(t_procedure / rt["s"], 3),
        "base_train_s_megu_classifier": round(train_time, 1),
        "megu_f1_before": round(float(f1_before), 4),
        "megu_f1_after": round(float(f1_after), 4),
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"fgs_megu_{dataset}_seed{seed}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out), flush=True)
    return out


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv and argv[0] == "all":
        jobs = [(ds, s) for ds in DATASETS for s in SEEDS]
    else:
        jobs = [(argv[0], int(a)) for a in argv[1:]]
    for ds, s in jobs:
        try:
            run_seed(ds, s)
        except Exception as e:  # keep sweeping; record the failure
            print(f"FAILED {ds} seed {s}: {type(e).__name__}: {e}",
                  flush=True)
