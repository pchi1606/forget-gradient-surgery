"""FGS-protocol adapter for ETR — run official ETR under the FGS harness.

Mirrors the MEGU adapter (external/megu/fgs_adapter_all.py):

  * official ETR code (AllminerLab/ETR @ c88ec74) run unmodified — their
    NodeClassifier GCN base (hidden 256, 100 epochs), their edge-task
    Erase/Rectify implementation (ExpETR.etr -> modify_weight_ef /
    modify_weight2_ef, i.e. the paper's Algorithm 2), published
    hyperparameters from the repo's config2.yaml via their own
    get_parameter (GCN: lr 0.001, wd 1e-4 [PubMed 1e-7],
    erase_ratio m = 0.01, lambda l = 0.3 for all three datasets),
  * FGS harness's persisted split/forget/attack/test sets from
    results/cache/<stem>_base.npz; the FGS TRAIN graph plays the role
    of ETR's full graph (official ETR's `data` is the full Planetoid
    graph; comparability requires the harness's training graph,
    exactly as in the MEGU adapter),
  * the FGS forget set (5% of train edges) substituted for ETR's random
    edge-deletion request; the remain graph is built directly from the
    harness's remain_e,
  * scoring: penultimate GCN layer embeddings (ReLU(conv1), 256-d) over
    the REMAIN graph, dot-product decoder, harness auc_score
    construction — identical to the MEGU adapter,
  * retrain reference read from results/per_seed/<stem>.json,
  * procedure time = FULL ExpETR.etr() wall-clock, including the
    full-graph / influenced-subgraph gradient+FIM computations that
    precede ETR's own internal timer (their reported 0.01-0.02 s
    excludes them; the FGS table charges the whole procedure),
  * disclosed deviations: base-model objective mismatch (node
    classifier, never edge-trained — same as MEGU), full-batch loader
    shim for ALL NeighborLoaders (base training + test evaluation;
    PyG 2.8 NeighborLoader needs pyg-lib/torch-sparse, no wheel for
    torch 2.14+cpu — ETR's unlearning computations are already
    full-batch in the official code and are untouched), ETR's own
    fixed node split (train_test_split random_state=100, test_ratio
    0.1 per parameter_parser default / paper's 90% train nodes),
    CiteSeer labeled-mask restriction (15 nodes with label -1 cannot
    be CE targets; same fix as MEGU Phase 2).

Usage: python fgs_adapter_all.py <dataset> <seed> [<seed> ...]
       python fgs_adapter_all.py all          # all 3 datasets x 10 seeds
Writes results/etr/fgs_etr_<ds>_seed<seed>.json per run.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ETR_ROOT = HERE
FGS_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT_DIR = os.path.join(FGS_ROOT, "results", "etr")
sys.path.insert(0, ETR_ROOT)
sys.path.insert(0, FGS_ROOT)

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from scipy.stats import rankdata
from sklearn.metrics import f1_score

# Environment shim (disclosed): lib_dataset/data_store.py imports
# ogb.nodeproppred at module level, but ogb is not installed in this
# venv and is only used inside DataStore.load_raw_data for the
# ogbn-* datasets. The adapter never instantiates DataStore (it
# supplies data directly, like the MEGU adapter), so a stub module
# satisfies the import without touching ETR's code or computation.
import types as _types
if "ogb" not in sys.modules:
    _ogb = _types.ModuleType("ogb")
    _np = _types.ModuleType("ogb.nodeproppred")
    _np.PygNodePropPredDataset = object
    _ogb.nodeproppred = _np
    sys.modules["ogb"] = _ogb
    sys.modules["ogb.nodeproppred"] = _np

from data import load_planetoid  # FGS loader (same Planetoid raw files)
from exp.exp_etr import ExpETR
from lib_utils.utils import (get_dataset_train, get_influence_nodes,
                             get_parameter, get_subgraph)

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


def fullbatch_loader(d, mask):
    """Single full-batch 'batch' with masked nodes first, so that
    output[:batch_size] / y[:batch_size] in ETR's train/eval loops see
    exactly the masked nodes — the full-batch equivalent of their
    NeighborLoader(input_nodes=mask) batches (disclosed shim)."""
    idx = torch.arange(d.num_nodes)
    order = torch.cat([idx[mask], idx[~mask]])
    pos = torch.empty_like(order)
    pos[order] = idx
    b = Data(x=d.x[order], y=d.y[order], edge_index=pos[d.edge_index])
    b.batch_size = int(mask.sum())
    return [b]


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
        dataset=dataset, target_model="GCN", unlearn_task="edge",
        unlearn_ratio=0.05, test_ratio=0.1, epochs=100, hops=2, dim=256,
        optimizer="Adam", label_smoothing=0.1, method="ETR", seed=seed,
        gpu=0, para_config=os.path.join(ETR_ROOT, "config2.yaml"),
    )
    args = get_parameter(args)  # official per-dataset lr/wd/erase_ratio/l

    exp = ExpETR.__new__(ExpETR)
    exp.args = args
    exp.device = torch.device("cpu")

    class _StubStore:
        @staticmethod
        def save_train_test_split(tr, te):
            pass
    exp.data_store = _StubStore()
    import logging
    exp.logger = logging.getLogger("ExpETR-adapter")
    exp.data = data
    exp.num_feats = data.num_features
    exp.train_test_split()          # ETR's own node split (random_state=100)
    # Adapter data fix (disclosed, same as MEGU Phase 2): 15 CiteSeer
    # nodes have label -1 and cannot be CE targets. Restrict ETR's node
    # train/test masks to labeled nodes. No-op on Cora/PubMed.
    labeled = data.y >= 0
    if not bool(labeled.all()):
        exp.data.train_mask &= labeled
        exp.data.test_mask &= labeled

    # --- FGS deletion request substituted for ETR's random one -------
    # Official edge branch of unlearning_request(), with the harness's
    # forget/remain sets in place of their random removal.
    exp.dataset_train = get_dataset_train(exp.data)
    exp.unique_nodes = np.unique(np.array(forget_e).ravel())
    exp.influenced_nodes = get_influence_nodes(
        args, exp.unique_nodes, exp.dataset_train.edge_index, 1)
    du = Data(x=exp.data.x, y=exp.data.y, edge_index=directed(remain_e),
              train_mask=exp.data.train_mask, test_mask=exp.data.test_mask,
              train_indices=exp.data.train_indices,
              test_indices=exp.data.test_indices)
    du.num_nodes = n
    exp.dataset_unlearned = du
    exp.dataset_train_unlearned = get_dataset_train(du)
    exp.influenced_subgraph = get_subgraph(exp.influenced_nodes,
                                           exp.dataset_train)
    exp.influenced_subgraph_unlearned = get_subgraph(
        exp.influenced_nodes, exp.dataset_train_unlearned)

    # Full-batch loader shims (disclosed; training/eval only — ETR's
    # unlearning computations are already full-batch official code).
    exp.dataset_train_loader = fullbatch_loader(exp.dataset_train,
                                                exp.dataset_train.train_mask)
    exp.data_test_loader = fullbatch_loader(exp.data, exp.data.test_mask)
    exp.dataset_train_unlearned_loader = fullbatch_loader(
        exp.dataset_train_unlearned, exp.dataset_train_unlearned.train_mask)
    exp.dataset_unlearned_test_loader = fullbatch_loader(
        exp.dataset_unlearned, exp.dataset_unlearned.test_mask)

    exp.determine_target_model()    # their NodeClassifier GCN (dim 256)
    exp.num_layers = 2

    t0 = time.time()
    train_time, _ = exp._train_model()
    base_train_s = time.time() - t0

    def macro_f1(model, edge_index, mask):
        model.eval()
        with torch.no_grad():
            pred = model(data.x, edge_index).argmax(1)
        m = mask & labeled
        return float(f1_score(data.y[m].numpy(), pred[m].numpy(),
                              average="macro"))

    f1_before = macro_f1(exp.target_model.model, data.edge_index,
                         data.test_mask)

    t0 = time.time()
    _etr_internal_time, acc_after = exp.etr()
    t_procedure = time.time() - t0  # full procedure incl. FIM/grad passes

    model = exp.model_unlearn
    f1_after = macro_f1(model, directed(remain_e), data.test_mask)

    # --- FGS scoring of the unlearned model ---
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
        "method": "ETR (official AllminerLab/ETR @ c88ec74, FGS adapter)",
        "dataset": dataset, "seed": seed, "n_forget": len(forget_e),
        "attack_auc": round(attack, 4), "util_auc": round(util, 4),
        "retrain_attack_auc": round(rt["attack"], 4),
        "retrain_util_auc": round(rt["util"], 4),
        "attack_gap_vs_retrain": round(attack - rt["attack"], 4),
        "util_gap_vs_retrain": round(util - rt["util"], 4),
        "procedure_s": round(t_procedure, 3),
        "retrain_s": rt["s"],
        "time_ratio_procedure_vs_retrain": round(t_procedure / rt["s"], 4),
        "base_train_s_etr_classifier": round(base_train_s, 1),
        "etr_f1_before": round(f1_before, 4),
        "etr_f1_after": round(f1_after, 4),
        "etr_test_acc_after": round(float(acc_after), 4),
        "n_influenced_nodes": int(len(exp.influenced_nodes)),
        "erase_ratio_m": args["erase_ratio"], "lambda_l": args["l"],
        "lr": args["lr"], "wd": args["wd"],
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"fgs_etr_{dataset}_seed{seed}.json")
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
            import traceback
            traceback.print_exc()
            print(f"FAILED {ds} seed {s}: {type(e).__name__}: {e}",
                  flush=True)
