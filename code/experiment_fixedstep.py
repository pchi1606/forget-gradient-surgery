"""Task A — Fixed-step stopping ablation for FGS (graph-unlearning).

Addresses the audit gap: "no fixed-step comparison" for the adaptive
(KS) stopping rule.

For each of the 15 GCN configs (cora/citeseer/pubmed x seeds
42/7/123/2024/31337):
  1. Load the cached base params + splits (results/cache/), identical
     forget set / attack negatives / test set as the published runs.
  2. Sample the retain-negative anchor FRESH with a documented stream:
     rng = np.random.default_rng(seed*1000 + 5). (The published runs used
     an in-flight rng; REPORT section 6.5 bounds anchor sensitivity at
     ~0.005, so we VALIDATE the re-run's KS-selected attack against the
     published per-seed fgs attack with tolerance 0.01.)
  3. Run the projected KS trajectory (max_steps=40, keep_iterates=True),
     then score attack AUC and utility AUC at EVERY iterate (0..40).
  4. Record per-iterate (step, ks, attack, util); per config compute the
     attack/util gap vs retrain at the KS-selected step and at FIXED
     steps {0, 10, 20, 30, 40} (step 0 = unlink-only baseline).

Outputs:
  results/fixedstep.json
  results/fixedstep_iterates/<dataset>_seed<seed>_iterates.npz  (per config)

The summary: for each fixed step S, the paired mean difference
(KS-selected gap minus fixed-S gap) across the 15 configs with paired
t-test p-value -- evidence for/against "adaptive stopping beats
fixed-step schedules".

Usage: python3 experiment_fixedstep.py            # all 15 configs
       python3 experiment_fixedstep.py cora 42    # single config -> /tmp
"""
import json
import os
import sys
import time

import numpy as np
from scipy.stats import rankdata, ttest_rel

from data import load_planetoid
from unlearn import unlearn_fgs
import gcn as M

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "results", "cache")
PER_SEED = os.path.join(HERE, "results", "per_seed")
OUT_DIR = os.path.join(HERE, "results", "fixedstep_iterates")

DATASETS = ("cora", "citeseer", "pubmed")
SEEDS = (42, 7, 123, 2024, 31337)
LEGACY_SEEDS = (42, 7, 123)  # cache/json without the "gcn_" prefix
FIXED_STEPS = (0, 10, 20, 30, 40)
FGS_ETA = 0.05
RN_SEED_OFFSET = 1000   # retain-neg anchor: rng(seed*1000 + 5)
RN_SEED_ADD = 5
ATTACK_TOL = 0.01       # validation tolerance vs published fgs attack


def auc_score(y, s):
    r = rankdata(s)
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# --- verbatim copies from experiment.py (same sampling behavior) ---
def sample_non_edges(rng, n, edge_set, k, allowed):
    allowed = np.asarray(allowed)
    out, seen = [], set()
    while len(out) < k:
        a = allowed[rng.integers(0, len(allowed))]
        b = allowed[rng.integers(0, len(allowed))]
        if a == b:
            continue
        e = (int(a), int(b)) if a < b else (int(b), int(a))
        if e in edge_set or e in seen:
            continue
        seen.add(e)
        out.append(e)
    return out


def hard_negatives(rng, forget_edges, labels, edge_set, by_label):
    """One non-edge per forget edge, matched on the sorted label pair."""
    out, seen = [], set()
    for (u, v) in forget_edges:
        key = tuple(sorted((int(labels[u]), int(labels[v]))))
        cands = by_label.get(key)
        if not cands:
            continue
        while True:
            a = cands[rng.integers(0, len(cands))]
            b = cands[rng.integers(0, len(cands))]
            if a == b:
                continue
            e = (a, b) if a < b else (b, a)
            if e in edge_set or e in seen:
                continue
            seen.add(e)
            out.append(e)
            break
    return out
# --- end verbatim copies ---


def _pairs(edgelist):
    u = np.array([e[0] for e in edgelist], dtype=np.int64)
    v = np.array([e[1] for e in edgelist], dtype=np.int64)
    return u, v


def cache_path(dataset, seed):
    p1 = os.path.join(CACHE_DIR, f"gcn_{dataset}_seed{seed}_base.npz")
    p2 = os.path.join(CACHE_DIR, f"{dataset}_seed{seed}_base.npz")
    return p1 if os.path.exists(p1) else p2


def json_path(dataset, seed):
    pre = "" if seed in LEGACY_SEEDS else "gcn_"
    return os.path.join(PER_SEED, f"{pre}{dataset}_seed{seed}.json")


def load_config(dataset, seed):
    Xsp, labels, edges = load_planetoid(dataset)
    n = Xsp.shape[0]
    X = Xsp.toarray().astype(np.float32)
    edge_set = set(edges)
    labeled = np.where(labels >= 0)[0]
    z = np.load(cache_path(dataset, seed))
    base = {k: z[k] for k in ("W1", "W2")}
    cfg = {"n": n, "X": X, "labels": labels, "edge_set": edge_set,
           "labeled": labeled, "base": base}
    for key in ("train_e", "val_e", "test_e", "train_neg", "test_neg",
                "forget_e", "remain_e", "attack_neg"):
        cfg[key] = [tuple(e) for e in z[key]]
    return cfg


def run_config(dataset, seed):
    t0 = time.time()
    c = load_config(dataset, seed)
    n, X = c["n"], c["X"]
    forget_e, remain_e = c["forget_e"], c["remain_e"]
    val_e, test_e, test_neg, attack_neg = (c["val_e"], c["test_e"],
                                          c["test_neg"], c["attack_neg"])
    # documented fresh retain-negative anchor (NOT the in-flight rng)
    rng_rn = np.random.default_rng(seed * RN_SEED_OFFSET + RN_SEED_ADD)
    rn = sample_non_edges(rng_rn, n, c["edge_set"], len(remain_e), c["labeled"])

    print(f"[fixedstep gcn {dataset} seed={seed}] trajectory...", flush=True)
    p, t_traj, n_steps, diag = unlearn_fgs(
        M, c["base"], forget_e, remain_e, n, X,
        retain_pos=remain_e, retain_neg=rn, eta=FGS_ETA,
        val_edges=val_e, max_steps=40, stop_rule="ks",
        use_projection=True, keep_iterates=True)
    iterates = diag["iterates"]
    K = len(iterates)
    chosen = int(diag["chosen_step"])
    assert K == 41, f"expected 41 iterates (0..40), got {K}"

    # score every iterate: one forward, attack + utility from the same Z
    An = M.norm_adj(remain_e, n)  # post-deletion adjacency (as in unlearn_fgs)
    fu, fv = _pairs(forget_e)
    au, av = _pairs(attack_neg)
    tu, tv = _pairs(test_e)
    eu, ev = _pairs(test_neg)
    atk_u = np.concatenate([fu, au])
    atk_v = np.concatenate([fv, av])
    tst_u = np.concatenate([tu, eu])
    tst_v = np.concatenate([tv, ev])
    y_attack = np.array([1] * len(forget_e) + [0] * len(attack_neg))
    y_test = np.array([1] * len(test_e) + [0] * len(test_neg))
    attacks, utils = [], []
    t1 = time.time()
    for prm in iterates:
        Z, _ = M.forward(prm, An, X)
        attacks.append(auc_score(y_attack, M.pair_scores(Z, atk_u, atk_v)))
        utils.append(auc_score(y_test, M.pair_scores(Z, tst_u, tst_v)))
    t_score = time.time() - t1

    # published reference values
    pub = json.load(open(json_path(dataset, seed)))
    ret_atk = pub["methods"]["retrain"]["attack"]
    ret_utl = pub["methods"]["retrain"]["util"]
    pub_fgs_atk = pub["methods"]["fgs"]["attack"]
    pub_fgs_steps = pub["methods"]["fgs"].get("steps")

    ks_vals = [float(v) for v in diag["ks"]]
    attacks = [float(v) for v in attacks]
    utils = [float(v) for v in utils]
    atk_chosen = attacks[chosen]
    dev = abs(atk_chosen - pub_fgs_atk)

    rec = {
        "dataset": dataset, "seed": seed,
        "cache_file": os.path.basename(cache_path(dataset, seed)),
        "json_file": os.path.basename(json_path(dataset, seed)),
        "rn_seed_stream": f"np.random.default_rng({seed}*1000+5)",
        "retrain_attack": ret_atk, "retrain_util": ret_utl,
        "published_fgs_attack": pub_fgs_atk,
        "published_fgs_steps": pub_fgs_steps,
        "rerun_chosen_step": chosen,
        "rerun_attack_at_chosen": atk_chosen,
        "rerun_util_at_chosen": utils[chosen],
        "attack_deviation_vs_published": dev,
        "n_iterates": K, "traj_seconds": round(t_traj, 1),
        "score_seconds": round(t_score, 1),
        "ks": ks_vals, "attack": attacks, "util": utils,
        "attack_gap_ks": atk_chosen - ret_atk,
        "util_gap_ks": utils[chosen] - ret_utl,
        "fixed": {},
    }
    for s in FIXED_STEPS:
        rec["fixed"][str(s)] = {
            "attack": attacks[s], "util": utils[s],
            "attack_gap": attacks[s] - ret_atk,
            "util_gap": utils[s] - ret_utl,
        }
    rec["total_seconds"] = round(time.time() - t0, 1)
    print(f"  chosen={chosen} attack_ks={atk_chosen:.4f} "
          f"published={pub_fgs_atk:.4f} dev={dev:.4f}", flush=True)

    # persist iterates for Task B (hybrid); float32 like the params
    os.makedirs(OUT_DIR, exist_ok=True)
    npz_path = os.path.join(OUT_DIR, f"{dataset}_seed{seed}_iterates.npz")
    payload = {}
    for i, prm in enumerate(iterates):
        for k, v in prm.items():
            payload[f"iter{i:02d}_{k}"] = v
    payload["ks"] = np.array(ks_vals)
    payload["attack"] = np.array(attacks)
    payload["util"] = np.array(utils)
    payload["chosen_step"] = np.array([chosen])
    np.savez(npz_path, **payload)
    rec["iterates_file"] = os.path.relpath(npz_path, HERE)
    return rec


def summarize(configs):
    """Paired comparison: (KS-selected gap - fixed-S gap) per config."""
    out = {}
    for s in FIXED_STEPS:
        d_atk = np.array([c["attack_gap_ks"] - c["fixed"][str(s)]["attack_gap"]
                          for c in configs])
        d_utl = np.array([c["util_gap_ks"] - c["fixed"][str(s)]["util_gap"]
                          for c in configs])

        def stats(d):
            t = ttest_rel(d, np.zeros(len(d)))
            return {
                "n": len(d),
                "mean_diff": float(d.mean()),
                "std_diff": float(d.std(ddof=1)),
                "cohen_dz": float(d.mean() / d.std(ddof=1)),
                "t_p": float(t.pvalue),
                "note": ("paired t-test of (KS-selected gap minus fixed-step "
                         "gap); negative mean_diff on ATTACK = KS-selected "
                         "iterate is closer to retrain than the fixed step"),
            }
        out[str(s)] = {
            "attack": stats(d_atk),
            "util": stats(d_utl),
            "mean_gap_ks_attack": float(np.mean(
                [c["attack_gap_ks"] for c in configs])),
            "mean_gap_fixed_attack": float(np.mean(
                [c["fixed"][str(s)]["attack_gap"] for c in configs])),
            "mean_gap_ks_util": float(np.mean(
                [c["util_gap_ks"] for c in configs])),
            "mean_gap_fixed_util": float(np.mean(
                [c["fixed"][str(s)]["util_gap"] for c in configs])),
        }
    return out


def main(configs, out_path):
    recs = [run_config(ds, seed) for (ds, seed) in configs]
    devs = [r["attack_deviation_vs_published"] for r in recs]
    result = {
        "description": ("Fixed-step stopping ablation for FGS (Task A). "
                        "Re-runs the 40-step projected KS trajectory with a "
                        "documented fresh retain-negative anchor "
                        "(seed*1000+5) and scores attack/utility AUC at every "
                        "iterate 0..40."),
        "backbone": "gcn",
        "n_configs": len(recs),
        "fixed_steps": list(FIXED_STEPS),
        "configs": recs,
        "fixed_step_summary": summarize(recs),
        "validation": {
            "check": ("re-run KS-selected attack vs published per-seed fgs "
                      "attack (anchor-resampling noise bound ~0.005)"),
            "tolerance": ATTACK_TOL,
            "max_attack_deviation": float(max(devs)),
            "pass": bool(max(devs) <= ATTACK_TOL),
            "per_config_deviation": {
                f"{r['dataset']}_seed{r['seed']}":
                    round(r["attack_deviation_vs_published"], 5)
                for r in recs},
        },
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1)
    print(f"\nwrote {out_path}")
    print(f"validation: max deviation {max(devs):.4f} "
          f"(tol {ATTACK_TOL}) -> {'PASS' if max(devs) <= ATTACK_TOL else 'FAIL'}")
    for s in FIXED_STEPS:
        sm = result["fixed_step_summary"][str(s)]
        print(f"  fixed step {s:>2}: attack diff "
              f"{sm['attack']['mean_diff']:+.4f} p={sm['attack']['t_p']:.3f} | "
              f"util diff {sm['util']['mean_diff']:+.4f} p={sm['util']['t_p']:.3f}")
    return result


if __name__ == "__main__":
    if len(sys.argv) == 3:
        ds, seed = sys.argv[1], int(sys.argv[2])
        r = run_config(ds, seed)
        print(json.dumps({k: r[k] for k in (
            "dataset", "seed", "rerun_chosen_step", "rerun_attack_at_chosen",
            "attack_deviation_vs_published", "n_iterates")}, indent=1))
    else:
        cfgs = [(ds, s) for ds in DATASETS for s in SEEDS]
        main(cfgs, os.path.join(HERE, "results", "fixedstep.json"))
