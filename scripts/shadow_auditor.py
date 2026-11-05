#!/usr/bin/env python3
"""Shadow-retrain auditor test (manuscript Section 8, Limitation 5).

Question: the attack-calibrated stopping rule selects the trajectory
iterate whose forget-edge ranking best matches the validation-edge
ranking under the *evaluation auditor's own scorer family*. Does the
CiteSeer gap closure survive an auditor that calibrates against an
independent reference -- a per-seed shadow retrain (a retrain-from-
scratch model the auditor trains itself, with its own initialization
seed, on the post-deletion graph)?

Procedure per CiteSeer seed (the paper's ten seeds):
  1. Rebuild the configuration and the 80-candidate attack trajectory
     with the frozen harness (scripts/pipeline.py replay of
     code/experiment.py; code/unlearn.py's projected trajectory with
     keep_iterates=True). The attack-calibrated stop is re-selected on
     this trajectory exactly as in the headline run.
  2. Train a shadow retrain: same post-deletion graph and training
     negatives as the reference retrain, but initialized from the
     auditor's own rng (seed + 777000), i.e. a model the unlearning
     procedure never saw.
  3. Shadow stop: the trajectory iterate whose forget-edge *scores*
     are closest in Kolmogorov-Smirnov distance to the shadow
     retrain's forget-edge scores.
  4. Report both stops' attack gaps to the *archived* reference
     retrain of the same seed (results/per_seed/).

What is recomputed vs reused -- stated plainly:
  * Recomputed here: the base model, the splits/sets (verified against
    the archived base attack AUC), the full 80-step trajectory and its
    per-iterate forget-edge scores, the shadow retrain, both stopping
    selections, and all gap aggregates.
  * Reused from the archive: only the reference retrain's attack AUC
    per seed (the gap denominator, from results/per_seed/) and the
    manuscript's expected values (expected/headline_numbers.json,
    key shadow_auditor_citeseer).

Manuscript values this test produced originally: attack-calibrated
CiteSeer gap +0.008 +/- 0.070 (median stop 30); shadow-calibrated gap
+0.060 +/- 0.050 (median stop 17); paired difference +0.052, p = 0.003.

Runtime: roughly 5-7 minutes per seed single-threaded (base training
and shadow training dominate); --jobs parallelizes across seeds.
Not part of reproduce.sh (which must stay short); run standalone:
    python3 scripts/shadow_auditor.py --jobs 4
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.stats import ks_2samp, ttest_rel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import (EPOCHS, HID, LR, FGS_ETA, archived_per_seed,  # noqa: E402
                      attack_auc, build_config, draw_retain_negatives,
                      draw_stop_negatives)

SHADOW_SEED_OFFSET = 777000
SEEDS = (42, 7, 123, 2024, 2025, 31337, 31338, 5555, 7777, 999)


def run_seed(seed):
    from unlearn import unlearn_fgs
    cfg = build_config("citeseer", seed)
    M = cfg["M"]
    arch = archived_per_seed("citeseer", seed)
    base_attack = attack_auc(cfg, cfg["base"], cfg["train_e"])
    assert abs(base_attack - arch["base_attack_auc"]) < 2e-3, (
        f"seed {seed}: replay base attack {base_attack:.4f} != archived "
        f"{arch['base_attack_auc']:.4f}; configuration replay diverged")
    rn = draw_retain_negatives(cfg)
    stop_neg = draw_stop_negatives(cfg, seed)
    _, _, _, diag = unlearn_fgs(
        M, cfg["base"], cfg["forget_e"], cfg["remain_e"], cfg["n"],
        cfg["X"], cfg["remain_e"], rn, eta=FGS_ETA,
        val_edges=cfg["val_e"], max_steps=80, stop_rule="attack",
        stop_neg=stop_neg, keep_iterates=True)
    iterates = diag["iterates"]
    attack_step = int(diag["chosen_step"])
    shadow, _ = M.train(cfg["remain_e"], cfg["n"], cfg["X"],
                        cfg["train_neg"], EPOCHS, lr=LR, hid=HID,
                        rng=np.random.default_rng(seed + SHADOW_SEED_OFFSET),
                        verbose=False)
    sf_shadow = M.scores_for(shadow, cfg["remain_e"], cfg["n"], cfg["X"],
                             cfg["forget_e"])
    best_step, best_ks = None, float("inf")
    attacks = []
    for i, prm in enumerate(iterates):
        sf = M.scores_for(prm, cfg["remain_e"], cfg["n"], cfg["X"],
                          cfg["forget_e"])
        ks = float(ks_2samp(sf, sf_shadow).statistic)
        attacks.append(attack_auc(cfg, prm, cfg["remain_e"]))
        if ks < best_ks:
            best_ks, best_step = ks, i
    retrain_attack = arch["methods"]["retrain"]["attack"]
    row = {
        "seed": seed,
        "shadow_step": int(best_step),
        "shadow_ks": best_ks,
        "shadow_gap": attacks[best_step] - retrain_attack,
        "attack_step": attack_step,
        "attack_gap": attacks[attack_step] - retrain_attack,
        "archived_fgs_attack_gap": (arch["methods"]["fgs_attack"]["attack"]
                                    - retrain_attack),
    }
    print(f"seed {seed}: shadow stop {best_step} gap {row['shadow_gap']:+.4f}; "
          f"attack-calibrated stop {attack_step} gap {row['attack_gap']:+.4f} "
          f"(archived {row['archived_fgs_attack_gap']:+.4f})", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()
    from fetch_data import fetch
    fetch("citeseer")  # no-op if the data files are already present
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            rows = list(ex.map(run_seed, args.seeds))
    else:
        rows = [run_seed(s) for s in args.seeds]
    rows.sort(key=lambda r: r["seed"])
    sg = np.array([r["shadow_gap"] for r in rows])
    ag = np.array([r["attack_gap"] for r in rows])
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "expected", "headline_numbers.json")) as f:
        exp = json.load(f)["shadow_auditor_citeseer"]
    print()
    print(f"CiteSeer shadow-auditor result (n={len(rows)}):")
    print(f"  shadow-calibrated gap  {sg.mean():+.3f} +/- {sg.std(ddof=1):.3f} "
          f"(manuscript {exp['shadow_gap_mean']:+.3f} +/- "
          f"{exp['shadow_gap_sd']:.3f}); median stop "
          f"{int(np.median([r['shadow_step'] for r in rows]))} "
          f"(manuscript {exp['median_shadow_step']})")
    print(f"  attack-calibrated gap  {ag.mean():+.3f} +/- {ag.std(ddof=1):.3f} "
          f"(manuscript {exp['attack_calibrated_gap_mean']:+.3f} +/- "
          f"{exp['attack_calibrated_gap_sd']:.3f}); median stop "
          f"{int(np.median([r['attack_step'] for r in rows]))} "
          f"(manuscript {exp['median_attack_step']})")
    if len(rows) > 2:
        t = ttest_rel(sg, ag)
        print(f"  paired difference {float(np.mean(sg - ag)):+.3f}, "
              f"paired t p = {t.pvalue:.4f} (manuscript +0.052, p = 0.003)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
