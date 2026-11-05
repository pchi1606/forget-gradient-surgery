#!/usr/bin/env python3
"""Smoke end-to-end re-run (invoked by reproduce.sh).

Runs ONE configuration -- Cora, seed 42, FGS with KS stopping plus the
retrain-from-scratch reference -- through the real pipeline code in
code/ (base training, forget-set construction, projected trajectory with
KS stopping, retraining, attack/utility scoring; nothing is loaded from
the archived outputs except the values compared against at the end).

The re-run attack/utility gaps (method minus retrain, same seed) must
match the archived per-seed values in results/per_seed/cora_seed42.json
within TOLERANCE. The manuscript reports that frozen-pipeline re-runs of
the earlier-batch seeds reproduce every method within 3 x 10^-3
(floating-point noise); this smoke check allows 5 x 10^-3 to absorb
BLAS/library variation across machines while remaining far below any
effect size reported in the paper (smallest headline gap magnitude of
interest: 0.008).

Runtime: about 2 minutes on 2 CPUs.
"""
import sys
import time

from pipeline import (EPOCHS, HID, LR, FGS_ETA, archived_per_seed,
                      attack_auc, build_config, draw_retain_negatives,
                      util_auc)

TOLERANCE = 5e-3
DATASET, SEED = "cora", 42


def main():
    from unlearn import unlearn_fgs, unlearn_retrain
    import numpy as np
    t0 = time.time()
    cfg = build_config(DATASET, SEED)
    M = cfg["M"]
    arch = archived_per_seed(DATASET, SEED)
    base_attack = attack_auc(cfg, cfg["base"], cfg["train_e"])
    print(f"base attack AUC re-run {base_attack:.4f} "
          f"(archived {arch['base_attack_auc']:.4f})")

    p, t_retrain = unlearn_retrain(
        M, cfg["remain_e"], cfg["n"], cfg["X"], cfg["train_neg"],
        epochs=EPOCHS, hid=HID, lr=LR,
        rng=np.random.default_rng(SEED + 999))
    retrain = (attack_auc(cfg, p, cfg["remain_e"]),
               util_auc(cfg, p, cfg["remain_e"]))
    print(f"retrain re-run: attack {retrain[0]:.4f} util {retrain[1]:.4f} "
          f"({t_retrain:.1f}s)")

    rn = draw_retain_negatives(cfg)
    p, t_fgs, steps, _ = unlearn_fgs(
        M, cfg["base"], cfg["forget_e"], cfg["remain_e"], cfg["n"],
        cfg["X"], cfg["remain_e"], rn, eta=FGS_ETA,
        val_edges=cfg["val_e"], stop_rule="ks")
    fgs = (attack_auc(cfg, p, cfg["remain_e"]),
           util_auc(cfg, p, cfg["remain_e"]))
    print(f"fgs (KS) re-run: attack {fgs[0]:.4f} util {fgs[1]:.4f} "
          f"steps {steps} ({t_fgs:.1f}s)")

    ok = True
    for i, name in enumerate(["attack", "util"]):
        got = fgs[i] - retrain[i]
        want = (arch["methods"]["fgs"][name]
                - arch["methods"]["retrain"][name])
        d = abs(got - want)
        status = "PASS" if d <= TOLERANCE else "FAIL"
        if d > TOLERANCE:
            ok = False
        print(f"SMOKE {status}: fgs {name} gap re-run {got:+.4f} vs "
              f"archived {want:+.4f} (|diff| {d:.4f}, tol {TOLERANCE})")
    print(f"smoke wall time {time.time() - t0:.0f}s")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
