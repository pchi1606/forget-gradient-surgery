# Reproduction package — Forget Gradient Surgery (FGS) for graph edge unlearning

**Author:** Prabhu Rangaraju

One-command reproduction of the results in *"Forget Gradient Surgery:
A Characterization of Projected Gradient Descent for Graph Edge
Unlearning"* (manuscript under review).

## One command

```bash
bash reproduce.sh
```

Requires Python 3.10+ and `pip`. No GPU needed or used at any step.

## What it does

1. Installs Python dependencies (`requirements.txt`: numpy, scipy —
   pure NumPy/SciPy implementation, no PyTorch, no GPU needed or used).
2. Fetches the Planetoid datasets (Cora / CiteSeer / PubMed) if not
   already bundled in `data/` (9.9 MB; DGL mirror).
3. Runs the GCN sweep (`run_sweep.sh`): 30 runs (3 datasets × 10 seeds).
   Each run trains a 2-layer GCN link predictor, forgets 5% of train
   edges, and records wall-clock time and attack/utility metrics for
   retrain-from-scratch vs FGS (KS-stopped) vs attack-calibrated FGS
   vs fine-tune / gradient-ascent baselines, plus the no-projection
   ablation (`experiment_addon3.py`).
4. Verifies headline manuscript numbers (`scripts/verify_reproduction.py`)
   and prints PASS/FAIL per check.

Expected runtime: ~7 hours single-threaded on a modern CPU
(`OMP_NUM_THREADS=1`; the cost is GCN training plus unlearning steps).
Outputs land in `results/per_seed/*.json` and
`results/cache/*.npz`. All stochastic steps use fixed seeds; reruns
are deterministic.

## Additional drivers (shipped, not run by reproduce.sh)

- `experiment_hybrid.py` — the cheap hybrid stopping variant (Table 4);
  reuses the 40-step trajectories, no new gradient runs.
- `experiment_ff.py` — the fraction-sensitivity sweep (1%/10%/20%
  deletion, 5 seeds per fraction per dataset; Figure 3).
- `experiment_fixedstep.py` — the fixed-step stopping ablation (Table 6);
  outputs in `results/fixedstep.json` (the 288 MB per-iterate arrays
  are not shipped; the 172 KB summary is).
- `analysis_v2.py` — regenerates Tables 3–6 from `results/per_seed/`.
- `sage.py` — GraphSAGE backbone for the smoke test (Section 6.4).

## Protocol equivalence (seeds 7/42/123)

The CiteSeer columns pool ten seeds. Seeds 7, 42, and 123 were run
earlier (filenames `<ds>_seed<N>.json`); the other seven use
`gcn_<ds>_seed<N>.json`. We verified the earlier runs use identical
code paths (`experiment.py`), data splits (70/15/15), hyperparameters
(2-layer GCN, 32-dim hidden, Adam 1e-2, 150 epochs), and forget
fraction (5%). The pooled comparison is valid; all statistical tests
are within-seed paired.

## What is verified

- KS-stopped FGS / retrain wall-time ratios, median over 10 seeds.
- **Accuracy headline:** CiteSeer FGS attack gap to retrain ~ +0.037.
- **Accuracy headline:** Cora no-projection ablation utility gap ~ -0.048.
- **Accuracy headline:** paired FGS − attack-calibrated FGS gap ~ +0.028
  on CiteSeer (the attack-calibration gain).
- Presence of all 30 per-seed result files, each containing the
  attack-calibrated variant and the no-projection ablation.

## Data

`data/` bundles the Planetoid Cora / CiteSeer / PubMed files
(`ind.*` format, 9.9 MB total) sourced from the DGL mirror
(`https://data.dgl.ai/dataset/{cora,citeseer,pubmed}.zip`; the original
kimiyoung/planetoid repo now 404s). `fetch_data.sh` re-downloads them
from the same mirror if missing.

## Citation

See `CITATION.md`.
