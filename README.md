# Reproduction package — Forget Gradient Surgery (FGS) for graph edge unlearning

This package accompanies the paper *Forget Gradient Surgery: A Characterization of
Projected Gradient Descent for Graph Edge Unlearning* (Prabhu Rangaraju), submitted to Neural Computing and
Applications, 2026. It contains the experiment harness, the analysis
scripts, the baseline adapters, and the raw per-seed result files from
which every aggregate number in the paper is computed.

## Quick start

```bash
bash reproduce.sh
```

This (1) recomputes every headline number from the archived per-seed
JSONs and compares them against `expected/headline_numbers.json`
(`scripts/verify_reproduction.py`; nonzero exit on any mismatch), then
(2) downloads the Cora data files from their public mirror and runs a
smoke end-to-end re-run of one configuration (Cora, seed 42: FGS with
KS stopping plus the retrain reference) through the real pipeline code,
checking its gaps against the archived per-seed values within
5 x 10^-3 (the paper reports frozen-pipeline re-runs reproduce within
3 x 10^-3; the smoke tolerance adds headroom for library variation).
Total runtime is a few minutes on 2 CPUs.

## Contents

- `reproduce.sh` — the one-command verification above.
- `scripts/verify_reproduction.py` — recomputation of the headline
  numbers: Table 4 attack/utility gaps for Fine-tune, NegGrad,
  GNNDelete, UtU (unlink-only), FGS (KS), FGS margin, FGS
  attack-calibrated and FGS no-projection; the GNNDelete lambda-sweep
  lambda=1.0 column (Table 5; see the archive note below); the
  fraction-sensitivity values of Section 6.2; the fixed-step ablation
  (Table 8); and the MEGU and ETR aggregates. It prints a pass/fail
  line per check and a count of values that are *not archived* (the
  lambda-sweep columns other than lambda=1.0 and the best-point
  utility values: their raw sweep outputs were not retained in the
  experiment tree, so the verifier reports them instead of passing
  them silently).
- `scripts/full_rerun.py` — frozen-pipeline re-run entry point:
  re-runs specified dataset/seed/method configurations through the
  harness and compares against the archived values (the seed-pooling
  exchangeability check and the same-machine timings of the paper are
  produced with this script).
- `scripts/compute_utu.py` — the unlink-only baseline from the harness
  (base model, no parameter update, scored on the post-deletion
  graph).
- `scripts/shadow_auditor.py` — the shadow-retrain auditor test of
  Section 8 (Limitation 5): per-seed shadow retrain vs
  attack-calibrated stopping on CiteSeer. It recomputes the base
  model, the 80-candidate trajectory, and a shadow retrain (auditor's
  own initialization seed), and reuses only the archived reference
  retrain AUCs as the gap denominator; its docstring states exactly
  what is recomputed vs reused. Standalone (not in `reproduce.sh`);
  roughly 5–7 minutes per seed, parallelizable with `--jobs`.
- `scripts/pipeline.py` — shared configuration builder used by the
  scripts above; it replays `code/experiment.py`'s rng stream exactly.
- `scripts/fetch_data.py` — downloads the Planetoid data files (see
  Data provenance below).
- `expected/headline_numbers.json` — the manuscript's printed values
  that the verifier checks against.
- `code/` — the experiment harness (`data.py`, `gcn.py`, `sage.py`,
  `unlearn.py`, `gnndelete.py`, `experiment*.py`), the analysis
  scripts (`analyze.py`, `analysis_v2.py`, `assemble_fixedstep.py`),
  run scripts, and `code/external/` with the baseline adapters
  (`megu/fgs_adapter_all.py`, `etr/fgs_adapter_all.py`,
  `gnndelete_cheng_metric.py`) together with the official MEGU/ETR
  code trees they wrap.
- `results/` — the archived raw outputs: `per_seed/` (headline
  n=10-seed runs), `per_seed_ff/` (deletion-fraction sweep),
  `fixedstep.json` (per-iterate fixed-step trajectories and the
  ablation summary), `analysis.json`, `analysis_v2.json`,
  `summary.csv`, `megu/` and `etr/` (per-seed outputs of the two
  added baselines), `gnndelete_validation/` (port-validation
  outputs).
- `manuscript/` — slot for the final submitted PDF (copied in when
  the filing packet is assembled; see `manuscript/README.md`).

## Environment

The headline harness (`code/gcn.py`, `code/unlearn.py`, the experiment
drivers) is pure NumPy/SciPy — no deep-learning framework is involved
in the paper's own method or baselines. The archived runs were
produced on a 2-CPU Linux VM with Python 3.12.3, NumPy 1.26.4,
SciPy 1.11.4. The verifier and the smoke re-run need only Python 3
with NumPy and SciPy.

The two external baselines are the exception: the MEGU and ETR
adapters under `code/external/` drive the official MEGU/ETR code,
which requires PyTorch and PyTorch Geometric. Their archived runs
used Python 3.12.3 with torch 2.14.0+cpu and torch-geometric
2.8.0.post1. Re-running those adapters is optional; their per-seed
outputs are archived under `results/megu/` and `results/etr/` and are
fully checked by the verifier.

## Data provenance, licenses, and split hygiene

Datasets: the Planetoid citation graphs Cora, CiteSeer, and PubMed
(Yang, Cohen & Salakhutdinov, ICML 2016), in the form distributed by
PyTorch Geometric and by the DGL mirror
(`https://data.dgl.ai/dataset/{cora,citeseer,pubmed}.zip`; the loader
in `code/data.py` documents this provenance — the original Planetoid
repository no longer serves the files). The datasets are publicly
distributed for research use; this package does **not** redistribute
the raw data. What the package contains is only derived result data:
per-seed aggregate metrics (AUCs, timings, step counts) as JSON/CSV
under `results/`. Raw graph files are obtained by the user via
`scripts/fetch_data.py` (invoked automatically by `reproduce.sh` for
the smoke run) and are covered by the datasets' original terms.

Splits and hygiene: the paper does not use the Planetoid node splits.
For each seed it draws its own edge split — 70% train / 15%
validation / 15% test edges — with the seed set
{42, 7, 123, 2024, 2025, 31337, 31338, 5555, 7777, 999} for the
headline (5% deletion) experiments; the deletion-fraction sweep
uses n=5 seeds per fraction, {42, 123, 999, 2024, 2025}. The forget set is 5% of the train edges (both endpoints
labeled); the retain set is the remaining train edges. Forget,
retain, validation, and test edge sets are disjoint by construction
(they partition the graph's edge set). Validation edges are never
trained on: they serve only the stopping rules (KS / margin /
attack-calibrated selection) and never enter any training loss.
Negative samples for training and evaluation are drawn disjoint from
all graph edges; the attack negatives are label-pair-matched hard
non-edges, and the attack-calibrated stopping rule's negatives are
drawn disjoint from the evaluation attack negatives (see
`code/experiment.py`). Fifteen CiteSeer nodes carry no label (-1) in
the Planetoid files; they take part in message passing but are
excluded from label-matched negative sampling and from the forget
set.

## License

Code in this package written for this paper is released under the MIT
License (see `LICENSE`). The MEGU and ETR code trees under
`code/external/` are the respective authors' official code, included
for reproducibility of the baseline runs and governed by their own
terms (see their READMEs). The Planetoid datasets are not included
(see above).

## Citation

See `CITATION.md`.
