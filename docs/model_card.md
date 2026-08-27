# Model card — GateMol-BBB

_TODO: fill before release._

## Overview
- **Task:** binary blood–brain barrier permeability (BBB+/BBB−) classification.
- **Architecture:** gMLP backbone with gated multimodal pooling (AutoResearch-discovered).
- **Variants:** Protocol I (initial) and Protocol II (refined). See `protocol_definitions.md`.

## Training data
- Protocol I: internal set only (8:1:1 split).
- Protocol II: internal + external integrated (8:2 split).
- _TODO: sizes, sources, label criterion (see gmbbb_label_criterion notes)._

## Intended use / limitations
- Research use for BBBP screening prioritization; not for clinical decisions.
- Performance validated on scaffold split + similarity-controlled holdout only.

## Metrics (per-seed mean ± std over 10 seeds, common holdout)

| Model | Subset | ROC-AUC | AUPRC | MCC | F1 | ACC |
|---|---|---|---|---|---|---|
| Protocol II | total (n=888) | 0.880 ± 0.004 | 0.933 ± 0.002 | 0.611 ± 0.017 | 0.900 ± 0.004 | 0.851 ± 0.006 |
| Protocol II | nn05 (n=329) | 0.870 ± 0.007 | 0.943 ± 0.005 | 0.564 ± 0.028 | 0.913 ± 0.006 | 0.860 ± 0.010 |
| Protocol I | total (n=888) | 0.799 ± 0.005 | 0.898 ± 0.004 | 0.305 ± 0.030 | 0.592 ± 0.065 | 0.563 ± 0.046 |
| Protocol I | nn05 (n=329) | 0.760 ± 0.010 | — | — | — | — |

- **Per-seed means, not ensemble figures.** The architecture search optimises
  single-model quality, while ensembling adds a seed-diversity bonus that
  favours high-variance tree baselines. Soft-vote numbers exist in the deposit's
  JSON records for provenance but are not the reported figures.
- Protocol I's threshold-dependent metrics (MCC / F1 / ACC) fall far below
  Protocol II's while its ranking metrics (ROC-AUC / AUPRC) hold up. Do not
  compare the two rows without accounting for the decision threshold.
- Sources: Protocol II — `ch3_full_metrics_888_nn05.csv`; Protocol I —
  `ch3_protocolI_tables.md` Table 3.4 / 3.5 (manuscript working files, not in
  this repo). Protocol I's nn05 row beyond ROC-AUC is not yet tabulated.

## Weights

Archived on Zenodo, CC BY-NC 4.0 (non-commercial — MolE's terms propagate):
[10.5281/zenodo.22119636](https://doi.org/10.5281/zenodo.22119636) (all
versions). Layout, integrity checks, and per-seed provenance records:
`../data/DOWNLOAD.md`.
