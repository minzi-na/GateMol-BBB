# Reproducibility notes

## RDKit version (critical)

RDKit descriptor set changes across versions. This study used **RDKit 2024.9.6**,
which produces **217** descriptors. A different version changes the count and
misaligns the RDKit feature block, silently degrading results. Pin exactly.

## Seeds & splits

- 10 seeds: `[42, 100, 200, 300, 400, 500, 600, 700, 800, 900]`
- Scaffold split for all training/evaluation.
- Nested holdout: `329 ⊆ 888 ⊆ 1089`.

## Evaluation aggregation

Primary metric = **per-seed mean ± std** (seed-wise metric, then averaged).
Soft-voting ensemble reported as supplement only. Threshold = `prob > 0.5`.
Rationale (per-seed vs ensemble) is documented in the paper Methods.

## Environments used in the study

| Component | Env |
|---|---|
| gMLP (GateMol) | anaconda base |
| MLP / XGBoost / LightGBM baselines | `rapids-25.02` |
| TabPFN | `tabpfn` |

_TODO: export exact package versions per environment before release._
