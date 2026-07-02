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

## Metrics (per-seed mean ± std, common holdout)
- _TODO: paste final ROC-AUC / MCC / F1 / ACC / AUPRC for 888 & nn05 329._
