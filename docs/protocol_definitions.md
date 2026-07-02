# Protocol definitions

> Placeholder — copy the authoritative version from
> `/home/minji/BBB_paper/protocol_definitions.md` in the file-population step.

## Protocol I (initial)
- Train on **internal only**, 8:1:1 split.
- Evaluate on internal test / external / holdout (1089).
- Best feature combo: `maccs + avalon + rdkit + mole`.
- Generalization assessed via **external validation**.

## Protocol II (refined)
- Train on **internal + external integrated**, 8:2 split (no separate test).
- Evaluate on similarity-filtered holdout: total **888** / nn05 **329**.
- No separate external set (external folded into training).
- Best feature combo: `maccs + scage1 + mole`.
- Generalization assessed via **similarity-controlled holdout**.

Holdout nesting: `329 ⊆ 888 ⊆ 1089`. Cross-protocol comparison uses common 888 / 329.
