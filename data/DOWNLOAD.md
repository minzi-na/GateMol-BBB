# Downloading model weights & precomputed features

Large binary artifacts are archived on Zenodo (not in git).

> **Zenodo DOI:** _TODO — add after upload_

## Archive contents (planned)

| Archive | Source (local, do not commit) | Size | Contents |
|---|---|---|---|
| `protocol1_weights.tar.gz` | `bbb-combo1/bbb_artifacts/phase2_best/` | ~784M | 10-seed `model.pth`, `config.json`, `rdkit_scaler.pkl` |
| `protocol2_weights.tar.gz` | `combos-v2-combo2/results/` (final iter198) | ~290M | 10-seed weights + configs |
| `pretrained_encoders.tar.gz` | `BBB/scage/weights/`, MolE checkpoint | ~200M | SCAGE + MolE encoders needed for embedding extraction |
| `feature_cache.tar.gz` | `feature_cache_merged/{pool.npz,holdout.npz}` | ~206M | precomputed multimodal features (train pool + holdout) |

## Usage

```bash
# after downloading & extracting into ./artifacts and ./features
python -m gatemol_bbb.evaluate --protocol 2 --weights artifacts/protocol2 \
    --features features/holdout.npz
```

## Notes

- Only the **final** model of each protocol is released (per study scope).
- Extraction scripts under `../scripts/` can regenerate `feature_cache` from SMILES
  if you prefer not to download it (requires the pretrained encoders).
