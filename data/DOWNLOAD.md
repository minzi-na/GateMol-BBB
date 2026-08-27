# Downloading model weights & precomputed features

Large binary artifacts are archived on Zenodo (not in git).

> **Zenodo DOI (all versions):** [10.5281/zenodo.22119636](https://doi.org/10.5281/zenodo.22119636)
> **This release (v1.0.0):** [10.5281/zenodo.22119637](https://doi.org/10.5281/zenodo.22119637)
>
> Cite the all-versions DOI unless you specifically mean v1.0.0. The deposit is
> pending publication; the DOIs resolve once it is released.

**License:** the deposit is released under CC BY-NC 4.0 — non-commercial use
only. Both protocols consume MolE pretrained embeddings as an input token, and
MolE is itself CC BY-NC 4.0.

## Archive contents

| File | Size | Contents |
|---|---|---|
| `protocolI.zip` | 756,325,555 B | Protocol I final model — 10 × (`model.pth` 82 MB, `rdkit_scaler.pkl`, `config.json`), `summary.json`, `verify_load.json` |
| `protocolII.zip` | 295,841,711 B | Protocol II final model (iter198) — 10 × `seed<NNNN>.pt` 32 MB, `meta.json`, `verify_gates{,_allsubsets}.json` |
| `features.zip` | 16,240,983 B | Cached holdout features — `{internal,external,nn03,nn05,total}.npz` + per-subset manifests |
| `eval.zip` | 181,764 B | 29 holdout evaluation records from the Protocol II architecture search, `results.tsv`, `reproducibility_report.md`, `architecture_log.md` |
| `MANIFEST.md` | 7,501 B | Full file inventory, per-subset metrics, provenance |
| `SHA256SUMS.txt` | 9,332 B | Checksums for all 91 payload files |

Total 1,068,606,846 B. Each zip expands to a directory of the same name, so
unzipping all four in one place reproduces the deposit layout:

```
protocolI/seed_{42,100,...,900}/{model.pth,rdkit_scaler.pkl,config.json}
protocolI/{summary.json,verify_load.json}
protocolII/{seed0042.pt,...,seed0900.pt,meta.json,verify_gates.json,verify_gates_allsubsets.json}
features/{internal,external,nn03,nn05,total}.npz
eval/protocolII_holdout_eval/iter<NNNN>_<commit>.json
```

Note the two protocols store weights differently — Protocol I uses one directory
per seed, Protocol II a flat `seed<NNNN>.pt`.

## Verifying

```bash
unzip -q 'protocol*.zip' 'features.zip' 'eval.zip'
sha256sum -c SHA256SUMS.txt     # covers all 91 payload files
```

`SHA256SUMS.txt` deliberately excludes the documentation files, so editing them
never invalidates the checksums.

## Usage

```bash
# after unzipping into the repo root (or pass explicit paths)
python -m gatemol_bbb.evaluate --protocol 2 --weights protocolII \
    --features features/total.npz
```

## Notes

- Only the **final** model of each protocol is released (per study scope).
- **Pretrained encoders (SCAGE, MolE) are not in this deposit.** Get them from
  upstream — [SCAGE](https://github.com/KazeDog/scage),
  [MolE](https://github.com/recursionpharma/mole_public). They are only needed to
  regenerate features from SMILES; the cached `features/*.npz` reproduce the
  reported numbers without them.
- Reproduction requires **RDKit 2024.09.6** — the descriptor count changes
  between releases and Protocol I depends on the 217-descriptor layout.
- Reported metrics are **per-seed means over 10 seeds**, not ensemble figures.
  Soft-vote and other subsets appear in the deposit's JSON records for
  provenance but are not the manuscript numbers.
