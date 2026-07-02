# Feature / embedding extraction

GateMol-BBB fuses up to 8 modalities. They come from two layers:

| Layer | Modalities | How | Module |
|---|---|---|---|
| RDKit-computable | ecfp, maccs, avalon, tt, rdkit | computed from SMILES | `rdkit_descriptors.py` |
| Pretrained encoders | scage1, scage2, mole | pre-extracted to CSV, then loaded | `scage_embed.py`, `mole_embed.py` |

`assemble_features.py` merges both layers into one `.npz` feature matrix
(the study's `prepare.py` assembly step). The one-shot encoder pipeline lives
in `../../../scripts/extract_embeddings.sh` (ported from the study's
`generate_all_embeddings.sh`): CSV → TXT → PKL → scage1 → scage2 → mole.

## Modality dimensions (RDKit 2024.9.6)

`ecfp 1024 · maccs 166 · avalon 512 · tt 1024 · rdkit 217 · scage1 512 · scage2 512 · mole 768`

- **scage1** = graph-level CLS embedding (`graph_embed.py`)
- **scage2** = atom-level mean-pooled embedding (`pool_atom_embed.py`)
- both from SCAGE `weights/pretrain/pretrain.pth`

Protocol I uses `maccs + avalon + rdkit + mole`; Protocol II uses `maccs + scage1 + mole`.

## Third-party encoders — licensing (must read before release)

SCAGE and MolE are **not vendored** here; the wrappers shell out to upstream.

- **SCAGE** — https://github.com/KazeDog/scage — **MIT**. Redistribution allowed
  with attribution.
- **MolE** — https://github.com/recursionpharma/mole_public — **CC BY-NC 4.0
  (non-commercial only)**. Do not redistribute its code, checkpoint, or (if it
  conflicts with the license) derived embeddings. Users obtain it upstream.

Cite both original papers.
