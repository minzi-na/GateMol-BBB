# Scripts

Shell wrappers to be added (thin CLIs over `src/gatemol_bbb`):

| Script | Status | Purpose |
|---|---|---|
| `extract_embeddings.sh` | added | one-shot scage1 + scage2 + mole encoder extraction (CSV→TXT→PKL→3 CSVs) |
| `train_protocol1.sh` | todo | train Protocol I (MACCS+Avalon+RDKit+MolE) over 10 seeds |
| `train_protocol2.sh` | todo | train Protocol II (MACCS+SCAGE1+MolE) over 10 seeds |
| `reproduce_holdout_eval.sh` | todo | per-seed + soft-vote evaluation on 888 / nn05 329 holdout |

RDKit modalities (ecfp/maccs/avalon/tt/rdkit) are computed in-process by
`gatemol_bbb.features.assemble_features`, not by a shell script.
