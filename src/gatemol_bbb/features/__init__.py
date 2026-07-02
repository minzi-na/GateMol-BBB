"""Feature / embedding extraction for GateMol-BBB.

- rdkit_descriptors: ecfp / maccs / avalon / tt / rdkit, computed from SMILES.
- scage_embed:       scage1 (graph CLS) + scage2 (atom mean-pool) via upstream SCAGE.
- mole_embed:        mole (768-d) via upstream MolE.
- assemble_features: merge all modalities into one .npz feature matrix.
"""
