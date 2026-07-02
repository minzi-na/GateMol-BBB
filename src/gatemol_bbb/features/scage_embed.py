"""SCAGE embedding extraction (thin wrapper — upstream code is NOT vendored).

SCAGE (self-conformation-aware Graph Transformer) is a third-party pretrained
molecular encoder. Two SCAGE-derived modalities are used by GateMol-BBB, both
512-dim and both from the same pretrained weights
(``weights/pretrain/pretrain.pth``), differing only in the pooling script:

  * ``scage1`` — graph-level CLS-token embedding, via ``graph_embed.py``
  * ``scage2`` — atom-level mean-pooled embedding, via ``pool_atom_embed.py``

The one-shot ``scripts/extract_embeddings.sh`` runs both (plus MolE) end to end.

Upstream:
    Repo:    https://github.com/KazeDog/scage
    License: MIT (c) 2024 Wei-Group, Jianbo Qiao
    Cite:    SCAGE — self-conformation-aware Graph Transformer for molecular
             property prediction. (add full citation)

Because SCAGE is a separate MIT-licensed project with large pretrained weights,
this repository does not copy its model code. To regenerate SCAGE embeddings:

    1. git clone https://github.com/KazeDog/scage && install its environment.
    2. Obtain the pretrained weights (see the SCAGE repo / paper).
    3. Run SCAGE's graph-embedding extraction on your SMILES to produce a CSV
       with a 'smiles' column plus 512 embedding columns.
    4. Point assemble_features.build_feature_npz(encoder_csvs=...) at the CSV.

The helper below shells out to a SCAGE checkout if you set SCAGE_REPO; otherwise
it documents the expected command. It intentionally has no hard dependency on
the SCAGE package so this repo installs without it.
"""

import os
import subprocess

# Which upstream script produces each SCAGE modality.
_SCAGE_SCRIPT = {
    "scage1": "graph_embed.py",      # graph-level CLS embedding
    "scage2": "pool_atom_embed.py",  # atom-level mean-pooled embedding
}


def extract_scage_csv(modality, input_pkl, output_csv,
                      weight_path="weights/pretrain/pretrain.pth",
                      scage_repo=None, python_exe="python", dry_run=False):
    """Invoke the upstream SCAGE extraction script for one modality.

    Parameters
    ----------
    modality : str
        Either ``'scage1'`` (graph CLS) or ``'scage2'`` (atom mean-pool).
    input_pkl : str
        Preprocessed SCAGE input (list-of-dict pickle) from ``prepare_data.py``.
    output_csv : str
        Destination CSV ('smiles' + 512 embedding columns).
    weight_path : str
        SCAGE pretrained weights; both modalities share the pretrain checkpoint.
    scage_repo : str, optional
        Path to a local SCAGE checkout. Defaults to env var ``SCAGE_REPO``.
    dry_run : bool
        If True, return the command without running it.
    """
    if modality not in _SCAGE_SCRIPT:
        raise ValueError(f"modality must be one of {list(_SCAGE_SCRIPT)}; got {modality!r}")
    scage_repo = scage_repo or os.environ.get("SCAGE_REPO")
    if not scage_repo:
        raise RuntimeError(
            "Set scage_repo= or the SCAGE_REPO env var to a local SCAGE checkout "
            "(https://github.com/KazeDog/scage). SCAGE code is not vendored here.")
    cmd = [
        python_exe, os.path.join(scage_repo, _SCAGE_SCRIPT[modality]),
        "--input_pkl", input_pkl,
        "--weight_path", weight_path,
        "--output_csv", output_csv,
    ]
    if dry_run:
        return cmd
    subprocess.run(cmd, cwd=scage_repo, check=True)
    return output_csv
