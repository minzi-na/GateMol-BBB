"""MolE embedding extraction (thin wrapper — upstream code is NOT vendored).

MolE is a third-party pretrained molecular transformer. GateMol-BBB uses its
768-dim embedding as the ``mole`` modality.

Upstream:
    Repo:    https://github.com/recursionpharma/mole_public
    License: Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0)
    Cite:    MolE (Recursion Pharmaceuticals). (add full citation)

LICENSING NOTE (important): MolE is released under CC BY-NC 4.0, i.e.
**non-commercial use only**. Its code and pretrained checkpoint are therefore
NOT redistributed in this repository, and neither is any embedding CSV derived
from it if that would conflict with the license. Users must obtain MolE and its
checkpoint from the upstream repo and confirm their use complies with CC BY-NC.

To regenerate MolE embeddings, run the upstream package in its own environment:

    from mole import mole_predict
    emb = mole_predict.encode(smiles=clean_smiles,
                              pretrained_model=CKPT_PATH,
                              batch_size=32, num_workers=0)

then save a CSV with a 'smiles' column plus 768 embedding columns and point
assemble_features.build_feature_npz(encoder_csvs={'mole': ...}) at it. The
one-shot ``scripts/extract_embeddings.sh`` wraps this step.

The helper below shells out to a MolE checkout; it has no import-time
dependency on the MolE package so this repo installs without it.
"""

import os
import subprocess


def extract_mole_csv(input_csv, output_csv, ckpt, mole_repo=None,
                     python_exe=None, smiles_col="smiles", batch_size=32,
                     dry_run=False):
    """Invoke the upstream MolE embedding script to write an embedding CSV.

    Parameters
    ----------
    input_csv : str
        CSV with a SMILES column (``smiles_col``).
    output_csv : str
        Destination CSV ('smiles' + 768 embedding columns).
    ckpt : str
        Path to the MolE pretrained checkpoint (obtained from upstream).
    mole_repo : str, optional
        Path to a local mole_public checkout. Defaults to env var ``MOLE_DIR``.
    python_exe : str, optional
        Interpreter for the MolE environment. Defaults to env var
        ``MOLE_PYTHON`` or ``python``.
    dry_run : bool
        If True, return the command without running it.
    """
    mole_repo = mole_repo or os.environ.get("MOLE_DIR")
    if not mole_repo:
        raise RuntimeError(
            "Set mole_repo= or the MOLE_DIR env var to a local mole_public checkout "
            "(https://github.com/recursionpharma/mole_public). MolE is CC BY-NC 4.0 "
            "and is not vendored here.")
    python_exe = python_exe or os.environ.get("MOLE_PYTHON", "python")
    cmd = [
        python_exe, os.path.join(mole_repo, "generate_mole_embed.py"),
        "--input", input_csv,
        "--output", output_csv,
        "--ckpt", ckpt,
        "--smiles_col", smiles_col,
        "--batch_size", str(batch_size),
    ]
    if dry_run:
        return cmd
    subprocess.run(cmd, cwd=mole_repo, check=True)
    return output_csv
