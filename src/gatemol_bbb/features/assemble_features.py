"""Assemble the full multimodal feature matrix for GateMol-BBB.

This is the "all modalities in one place" step, ported from the study's
``prepare.py`` (``_build_subset_cache``). It does NOT compute the neural
embeddings from scratch — it combines:

  * RDKit-computable modalities (ecfp, maccs, avalon, tt, rdkit), computed here
    from SMILES via :mod:`gatemol_bbb.features.rdkit_descriptors`; and
  * pretrained-encoder embeddings (scage1, scage2, mole), which must be
    extracted first with :mod:`gatemol_bbb.features.scage_embed` /
    :mod:`gatemol_bbb.features.mole_embed` and supplied as per-SMILES CSVs.

Output: a ``.npz`` with ``smiles``, ``labels``, and one array per modality
(rows aligned). A model for a given feature combination then concatenates the
chosen modality blocks in order.

Full modality set and dimensions (RDKit 2024.9.6):
    ecfp 1024 | maccs 166 | avalon 512 | tt 1024 | rdkit 217
    scage1 512 | scage2 512 | mole 768
"""

import json

import numpy as np
import pandas as pd
from rdkit import Chem

from gatemol_bbb.features import rdkit_descriptors as rd

ALL_FP_TYPES = ["ecfp", "maccs", "avalon", "tt", "rdkit", "scage1", "scage2", "mole"]

FP_DIM = {
    "ecfp": 1024,
    "maccs": 166,
    "avalon": 512,
    "tt": 1024,
    "rdkit": rd.rdkit_descriptor_length(),  # 217 @ RDKit 2024.9.6
    "scage1": 512,
    "scage2": 512,
    "mole": 768,
}

# Modalities that come from external pretrained encoders (loaded, not computed).
ENCODER_MODALITIES = ("scage1", "scage2", "mole")


def load_embed_csv(path: str) -> dict:
    """Load a precomputed embedding CSV (a 'smiles' column + embedding columns).

    Keys are canonical SMILES so lookups align with the RDKit-computed rows.
    """
    df = pd.read_csv(path)
    embed_cols = [c for c in df.columns if c != "smiles"]
    out = {}
    for _, row in df.iterrows():
        cs = rd.canon_smiles(row["smiles"])
        if cs:
            out[cs] = row[embed_cols].to_numpy(dtype=np.float32, copy=False)
    return out


def build_feature_npz(label_csv, encoder_csvs, out_npz, label_col="p_np",
                      manifest_path=None):
    """Build and save the full 8-modality feature matrix.

    Parameters
    ----------
    label_csv : str
        CSV with a 'smiles' column and a binary label column (``label_col``).
        Labels may be {0,1} or {'BBB-','BBB+'}.
    encoder_csvs : dict
        ``{'scage1': path, 'scage2': path, 'mole': path}`` — precomputed
        embedding CSVs. Missing SMILES fall back to a zero vector (matching the
        original pipeline).
    out_npz : str
        Output ``.npz`` path.
    manifest_path : str, optional
        If given, write a JSON manifest (sizes, per-modality dims, failures).
    """
    df = pd.read_csv(label_csv)[["smiles", label_col]].copy()
    df["smiles"] = df["smiles"].apply(rd.canon_smiles)
    df = df.dropna(subset=["smiles"]).drop_duplicates(subset="smiles").reset_index(drop=True)
    df[label_col] = df[label_col].replace({"BBB-": 0, "BBB+": 1}).astype(int)

    embed = {m: load_embed_csv(encoder_csvs[m]) for m in ENCODER_MODALITIES}

    valid_smiles, valid_labels = [], []
    rows = {t: [] for t in ALL_FP_TYPES}
    failed = 0
    for _, row in df.iterrows():
        smi = row["smiles"]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            failed += 1
            continue
        try:
            feats = {
                "ecfp": rd.get_ecfp(mol),
                "maccs": rd.get_maccs(mol),
                "avalon": rd.get_avalon(mol),
                "tt": rd.get_tt(mol),
                "rdkit": rd.get_rdkit_desc(mol),
                "scage1": embed["scage1"].get(smi, np.zeros(FP_DIM["scage1"], dtype=np.float32)),
                "scage2": embed["scage2"].get(smi, np.zeros(FP_DIM["scage2"], dtype=np.float32)),
                "mole": embed["mole"].get(smi, np.zeros(FP_DIM["mole"], dtype=np.float32)),
            }
        except Exception:
            failed += 1
            continue
        valid_smiles.append(smi)
        valid_labels.append(int(row[label_col]))
        for t in ALL_FP_TYPES:
            rows[t].append(feats[t])

    labels = np.array(valid_labels, dtype=np.int64)
    feat_arrays = {t: np.stack(rows[t], axis=0) for t in ALL_FP_TYPES}
    np.savez(out_npz,
             smiles=np.array(valid_smiles, dtype=object),
             labels=labels,
             **feat_arrays)

    if manifest_path:
        manifest = {
            "size": len(valid_smiles),
            "failed": failed,
            "fp_dims": {t: int(feat_arrays[t].shape[1]) for t in ALL_FP_TYPES},
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    return out_npz
