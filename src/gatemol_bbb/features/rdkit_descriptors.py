"""RDKit-based molecular featurization for GateMol-BBB.

Self-contained; the only dependency is RDKit. These functions reproduce the
fingerprint / descriptor blocks used in the study's ``prepare.py`` pipelines.

Modality dimensions (as used in the paper):
    ecfp   1024   Morgan (radius=2)
    maccs   166   MACCS keys with the always-zero bit 0 dropped
    avalon  512   Avalon fingerprint
    tt     1024   hashed topological torsion
    rdkit   217   RDKit 2D descriptors  (RDKit 2024.9.6 -> 217; version-sensitive)

Protocol I uses: maccs + avalon + rdkit + mole
Protocol II uses: maccs + scage1 + mole
(SCAGE / MolE embeddings come from separate pretrained encoders — see
``scage_embed.py`` and ``mole_embed.py``.)

IMPORTANT: the RDKit descriptor count depends on the RDKit version. This study
used RDKit 2024.9.6 (217 descriptors). Pin the environment exactly, or the
``rdkit`` block will misalign. See ``docs/reproducibility.md``.
"""

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys, rdMolDescriptors, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors


def canon_smiles(smi: str):
    """Return the canonical SMILES, or None if RDKit cannot parse it."""
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m, canonical=True) if m else None


def _to_numpy_bitvect(bv, n_bits=None, drop_first=False):
    if n_bits is None:
        n_bits = bv.GetNumBits()
    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(bv, arr)
    if drop_first:
        arr = arr[1:]
    return arr


def get_ecfp(mol, radius=2, nbits=1024):
    return _to_numpy_bitvect(
        AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits), n_bits=nbits)


def get_maccs(mol):
    # MACCS bit 0 is always zero by definition; drop it -> 166 dims.
    bv = MACCSkeys.GenMACCSKeys(mol)
    return _to_numpy_bitvect(bv, n_bits=bv.GetNumBits(), drop_first=True)


def get_avalon(mol, nbits=512):
    from rdkit.Avalon import pyAvalonTools
    return _to_numpy_bitvect(pyAvalonTools.GetAvalonFP(mol, nbits), n_bits=nbits)


def get_tt(mol, nbits=1024):
    bv = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=nbits)
    return _to_numpy_bitvect(bv, n_bits=nbits)


def get_rdkit_desc(mol):
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(
        [d[0] for d in Descriptors._descList])
    try:
        descs = np.array(calc.CalcDescriptors(mol), dtype=np.float32)
    except Exception:
        descs = np.zeros(len(Descriptors._descList), dtype=np.float32)
    return np.nan_to_num(descs, nan=0.0, posinf=0.0, neginf=0.0)


def rdkit_descriptor_length():
    """Number of RDKit 2D descriptors for the installed RDKit version."""
    return len(Descriptors._descList)


# Dispatch table for the RDKit-computable modalities.
_FEATURIZERS = {
    "ecfp": get_ecfp,
    "maccs": get_maccs,
    "avalon": get_avalon,
    "tt": get_tt,
    "rdkit": get_rdkit_desc,
}


def featurize_smiles(smiles, fp_types):
    """Compute the requested RDKit modalities for one SMILES.

    Returns an ``OrderedDict {fp_type: np.ndarray}`` in ``fp_types`` order, or
    ``None`` if the SMILES cannot be parsed. Only RDKit-computable modalities
    are accepted here; ``scage*`` / ``mole`` must be supplied by their encoders.
    """
    from collections import OrderedDict

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    out = OrderedDict()
    for t in fp_types:
        if t not in _FEATURIZERS:
            raise ValueError(
                f"'{t}' is not RDKit-computable; provide it via scage_embed / mole_embed.")
        out[t] = _FEATURIZERS[t](mol)
    return out
