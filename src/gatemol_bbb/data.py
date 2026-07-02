"""Data utilities for GateMol-BBB: scaffold splitting and feature-combo assembly.

Works on the ``.npz`` feature matrices produced by
:func:`gatemol_bbb.features.assemble_features.build_feature_npz`, which store
``smiles``, ``labels``, and one array per modality.

Scaffold split matches the study's ``prepare.py`` exactly: group by Murcko
scaffold, sort groups by size (descending), then greedily fill train / val
(/ test) up to the size caps.
"""

import random
from collections import OrderedDict

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.preprocessing import StandardScaler


def set_seed(seed: int):
    """Deterministic seeding for reproducible splits and training."""
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def murcko_scaffold(smi: str) -> str:
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m)) if m else ""


def scaffold_split(smiles_list, seed: int, train_ratio=0.8, val_ratio=0.0,
                   mode="scaffold"):
    """Return ``(train_idx, val_idx, test_idx)`` index lists.

    With ``val_ratio == 0`` (Protocol II, 8:2) all non-train indices go to val
    and ``test_idx`` is empty. With ``val_ratio > 0`` (Protocol I, 8:1:1) the
    remainder after train+val goes to test. ``mode`` is ``'scaffold'`` (groups
    sorted largest-first — deterministic) or ``'random_scaffold'`` (shuffled).
    """
    set_seed(seed)
    scaffolds = [murcko_scaffold(s) for s in smiles_list]
    groups = {}
    for i, sc in enumerate(scaffolds):
        groups.setdefault(sc, []).append(i)
    groups = list(groups.values())

    if mode == "scaffold":
        groups = sorted(groups, key=len, reverse=True)
    elif mode == "random_scaffold":
        random.Random(seed).shuffle(groups)
    else:
        raise ValueError("mode must be 'scaffold' or 'random_scaffold'")

    n = len(smiles_list)
    train_cap = int(round(train_ratio * n))
    val_cap = int(round(val_ratio * n))

    train_idx, val_idx, test_idx = [], [], []
    for g in groups:
        if len(train_idx) + len(g) <= train_cap:
            train_idx += g
        elif val_ratio == 0.0 or len(val_idx) + len(g) <= val_cap:
            val_idx += g
        else:
            test_idx += g
    return train_idx, val_idx, test_idx


def select_combo(npz, combo):
    """Assemble the flat feature matrix for a modality ``combo`` from an npz.

    Returns ``(X, y, smiles, mod_dims)`` where ``X`` concatenates the chosen
    modality blocks in ``combo`` order and ``mod_dims`` is the
    ``OrderedDict{name: dim}`` needed to build the model.
    """
    blocks = [np.asarray(npz[t], dtype=np.float32) for t in combo]
    X = np.concatenate(blocks, axis=1)
    y = np.asarray(npz["labels"], dtype=np.float32)
    smiles = np.asarray(npz["smiles"]).astype(str)
    mod_dims = OrderedDict((t, blocks[i].shape[1]) for i, t in enumerate(combo))
    return X, y, smiles, mod_dims


def _rdkit_slice(combo, mod_dims):
    if "rdkit" not in combo:
        return None, None
    offset = 0
    for t in combo:
        if t == "rdkit":
            return offset, offset + mod_dims[t]
        offset += mod_dims[t]
    return None, None


def fit_rdkit_scaler(X_train, combo, mod_dims):
    """Fit a StandardScaler on the rdkit block of the training matrix (or None)."""
    rd_start, rd_end = _rdkit_slice(combo, mod_dims)
    if rd_start is None:
        return None, (None, None)
    scaler = StandardScaler().fit(X_train[:, rd_start:rd_end])
    return scaler, (rd_start, rd_end)


def apply_rdkit_scaler(X, scaler, rd_span):
    """Return a copy of X with the rdkit block standardized (no-op if scaler is None)."""
    rd_start, rd_end = rd_span
    if scaler is None or rd_start is None:
        return X
    X = X.copy()
    X[:, rd_start:rd_end] = scaler.transform(X[:, rd_start:rd_end])
    return X
