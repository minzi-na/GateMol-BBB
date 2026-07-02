"""Training entry point for GateMol-BBB (both protocols).

Each protocol reproduces its original AutoResearch training loop:

  Protocol I  — Adam, BCEWithLogits(pos_weight=0.08), early-stop on val loss.
                Model: protocol1_gmlp (maccs+avalon+rdkit+mole). 8:1:1 scaffold
                split (train/val used; test held out).
  Protocol II — AdamW with decoupled no-decay on norms/biases + weight EMA
                (decay 0.999), early-stop on val ROC-AUC. Model: protocol2_gmlp
                (maccs+scage1+mole). 8:2 scaffold split.

Trains one model per seed and saves weights + config (+ RDKit scaler) per seed,
matching the released-artifact layout. NOTE: the released Protocol I weights use
Phase-2 Optuna-tuned hyperparameters (dropout/lr/weight_decay) stored in each
artifact's config.json; the defaults here are the Phase-1-final settings.

Usage:
    python -m gatemol_bbb.train --protocol 1 --features features/protocol1.npz --out artifacts/p1
    python -m gatemol_bbb.train --protocol 2 --features features/protocol2.npz --out artifacts/p2
"""

import argparse
import json
import os
import pickle
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tud

from gatemol_bbb import data as gdata
from gatemol_bbb.evaluate import binary_metrics, predict_probs

SEEDS = [42, 100, 200, 300, 400, 500, 600, 700, 800, 900]

DEFAULT_COMBO = {
    1: ["maccs", "avalon", "rdkit", "mole"],
    2: ["maccs", "scage1", "mole"],
}

BASE_CONFIG = {
    "d_model": 512, "d_ffn": 1048, "depth": 4,
    "num_epochs": 50, "patience": 10, "batch_size": 128, "lr": 1e-4,
}


def build_model(protocol, mod_dims, dropout):
    if protocol == 1:
        from gatemol_bbb.models.protocol1_gmlp import MultiModalGMLPFromFlat
        return MultiModalGMLPFromFlat(
            mod_dims, d_model=BASE_CONFIG["d_model"], d_ffn=BASE_CONFIG["d_ffn"],
            depth=BASE_CONFIG["depth"], dropout=dropout, use_gated_pool=True,
            stochastic_depth_rate=0.05)
    from gatemol_bbb.models.protocol2_gmlp import MultiModalGMLPFromFlat
    return MultiModalGMLPFromFlat(
        mod_dims, d_model=BASE_CONFIG["d_model"], d_ffn=BASE_CONFIG["d_ffn"],
        depth=BASE_CONFIG["depth"], dropout=dropout, use_gated_pool=True)


def _loaders(X_tr, y_tr, X_val, y_val, bs):
    tr = tud.DataLoader(tud.TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
                        batch_size=bs, shuffle=True)
    va = tud.DataLoader(tud.TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
                        batch_size=bs, shuffle=False)
    return tr, va


def train_protocol1(model, tr, va, device):
    """Adam + BCEWithLogits(pos_weight=0.08), early-stop on val loss."""
    opt = optim.Adam(model.parameters(), lr=BASE_CONFIG["lr"], weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.08]).to(device))
    best_val, best_state, bad = float("inf"), None, 0
    for _ in range(BASE_CONFIG["num_epochs"]):
        model.train()
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        vloss, nb = 0.0, 0
        with torch.no_grad():
            for x, y in va:
                x, y = x.to(device), y.to(device)
                vloss += loss_fn(model(x), y).item(); nb += 1
        vloss /= max(nb, 1)
        if vloss < best_val:
            best_val, best_state, bad = vloss, deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= BASE_CONFIG["patience"]:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model


def train_protocol2(model, tr, va, device):
    """AdamW (no-decay on norms/biases) + weight EMA, early-stop on val AUC."""
    from sklearn.metrics import roc_auc_score
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower()
         else decay).append(p)
    opt = optim.AdamW([{"params": decay, "weight_decay": 1e-5},
                       {"params": no_decay, "weight_decay": 0.0}], lr=BASE_CONFIG["lr"])
    loss_fn = nn.BCEWithLogitsLoss()

    ema_decay = 0.999
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_score, best_state, bad = float("-inf"), None, 0
    for _ in range(BASE_CONFIG["num_epochs"]):
        model.train()
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    if v.dtype.is_floating_point:
                        ema[k].mul_(ema_decay).add_(v.detach(), alpha=1 - ema_decay)
                    else:
                        ema[k].copy_(v.detach())
        # Evaluate on EMA weights
        train_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(ema)
        model.eval()
        yt, yp = [], []
        with torch.no_grad():
            for x, y in va:
                yp.extend(torch.sigmoid(model(x.to(device))).cpu().numpy().tolist())
                yt.extend(y.numpy().tolist())
        model.load_state_dict(train_state)
        score = float(roc_auc_score(yt, yp)) if len(set(yt)) > 1 else 0.0
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().clone() for k, v in ema.items()}
            bad = 0
        else:
            bad += 1
            if bad >= BASE_CONFIG["patience"]:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model


def run(protocol, features_npz, out_dir, combo=None, seeds=SEEDS, dropout=None):
    combo = combo or DEFAULT_COMBO[protocol]
    dropout = dropout if dropout is not None else (0.1 if protocol == 1 else 0.2)
    train_ratio, val_ratio = (0.8, 0.1) if protocol == 1 else (0.8, 0.0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    npz = np.load(features_npz, allow_pickle=True)
    X, y, smiles, mod_dims = gdata.select_combo(npz, combo)
    os.makedirs(out_dir, exist_ok=True)

    for seed in seeds:
        tr_idx, va_idx, _ = gdata.scaffold_split(smiles, seed, train_ratio, val_ratio)
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[va_idx], y[va_idx]

        scaler, rd_span = gdata.fit_rdkit_scaler(X_tr, combo, mod_dims)
        X_tr = gdata.apply_rdkit_scaler(X_tr, scaler, rd_span)
        X_va = gdata.apply_rdkit_scaler(X_va, scaler, rd_span)

        gdata.set_seed(seed)
        model = build_model(protocol, mod_dims, dropout).to(device)
        tr, va = _loaders(X_tr, y_tr, X_va, y_va, BASE_CONFIG["batch_size"])
        trainer = train_protocol1 if protocol == 1 else train_protocol2
        model = trainer(model, tr, va, device)

        val_metrics = binary_metrics(y_va, predict_probs(model, X_va, device))
        seed_dir = os.path.join(out_dir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(seed_dir, "model.pth"))
        if scaler is not None:
            with open(os.path.join(seed_dir, "rdkit_scaler.pkl"), "wb") as f:
                pickle.dump(scaler, f)
        with open(os.path.join(seed_dir, "config.json"), "w") as f:
            json.dump({"protocol": protocol, "seed": seed, "combo": combo,
                       "mod_dims": {k: int(v) for k, v in mod_dims.items()},
                       "dropout": dropout, "base_config": BASE_CONFIG,
                       "val_metrics": val_metrics}, f, indent=2)
        print(f"[P{protocol}] seed={seed:>4d}  val_auc={val_metrics['roc_auc']:.4f}  "
              f"val_mcc={val_metrics['mcc']:.4f}")


def main():
    ap = argparse.ArgumentParser(description="Train GateMol-BBB (Protocol I or II)")
    ap.add_argument("--protocol", type=int, required=True, choices=[1, 2])
    ap.add_argument("--features", required=True, help="feature .npz from assemble_features")
    ap.add_argument("--out", required=True, help="output artifact dir")
    ap.add_argument("--combo", nargs="+", default=None, help="modality combo (default per protocol)")
    ap.add_argument("--dropout", type=float, default=None)
    args = ap.parse_args()
    run(args.protocol, args.features, args.out, combo=args.combo, dropout=args.dropout)


if __name__ == "__main__":
    main()
