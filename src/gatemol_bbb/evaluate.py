"""Evaluation metrics and aggregation for GateMol-BBB.

Primary reporting is **per-seed mean ± std**: compute metrics per seed, then
average across seeds. **Soft-voting** (average the per-seed probabilities, then
compute metrics once) is reported as a supplement. Classification threshold is
``prob > 0.5`` throughout. See ``docs/reproducibility.md`` for the rationale.
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score, matthews_corrcoef,
)


def binary_metrics(y_true, y_prob, threshold=0.5):
    """Full metric dict for one probability vector."""
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    y_pred = (y_prob > threshold).astype(int)
    has_both = len(set(y_true.tolist())) > 1

    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 4:
        tn, fp, _, _ = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        specificity = 0.0

    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if has_both else 0.0,
        "auprc": float(average_precision_score(y_true, y_prob)) if has_both else 0.0,
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
    }


def predict_probs(model, X, device=None, batch_size=256):
    """Sigmoid probabilities for a feature matrix X (numpy or tensor)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    X = torch.as_tensor(np.asarray(X, dtype=np.float32))
    probs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = X[i:i + batch_size].to(device)
            probs.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(probs) if probs else np.array([])


def per_seed_summary(seed_metric_dicts):
    """Aggregate a list of per-seed metric dicts into mean/std per metric."""
    keys = seed_metric_dicts[0].keys()
    out = {}
    for k in keys:
        vals = np.array([m[k] for m in seed_metric_dicts], dtype=float)
        out[k] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=0))}
    return out


def soft_vote_metrics(seed_prob_arrays, y_true, threshold=0.5):
    """Average per-seed probabilities (soft-voting ensemble), then score once."""
    ensemble = np.mean(np.stack(seed_prob_arrays, axis=0), axis=0)
    return binary_metrics(y_true, ensemble, threshold=threshold), ensemble


def format_summary(summary, metrics=("roc_auc", "mcc", "f1", "accuracy", "auprc")):
    """Human-readable 'mean±std' string for the headline metrics."""
    return "  ".join(
        f"{k}={summary[k]['mean']:.4f}±{summary[k]['std']:.4f}"
        for k in metrics if k in summary
    )
