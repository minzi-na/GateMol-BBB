"""GateMol-BBB — Protocol I final architecture.

Feature combination: MACCS + Avalon + RDKit + MolE (seq_len = 4 modality tokens).
Verbatim model definition from the AutoResearch milestone
``07_phase2_hpo_final`` (commit ``88fdf45``): iter90 attention-pooling
architecture with Phase-2 HPO applied.

Only the ``nn.Module`` definitions are kept here; the original training/eval
harness (``bbb_prepare`` data loading, seed loop, artifact I/O) lives in
``gatemol_bbb.train`` / ``gatemol_bbb.evaluate``.

The architecture is fixed. The exact tuned hyperparameters for the released
weights (dropout, lr, weight_decay from Phase-2 Optuna) are stored per seed in
each artifact's ``config.json`` — load those when reconstructing the model,
not the class defaults below.
"""

import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (no mean centering)."""

    def __init__(self, d, eps=1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.scale


class SpatialGatingUnit(nn.Module):
    """Attention-based SGU with pre-norm on u and v + learned temperature.

    Self-attention replaces the Conv1d spatial projection. The attention
    diagonal is masked so mixing is cross-modal only (iter57), with a learnable
    log-temperature and dropout on the attention weights for regularization.
    """

    def __init__(self, d_ffn, seq_len, attn_drop=0.1):
        super().__init__()
        self.norm_v = nn.LayerNorm(d_ffn)   # normalize v (gate computation)
        self.norm_u = nn.LayerNorm(d_ffn)   # normalize u (input signal)
        self.d_ffn = d_ffn
        # Learnable log-temperature: init so scale ~= 1/sqrt(d_ffn) at start
        self.log_temp = nn.Parameter(torch.tensor(-0.5 * math.log(d_ffn)))
        self.attn_drop = nn.Dropout(attn_drop)
        # Self-attention Q, K, V projections for the v gate
        self.q_proj = nn.Linear(d_ffn, d_ffn)
        self.k_proj = nn.Linear(d_ffn, d_ffn)
        self.v_proj = nn.Linear(d_ffn, d_ffn)
        # Init near-identity for stable start
        nn.init.eye_(self.v_proj.weight)
        nn.init.zeros_(self.v_proj.bias)

    def forward(self, x):
        u, v = x.chunk(2, dim=-1)        # (B, seq_len, d_ffn) each
        u = self.norm_u(u)               # normalize u before gating
        v = self.norm_v(v)               # normalize v before attention
        Q = self.q_proj(v)
        K = self.k_proj(v)
        V = self.v_proj(v)
        scale = self.log_temp.exp()      # learned scalar temperature
        scores = Q @ K.transpose(-1, -2) * scale   # (B, seq_len, seq_len)
        # Mask diagonal to force cross-modal attention only (no self-attention)
        seq_len = scores.size(-1)
        diag_mask = torch.eye(seq_len, device=scores.device, dtype=torch.bool)
        scores = scores.masked_fill(diag_mask.unsqueeze(0), float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)
        v_out = attn @ V
        return u * v_out


class gMLPBlock(nn.Module):
    """gMLP block with optional stochastic depth regularization.

    During training the block output is dropped with probability ``drop_prob``
    (residual path always preserved); at inference the full block runs.
    """

    def __init__(self, d_model, d_ffn, seq_len, drop_prob=0.0):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.channel_proj1 = nn.Linear(d_model, d_ffn * 2)
        self.channel_proj2 = nn.Linear(d_ffn, d_model)
        self.sgu = SpatialGatingUnit(d_ffn, seq_len)
        self.drop_prob = drop_prob

    def forward(self, x):
        residual = x
        if self.training and self.drop_prob > 0.0:
            keep_prob = 1.0 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # (B, 1, 1)
            mask = torch.rand(shape, device=x.device) < keep_prob
            if not mask.any():
                return residual
        x = self.norm(x)
        x = F.silu(self.channel_proj1(x))
        x = self.sgu(x)
        x = self.channel_proj2(x)
        if self.training and self.drop_prob > 0.0:
            x = x * mask / keep_prob
        return x + residual


class gMLP(nn.Module):
    def __init__(self, d_model=512, d_ffn=1048, seq_len=4, num_layers=4,
                 stochastic_depth_rate=0.1):
        super().__init__()
        # Linear schedule: block 0 gets 0, block (num_layers-1) gets max rate
        drop_probs = [stochastic_depth_rate * i / max(num_layers - 1, 1)
                      for i in range(num_layers)]
        self.model = nn.Sequential(
            *[gMLPBlock(d_model, d_ffn, seq_len, drop_prob=dp)
              for dp in drop_probs]
        )

    def forward(self, x):
        return self.model(x)


class MultiModalGMLPFromFlat(nn.Module):
    """gMLP over a flat concatenation of per-modality feature blocks.

    ``mod_dims`` is an ordered mapping ``{modality_name: input_dim}``; the input
    tensor ``x`` is a flat concat of those blocks in the same order. Each block
    is projected to ``d_model``, stacked into ``seq_len`` tokens, mixed by the
    gMLP backbone, combined with a learned skip, and attention-pooled to a
    single vector for the binary head.
    """

    def __init__(self, mod_dims: OrderedDict,
                 d_model=512, d_ffn=1048, depth=4,
                 dropout=0.2, use_gated_pool=True,
                 stochastic_depth_rate=0.05):
        super().__init__()
        self.mod_names = list(mod_dims.keys())
        self.mod_dims = [mod_dims[n] for n in self.mod_names]
        self.seq_len = len(self.mod_names)
        self.use_gated_pool = use_gated_pool

        self.proj = nn.ModuleDict({
            name: nn.Linear(in_dim, d_model)
            for name, in_dim in zip(self.mod_names, self.mod_dims)
        })
        self.backbone = gMLP(
            d_model=d_model, d_ffn=d_ffn,
            seq_len=self.seq_len, num_layers=depth,
            stochastic_depth_rate=stochastic_depth_rate,
        )
        self.norm = nn.LayerNorm(d_model)
        if use_gated_pool:
            # Attention pooling: single input-dependent query (iter90, Phase 1 best)
            self.pool_query = nn.Parameter(torch.zeros(d_model))
        # Learnable gate for input skip: z=0 -> pure backbone, z=1 -> full skip
        self.skip_gate = nn.Parameter(torch.zeros(1))
        self.head = nn.Linear(d_model, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        chunks = torch.split(x, self.mod_dims, dim=1)
        tokens = [self.proj[name](chunk)
                  for name, chunk in zip(self.mod_names, chunks)]
        X0 = torch.stack(tokens, dim=1)         # (B, seq_len, d_model) — pre-backbone
        X = self.backbone(X0)
        # Skip connection: learned convex combination with pre-backbone tokens
        gate = torch.sigmoid(self.skip_gate)
        X = (1.0 - gate) * X + gate * X0
        if self.use_gated_pool:
            scores = (X @ self.pool_query) / (X.shape[-1] ** 0.5)  # (B, seq_len)
            w = torch.softmax(scores, dim=-1)
            Xp = (X * w.unsqueeze(-1)).sum(dim=1)                  # (B, d_model)
        else:
            Xp = X.mean(dim=1)
        Xp = self.drop(self.norm(Xp))
        return self.head(Xp).squeeze(-1)
