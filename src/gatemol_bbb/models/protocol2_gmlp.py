"""GateMol-BBB — Protocol II final architecture.

Feature combination: MACCS + SCAGE1 + MolE (seq_len = 3 modality tokens).
Verbatim model definition from the AutoResearch milestone
``07_iter196-198_singlehead_attn_biased_pool`` (commit ``2e8cbdc``): the Phase-1
final architecture, which Phase-2 HPO did not beat, so it is the final model.

Only the ``nn.Module`` definitions are kept here; the original training/eval
harness (seed loop, ``build_and_train`` public API, artifact I/O) lives in
``gatemol_bbb.train`` / ``gatemol_bbb.evaluate``.

BASE_CONFIG for the released weights:
    d_model=512, d_ffn=1048, depth=4, dropout=0.2, use_gated_pool=True,
    lr=1e-4, weight_decay=1e-5, num_epochs=50, patience=10, batch_size=128,
    es_metric="val_auc".
Note the four-way pool combines gated + mean + max + attention with learnable
``pool_logits`` initialized to ``[0, 0, 0, 0.6]`` (single attention head).
"""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialGatingUnit(nn.Module):
    """Conv1d spatial-gating unit with a learnable identity/mix gate."""

    def __init__(self, d_ffn, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(d_ffn)
        self.spatial_proj = nn.Conv1d(seq_len, seq_len, kernel_size=1)
        nn.init.constant_(self.spatial_proj.bias, 1.0)
        self.gate_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v)
        s = self.gate_scale.exp()
        v = s * self.spatial_proj(v) + (1 - s) * v
        return u * v


class gMLPBlock(nn.Module):
    """gMLP block with stochastic depth (drop_path)."""

    def __init__(self, d_model, d_ffn, seq_len, drop_path=0.025):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.channel_proj1 = nn.Linear(d_model, d_ffn * 2)
        self.channel_proj2 = nn.Linear(d_ffn, d_model)
        self.sgu = SpatialGatingUnit(d_ffn, seq_len)
        self.drop_path = drop_path

    def forward(self, x):
        residual = x
        if self.training and self.drop_path > 0 and torch.rand(1).item() < self.drop_path:
            return residual
        x = self.norm(x)
        x = F.gelu(self.channel_proj1(x))
        x = self.sgu(x)
        x = self.channel_proj2(x)
        return x + residual


class gMLP(nn.Module):
    def __init__(self, d_model=256, d_ffn=512, seq_len=256, num_layers=6):
        super().__init__()
        self.model = nn.Sequential(
            *[gMLPBlock(d_model, d_ffn, seq_len) for _ in range(num_layers)]
        )

    def forward(self, x):
        return self.model(x)


class MultiModalGMLPFromFlat(nn.Module):
    """gMLP over a flat concatenation of per-modality feature blocks.

    Adds modality dropout on the input tokens and a four-way pooling head
    (gated + mean + max + single-head attention) mixed by learnable
    ``pool_logits``, followed by a 2-layer SiLU MLP classification head.
    """

    def __init__(self, mod_dims: OrderedDict, d_model=512, d_ffn=1024,
                 depth=4, dropout=0.2, use_gated_pool=True):
        super().__init__()
        self.mod_names = list(mod_dims.keys())
        self.mod_dims = [mod_dims[n] for n in self.mod_names]
        self.seq_len = len(self.mod_names)
        self.use_gated_pool = use_gated_pool

        self.proj = nn.ModuleDict({
            name: nn.Linear(in_dim, d_model)
            for name, in_dim in zip(self.mod_names, self.mod_dims)
        })
        self.backbone = gMLP(seq_len=self.seq_len, d_model=d_model,
                             d_ffn=d_ffn, num_layers=depth)
        self.final_norm = nn.LayerNorm(d_model)
        self.norm = nn.LayerNorm(d_model)
        if use_gated_pool:
            self.alpha = nn.Parameter(torch.zeros(self.seq_len))
            self.n_attn_heads = 1
            self.pool_query = nn.Parameter(
                torch.zeros(self.n_attn_heads, d_model // self.n_attn_heads))
            self.pool_logits = nn.Parameter(torch.tensor([0.0, 0.0, 0.0, 0.6]))
            self.gated_norm = nn.LayerNorm(d_model)
            self.mean_norm = nn.LayerNorm(d_model)
            self.max_norm = nn.LayerNorm(d_model)
            self.attn_norm = nn.LayerNorm(d_model)
            self.attn_head_proj = nn.Linear(d_model, d_model)
            nn.init.eye_(self.attn_head_proj.weight)
            nn.init.zeros_(self.attn_head_proj.bias)
        self.head = nn.Sequential(
            nn.Dropout(0.08),
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, 1),
        )
        self.drop = nn.Dropout(dropout)
        self.mod_drop_p = 0.1

    def forward(self, x):
        chunks = torch.split(x, self.mod_dims, dim=1)
        tokens = [self.proj[name](chunk)
                  for name, chunk in zip(self.mod_names, chunks)]
        X = torch.stack(tokens, dim=1)
        if self.training and self.mod_drop_p > 0:
            B = X.shape[0]
            mask = (torch.rand(B, self.seq_len, device=X.device) > self.mod_drop_p).float()
            X = X * mask.unsqueeze(-1)
        X = self.backbone(X)
        X = self.final_norm(X)
        if self.use_gated_pool:
            w = torch.softmax(self.alpha, dim=0)
            gated = (X * w.view(1, -1, 1)).sum(dim=1)
            mean = X.mean(dim=1)
            max_pool = X.max(dim=1).values
            B, L, D = X.shape
            H = self.n_attn_heads
            X_h = X.view(B, L, H, D // H)
            attn_scores = (X_h * self.pool_query.view(1, 1, H, -1)).sum(dim=-1) / ((D // H) ** 0.5)
            attn_w = torch.softmax(attn_scores, dim=1)
            attn_pool = (X_h * attn_w.unsqueeze(-1)).sum(dim=1).reshape(B, D)
            gated = self.gated_norm(gated)
            mean = self.mean_norm(mean)
            max_pool = self.max_norm(max_pool)
            attn_pool = self.attn_norm(attn_pool)
            attn_pool = self.attn_head_proj(attn_pool)
            pw = torch.softmax(self.pool_logits, dim=0)
            Xp = pw[0] * gated + pw[1] * mean + pw[2] * max_pool + pw[3] * attn_pool
        else:
            Xp = X.mean(dim=1)
        Xp = self.drop(self.norm(Xp))
        return self.head(Xp).squeeze(-1)
