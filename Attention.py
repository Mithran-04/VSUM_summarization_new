# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNProjection(nn.Module):
    """Projects 1024-D CNN visual features into hidden dimension D."""
    def __init__(self, input_size=1024, hidden_size=512, dropout=0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.projection(x)


class SemanticProjection(nn.Module):
    """Projects 768-D SigLIP2 semantic features into hidden dimension D."""
    def __init__(self, semantic_size=768, hidden_size=512, dropout=0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(semantic_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.projection(x)


class CrossModalFusion(nn.Module):
    """Fuses projected visual and semantic representations: [B, T, D]."""
    def __init__(self, hidden_size=512, dropout=0.1):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout)
        )

    def forward(self, visual_features, semantic_features):
        combined = torch.cat([visual_features, semantic_features], dim=-1)
        return self.fusion(combined)


class BiMambaSSM(nn.Module):
    """
    Bi-directional Mamba selective state-space block.
    Used as the sequence model of the selector (replaces the Bi-LSTM)
    and of the generator/decoder (replaces the Transformer).
    """
    def __init__(self, d_model=512, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = expand * d_model
        self.d_state = d_state

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1
        )

        self.x_proj_fw = nn.Linear(self.d_inner, self.d_inner + 2 * d_state, bias=False)
        self.x_proj_bw = nn.Linear(self.d_inner, self.d_inner + 2 * d_state, bias=False)

        self.dt_proj_fw = nn.Linear(self.d_inner, self.d_inner, bias=True)
        self.dt_proj_bw = nn.Linear(self.d_inner, self.d_inner, bias=True)
        for dt_proj in (self.dt_proj_fw, self.dt_proj_bw):
            dt = torch.exp(torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
            with torch.no_grad():
                dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))   # inverse softplus

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def _ssm_recurrence(self, x, x_proj, dt_proj):
        """Discretization (A: ZOH, B: Euler, as in Mamba) and recurrence along T."""
        B, T, D_in = x.shape
        N = self.d_state
        A = -torch.exp(self.A_log.float())  # [D_in, N]

        ssm_param = x_proj(x)
        delta_logits, B_param, C_param = torch.split(ssm_param, [D_in, N, N], dim=-1)

        delta = F.softplus(dt_proj(delta_logits))                      # [B,T,D_in]
        A_bar = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))   # [B,T,D_in,N]
        B_bar = delta.unsqueeze(-1) * B_param.unsqueeze(2)                     # [B,T,D_in,N]

        h = torch.zeros(B, D_in, N, device=x.device, dtype=x.dtype)
        h_trajectory = []
        for t in range(T):
            h = A_bar[:, t] * h + B_bar[:, t] * x[:, t].unsqueeze(-1)
            h_trajectory.append(h)

        h_stack = torch.stack(h_trajectory, dim=1)                     # [B,T,D_in,N]
        y = torch.einsum('btdn,btn->btd', h_stack, C_param) + x * self.D
        return y, delta, h_stack

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(0)

        B, T, _ = x.shape
        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)

        x_inner = x_inner.transpose(1, 2)
        x_inner = self.conv1d(x_inner)[:, :, :T].transpose(1, 2)
        x_inner = F.silu(x_inner)

        y_fw, delta_fw, h_fw = self._ssm_recurrence(x_inner, self.x_proj_fw, self.dt_proj_fw)

        x_inner_bw = torch.flip(x_inner, dims=[1])
        y_bw_rev, delta_bw_rev, h_bw_rev = self._ssm_recurrence(x_inner_bw, self.x_proj_bw, self.dt_proj_bw)

        y_bw = torch.flip(y_bw_rev, dims=[1])
        delta_bw = torch.flip(delta_bw_rev, dims=[1])
        h_bw = torch.flip(h_bw_rev, dims=[1])

        y_out = (y_fw + y_bw) * F.silu(z)
        out = self.out_proj(y_out)

        return out, delta_fw, delta_bw, h_fw, h_bw


class FSSAScorer(nn.Module):
    """
    Frame-level Semantic-alignment scorer (old method), with the Bi-LSTM replaced by BiMamba:

        S_t = alpha * cos(X_t, T_t) + (1 - alpha) * h_t,   alpha = sigmoid(alpha_logit) in [0, 1]

    h_t is a single score head on the BiMamba encoder output, squashed to [0, 1].
    There is deliberately NO z-score here, so the sparsity loss can control the score mean.
    """
    def __init__(self, d_model=512, summary_rate=0.15):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(d_model, 128), nn.SiLU(), nn.Linear(128, 1))
        # start h_t near the target sparsity
        self.bias = nn.Parameter(torch.tensor(math.log(summary_rate / (1 - summary_rate))))
        # alpha ~= 0.18 at init, so alpha * mean(cos) stays below the sparsity target
        self.alpha_logit = nn.Parameter(torch.tensor(-1.5))

    def forward(self, temporal, cos_sim):
        h_t = torch.sigmoid(self.head(temporal) + self.bias)       # [B,T,1]
        alpha = torch.sigmoid(self.alpha_logit)
        # scores = alpha * cos_sim + (1 - alpha) * h_t               # [B,T,1]
        scores = h_t               # [B,T,1]
        return scores, h_t, alpha


class FeatureDiscriminator(nn.Module):
    """
    Discriminator of the old method: judges whether a feature sequence is the original
    (X, T) or the reconstructed (X_hat, T_hat). Input is the concatenated, per-modality
    L2-normalized features [B, T, cnn+sem]; output is one real/fake logit per frame [B, T, 1].
    It is only used during training and is NOT part of the Summarizer / checkpoint used at test time.
    """
    def __init__(self, in_size=1792, hidden_size=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        return self.net(x)