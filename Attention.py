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
    """Fuses projected visual and semantic representations: [B, T, D_fused]."""
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
    Bi-Directional Mamba Selective State-Space Model Block with
    explicit Zero-Order Hold (ZOH) discretization mechanics and
    hidden trajectory interception.
    """
    def __init__(self, d_model=512, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = expand * d_model
        self.d_state = d_state

        # Input projections
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1
        )

        # SSM parameter projections (Forward & Backward)
        self.x_proj_fw = nn.Linear(self.d_inner, self.d_inner + 2 * d_state, bias=False)
        self.x_proj_bw = nn.Linear(self.d_inner, self.d_inner + 2 * d_state, bias=False)

        self.dt_proj_fw = nn.Linear(self.d_inner, self.d_inner, bias=True)
        self.dt_proj_bw = nn.Linear(self.d_inner, self.d_inner, bias=True)
        for dt_proj in (self.dt_proj_fw, self.dt_proj_bw):
            dt = torch.exp(torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
            with torch.no_grad():
                dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))   # inverse softplus

        # Continuous state matrix A initialization (S4D structured initialization)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)


    def _ssm_recurrence(self, x, x_proj, dt_proj):
        """Executes ZOH discretization and SSM recurrence along sequence length."""
        B, T, D_in = x.shape
        N = self.d_state
        A = -torch.exp(self.A_log.float())  # [D_in, N]

        # Parameter projection
        ssm_param = x_proj(x)  # [B, T, D_in + 2*N]
        delta_logits, B_param, C_param = torch.split(
            ssm_param, [D_in, N, N], dim=-1
        )  # Δ_t: [B, T, D_in], B_t: [B, T, N], C_t: [B, T, N]

        delta = F.softplus(dt_proj(delta_logits))  # [B, T, D_in] > 0

        # Discretization via Zero-Order Hold (ZOH)
        # Ā_t = exp(Δ_t ⊗ A) -> [B, T, D_in, N]
        delta_A = delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
        A_bar = torch.exp(delta_A)

        # B̄_t = (Δ_t ⊗ I) * B_t -> [B, T, D_in, N]
        B_bar = delta.unsqueeze(-1) * B_param.unsqueeze(2)

        # Recurrent state computation
        h = torch.zeros(B, D_in, N, device=x.device, dtype=x.dtype)
        h_trajectory = []

        for t in range(T):
            h = A_bar[:, t] * h + B_bar[:, t] * x[:, t].unsqueeze(-1)
            h_trajectory.append(h)

        h_stack = torch.stack(h_trajectory, dim=1)  # [B, T, D_in, N]

        # Output calculation: y_t = C_t * h_t
        y = torch.einsum('btdn,btn->btd', h_stack, C_param) + x * self.D
        return y, delta, h_stack

    def forward(self, x):
        # Guard against 2D inputs [T, D] -> reshape to 3D [1, T, D]
        if x.ndim == 2:
            x = x.unsqueeze(0)

        B, T, _ = x.shape
        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)

        # 1D Depthwise Convolution
        x_inner = x_inner.transpose(1, 2)
        x_inner = self.conv1d(x_inner)[:, :, :T].transpose(1, 2)
        x_inner = F.silu(x_inner)

        # Forward SSM Pass
        y_fw, delta_fw, h_fw = self._ssm_recurrence(
            x_inner, self.x_proj_fw, self.dt_proj_fw
        )

        # Backward SSM Pass (Reverse temporal sequence)
        x_inner_bw = torch.flip(x_inner, dims=[1])
        y_bw_rev, delta_bw_rev, h_bw_rev = self._ssm_recurrence(
            x_inner_bw, self.x_proj_bw, self.dt_proj_bw
        )

        y_bw = torch.flip(y_bw_rev, dims=[1])
        delta_bw = torch.flip(delta_bw_rev, dims=[1])
        h_bw = torch.flip(h_bw_rev, dims=[1])

        # Dual State Integration & Out Projection
        y_out = (y_fw + y_bw) * F.silu(z)
        out = self.out_proj(y_out)

        return out, delta_fw, delta_bw, h_fw, h_bw


class DualParameterInterceptorScorer(nn.Module):
    def __init__(self, d_inner=1024, d_state=16, summary_rate=0.15):
        super().__init__()
        self.delta_encoder = nn.Sequential(
            nn.Linear(d_inner * 2, 128), nn.SiLU(), nn.Linear(128, 1))
        self.state_encoder = nn.Sequential(
            nn.Linear(2, 16), nn.SiLU(), nn.Linear(16, 1))
        self.fusion = nn.Sequential(
            nn.Linear(2, 16), nn.SiLU(), nn.Linear(16, 1))
        self.scale = nn.Parameter(torch.tensor(2.0))
        self.bias = nn.Parameter(torch.tensor(math.log(summary_rate / (1 - summary_rate))))

    @staticmethod
    def _zscore(x):                       # x: [B, T, 1], normalize over time
        return (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True, unbiased=False) + 1e-6)

    @staticmethod
    def _state_norm(h):                   # h: [B, T, D, N] -> [B, T]
        dh = h[:, 1:] - h[:, :-1]
        dh = torch.cat([dh[:, :1], dh], dim=1)
        return torch.norm(dh, dim=(-2, -1))

    def forward(self, delta_fw, delta_bw, h_fw, h_bw):
        # Delta branch
        delta_t = self.delta_encoder(torch.cat([delta_fw, delta_bw], dim=-1))   # [B,T,1]
        delta_hat = self._zscore(delta_t)
        delta_score = torch.sigmoid(delta_hat)                                   # diagnostics

        # State-change branch (log + z-score so it can't saturate)
        norm_fw, norm_bw = self._state_norm(h_fw), self._state_norm(h_bw)
        d_t = 0.5 * (norm_fw + norm_bw).unsqueeze(-1)
        s = torch.log1p(torch.stack([norm_fw, norm_bw], dim=-1))                 # [B,T,2]
        s = (s - s.mean(1, keepdim=True)) / (s.std(1, keepdim=True, unbiased=False) + 1e-6)
        state_hat = self.state_encoder(s)                                        # [B,T,1]
        state_hat = self._zscore(state_hat)
        state_score = torch.sigmoid(state_hat)                                   # diagnostics

        # Fuse the z-scored signals directly (no stacked sigmoids)
        logit = self.fusion(torch.cat([delta_hat, state_hat], dim=-1))
        z = self._zscore(logit)
        scores = torch.sigmoid(self.scale * z + self.bias)                       # [B,T,1]
        return scores, delta_score, state_score, delta_t, d_t