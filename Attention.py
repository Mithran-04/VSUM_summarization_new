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
    """
    Learns frame importance from:
        1. Forward + backward SSM step sizes (Delta)
        2. Forward + backward hidden-state changes

    The two signals are learned separately and then fused.
    """

    def __init__(self, d_inner=1024, d_state=16):
        super().__init__()

        # ---------------------------------------------------------
        # 1. Learned Delta representation
        #
        # Input:
        #   delta_fw: [B, T, 1024]
        #   delta_bw: [B, T, 1024]
        #
        # Concatenated:
        #   [B, T, 2048]
        #
        # Output:
        #   [B, T, 1]
        # ---------------------------------------------------------
        self.delta_encoder = nn.Sequential(
            nn.Linear(d_inner * 2, 128),
            nn.SiLU(),
            nn.Linear(128, 1)
        )

        # ---------------------------------------------------------
        # 2. State-change representation
        #
        # Input:
        #   forward state change magnitude
        #   backward state change magnitude
        #
        # [B, T, 2] -> [B, T, 1]
        # ---------------------------------------------------------
        self.state_encoder = nn.Sequential(
            nn.Linear(2, 16),
            nn.SiLU(),
            nn.Linear(16, 1)
        )

        # ---------------------------------------------------------
        # 3. Learn how Delta importance and state-change importance
        #    should be combined.
        #
        # [B, T, 2] -> [B, T, 1]
        # ---------------------------------------------------------
        self.fusion = nn.Sequential(
            nn.Linear(2, 16),
            nn.SiLU(),
            nn.Linear(16, 1)
        )

    def forward(
        self,
        delta_fw,
        delta_bw,
        h_fw,
        h_bw
    ):
        """
        Parameters
        ----------
        delta_fw : [B, T, D_inner]
        delta_bw : [B, T, D_inner]

        h_fw : [B, T, D_inner, N]
        h_bw : [B, T, D_inner, N]

        Returns
        -------
        scores       : [B, T, 1]
        delta_score  : [B, T, 1]
        state_score  : [B, T, 1]
        delta_t      : [B, T, 1]
        d_t          : [B, T, 1]
        """

        # =========================================================
        # 1. LEARNED DELTA REPRESENTATION
        # =========================================================

        # Instead of averaging 1024 forward + 1024 backward
        # channels, preserve all Delta information.
        delta_features = torch.cat(
            [delta_fw, delta_bw],
            dim=-1
        )
        # [B, T, 2048]

        delta_t = self.delta_encoder(
            delta_features
        )
        # [B, T, 1]

        # Normalize Delta importance across the temporal dimension.
        mean_delta = delta_t.mean(
            dim=1,
            keepdim=True
        )

        std_delta = delta_t.std(
            dim=1,
            keepdim=True
        ) + 1e-6

        delta_hat = (
            delta_t - mean_delta
        ) / std_delta

        delta_score = torch.sigmoid(
            delta_hat
        )
        # [B, T, 1]

        # =========================================================
        # 2. HIDDEN STATE CHANGE
        # =========================================================

        # Forward hidden-state difference
        dh_fw = (
            h_fw[:, 1:] -
            h_fw[:, :-1]
        )

        # Pad first timestep
        dh_fw = torch.cat(
            [dh_fw[:, :1], dh_fw],
            dim=1
        )

        # Backward hidden-state difference
        dh_bw = (
            h_bw[:, 1:] -
            h_bw[:, :-1]
        )

        # Pad first timestep
        dh_bw = torch.cat(
            [dh_bw[:, :1], dh_bw],
            dim=1
        )

        # Magnitude of forward state transition
        norm_fw = torch.norm(
            dh_fw,
            dim=(-2, -1)
        )
        # [B, T]

        # Magnitude of backward state transition
        norm_bw = torch.norm(
            dh_bw,
            dim=(-2, -1)
        )
        # [B, T]

        # Keep this quantity for diagnostics
        d_t = 0.5 * (
            norm_fw + norm_bw
        ).unsqueeze(-1)
        # [B, T, 1]

        # Build two-dimensional state feature
        state_features = torch.stack(
            [norm_fw, norm_bw],
            dim=-1
        )
        # [B, T, 2]

        state_raw = self.state_encoder(
            state_features
        )
        # [B, T, 1]

        state_score = torch.sigmoid(
            state_raw
        )
        # [B, T, 1]

        # =========================================================
        # 3. LEARNED FUSION
        # =========================================================

        fusion_features = torch.cat(
            [
                delta_score,
                state_score
            ],
            dim=-1
        )
        # [B, T, 2]

        scores = torch.sigmoid(
            self.fusion(
                fusion_features
            )
        )
        # [B, T, 1]

        return (
            scores,
            delta_score,
            state_score,
            delta_t,
            d_t
        )