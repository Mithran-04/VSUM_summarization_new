# Decoder.py
import torch
import torch.nn as nn
from Attention import BiMambaSSM

class MambaStateSpaceDecoder(nn.Module):
    """Continuous ODE State-Space Decoder for feature reconstruction."""
    def __init__(self, hidden_size=512, d_state=16):
        super().__init__()
        self.mamba_dec = BiMambaSSM(d_model=hidden_size, d_state=d_state)
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, weighted_features):
        reconstructed, _, _, _, _ = self.mamba_dec(weighted_features)
        reconstructed = self.norm(reconstructed)
        return self.output_proj(reconstructed)