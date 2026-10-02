# Decoder.py
import torch
import torch.nn as nn
from Attention import BiMambaSSM

class MambaStateSpaceDecoder(nn.Module):
    def __init__(self, hidden_size=512, d_state=16, out_size=1792):
        super().__init__()
        self.mamba_dec = BiMambaSSM(d_model=hidden_size, d_state=d_state)
        self.output_proj = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        out, *_ = self.mamba_dec(x)
        return self.output_proj(out)          # no LayerNorm