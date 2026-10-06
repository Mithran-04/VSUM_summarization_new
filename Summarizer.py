# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from Attention import (
    CNNProjection,
    SemanticProjection,
    CrossModalFusion,
    BiMambaSSM,
    FSSAScorer
)
from layers.decoder import MambaStateSpaceDecoder


class Summarizer(nn.Module):
    """
    features -> projections -> fusion -> BiMamba encoder -> score head (h_t)
    S_t = alpha * cos(X_t, T_t) + (1 - alpha) * h_t
    soft re-weighting by S_t -> Mamba decoder -> reconstructed (X, T)
    """
    def __init__(self, cnn_size=1024, semantic_size=768, hidden_size=512,
                 d_state=16, dropout=0.1, summary_rate=0.15):
        super().__init__()
        self.summary_rate = summary_rate
        self.cnn_projection = CNNProjection(cnn_size, hidden_size, dropout)
        self.semantic_projection = SemanticProjection(semantic_size, hidden_size, dropout)
        self.cross_modal_fusion = CrossModalFusion(hidden_size, dropout)
        self.mamba_encoder = BiMambaSSM(d_model=hidden_size, d_state=d_state)
        self.scorer = FSSAScorer(d_model=hidden_size, summary_rate=summary_rate)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.decoder = MambaStateSpaceDecoder(hidden_size, d_state,
                                              out_size=cnn_size + semantic_size)

    def forward(self, cnn_features, semantic_features):
        if cnn_features.ndim == 2:
            cnn_features, semantic_features = cnn_features.unsqueeze(0), semantic_features.unsqueeze(0)

        cnn_proj = self.cnn_projection(cnn_features)
        sem_proj = self.semantic_projection(semantic_features)
        fused = self.cross_modal_fusion(cnn_proj, sem_proj)                  # [B,T,D]

        temporal, *_ = self.mamba_encoder(fused)                             # [B,T,D]

        # Cross-modal cosine in the shared projected space (detached: acts as a stable prior),
        # min-max normalised per video so it lives in [0, 1] like h_t.
        cos = F.cosine_similarity(cnn_proj.detach(), sem_proj.detach(), dim=-1).unsqueeze(-1)
        cos_min = cos.amin(dim=1, keepdim=True)
        cos_max = cos.amax(dim=1, keepdim=True)
        cos = (cos - cos_min) / (cos_max - cos_min + 1e-6)

        scores, h_t, alpha = self.scorer(temporal, cos)                      # [B,T,1]

        # Soft re-weighting (no hard top-k in the training path).
        # For the exact paper form use:  dec_in = fused * scores
        dec_in = fused * scores + self.mask_token * (1 - scores)
        recon = self.decoder(dec_in)                                         # [B,T,cnn+sem]

        target = torch.cat([F.normalize(cnn_features, dim=-1),
                            F.normalize(semantic_features, dim=-1)], dim=-1)

        return {
            'scores': scores,
            'h_t': h_t,
            'alpha': alpha,
            'reconstructed_features': recon,
            'target': target
        }