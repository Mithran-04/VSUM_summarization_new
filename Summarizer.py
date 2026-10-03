# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from Attention import (
    CNNProjection,
    SemanticProjection,
    CrossModalFusion,
    BiMambaSSM,
    DualParameterInterceptorScorer
)
from layers.decoder import MambaStateSpaceDecoder

import torch.nn.functional as F

class Summarizer(nn.Module):
    def __init__(self, cnn_size=1024, semantic_size=768, hidden_size=512,
                 d_state=16, dropout=0.1, summary_rate=0.15):
        super().__init__()
        self.summary_rate = summary_rate
        self.cnn_projection = CNNProjection(cnn_size, hidden_size, dropout)
        self.semantic_projection = SemanticProjection(semantic_size, hidden_size, dropout)
        self.cross_modal_fusion = CrossModalFusion(hidden_size, dropout)
        self.mamba_encoder = BiMambaSSM(d_model=hidden_size, d_state=d_state)
        self.interceptor_scorer = DualParameterInterceptorScorer(
            d_inner=hidden_size * 2, d_state=d_state, summary_rate=summary_rate,  d_model=hidden_size)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.decoder = MambaStateSpaceDecoder(hidden_size, d_state,
                                              out_size=cnn_size + semantic_size)
        # self.decoder = MambaStateSpaceDecoder(hidden_size, d_state,
        #                                               out_size=cnn_size)

    def forward(self, cnn_features, semantic_features):
        if cnn_features.ndim == 2:
            cnn_features, semantic_features = cnn_features.unsqueeze(0), semantic_features.unsqueeze(0)

        cnn_proj = self.cnn_projection(cnn_features)
        sem_proj = self.semantic_projection(semantic_features)
        fused = self.cross_modal_fusion(cnn_proj, sem_proj)                      # [B,T,512]
        # fused = cnn_proj

        temporal, d_fw, d_bw, h_fw, h_bw = self.mamba_encoder(fused)
        scores, delta_score, state_score, delta_t, d_t = self.interceptor_scorer(
            d_fw, d_bw, h_fw, h_bw, temporal)

        # Hard top-k with straight-through gradient
        B, T, _ = scores.shape
        k = max(1, int(round(T * self.summary_rate)))
        idx = scores.squeeze(-1).topk(k, dim=1).indices                          # [B,k]
        hard = torch.zeros_like(scores).scatter_(1, idx.unsqueeze(-1), 1.0)
        mask = hard + scores - scores.detach()

        # The decoder sees only selected frames' own features (no neighbor leakage)
        dec_in = fused * mask + self.mask_token * (1 - mask)
        recon = self.decoder(dec_in)                                             # [B,T,1792]

        # Fixed target: normalized raw inputs
        # target = torch.cat([F.normalize(cnn_features, dim=-1),
        #                     F.normalize(semantic_features, dim=-1)], dim=-1)
        target = torch.cat([F.normalize(cnn_features, dim=-1),
                                 F.normalize(semantic_features, dim=-1)
                                 ],dim=-1)

        return {'scores': scores, 'delta_score': delta_score, 'state_score': state_score,
                'delta_t': delta_t, 'd_t': d_t, 'fused_features': fused,
                'temporal_features': temporal, 'selected_idx': idx,
                'reconstructed_features': recon, 'mask': mask, 'target': target}