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

class Summarizer(nn.Module):
    """
    Complete Unsupervised Video Summarization Architecture with
    Bi-Directional Mamba Interception.
    """
    def __init__(
        self,
        cnn_size=1024,
        semantic_size=768,
        hidden_size=512,
        d_state=16,
        dropout=0.1
    ):
        super().__init__()

        # Projections
        self.cnn_projection = CNNProjection(input_size=cnn_size, hidden_size=hidden_size, dropout=dropout)
        self.semantic_projection = SemanticProjection(semantic_size=semantic_size, hidden_size=hidden_size, dropout=dropout)

        # Multimodal Fusion
        self.cross_modal_fusion = CrossModalFusion(hidden_size=hidden_size, dropout=dropout)

        # Bi-Directional Mamba Encoder
        self.mamba_encoder = BiMambaSSM(d_model=hidden_size, d_state=d_state)

        # Dual Parameter Interceptor Scorer
        self.interceptor_scorer = DualParameterInterceptorScorer(
            d_inner=hidden_size * 2,
            d_state=d_state
        )

        # Mamba Reconstruction Decoder
        self.decoder = MambaStateSpaceDecoder(hidden_size=hidden_size, d_state=d_state)

    def forward(self, cnn_features, semantic_features):
        # 1. Feature Projections
        
        cnn_proj = self.cnn_projection(cnn_features)          # [B, T, 512]
        sem_proj = self.semantic_projection(semantic_features) # [B, T, 512]
        #print("1. Projections:", cnn_proj.shape, sem_proj.shape)

        # 2. Multimodal Fusion
        fused_features = self.cross_modal_fusion(cnn_proj, sem_proj) # [B, T, 512]
        #print("2. Fused:", fused_features.shape)

        # 3. Bi-Directional Mamba SSM Encoding
        temporal_features, delta_fw, delta_bw, h_fw, h_bw = self.mamba_encoder(fused_features)
        #print("3. Encoder output:", temporal_features.shape)

        # 4. Intercept State Mechanics & Compute Scores
        scores, delta_score, state_score, delta_t, d_t = self.interceptor_scorer(delta_fw, delta_bw, h_fw, h_bw)
        #print("4. Scores shape:", scores.shape)

        # 5. Soft Gating / Frame Selection
        weighted_features = temporal_features * scores  # [B, T, 512]
        #print("5. Weighted features shape:", weighted_features.shape)

        # 6. Continuous Decoder Reconstruction
        reconstructed_features = self.decoder(weighted_features)
        #print("6. Decoder output shape:", reconstructed_features.shape)

        return {
            'scores': scores,
            'delta_score': delta_score,
            'state_score': state_score,
            'delta_t': delta_t,
            'd_t': d_t,
            'fused_features': fused_features,
            'temporal_features': temporal_features,
            'weighted_features': weighted_features,
            'reconstructed_features': reconstructed_features
        }