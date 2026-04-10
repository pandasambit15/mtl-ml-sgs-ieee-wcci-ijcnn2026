#!/usr/bin/env python3
"""
Q2 Multi-Task Models: Regime as Intermediary
=============================================

Architecture: Ri → Regime → Coefficients
- Richardson number is predicted first
- Regime classification uses predicted Ri
- Coefficient heads use regime logits (not Ri directly)

Q2 Hypothesis: Using regime classification as an intermediary between Ri and 
               coefficients provides better regime-aware predictions, especially
               for ARM data with diverse stability regimes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


# ==================== SHARED COMPONENTS ====================

class Q2_PredictionHeads(nn.Module):
    """
    Q2 Prediction heads: Regime acts as intermediary between Ri and coefficients.
    Architecture: Ri → Regime(Ri) → Coefficients(regime_logits)
    """
    def __init__(self, feature_dim: int):
        super().__init__()
        
        # Richardson prediction head (from backbone only)
        self.richardson_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Regime head takes [backbone, ri_pred]
        self.regime_head = nn.Sequential(
            nn.Linear(feature_dim + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # 3 regime classes
        )
        
        # KEY Q2 CHANGE: Coefficient heads take [backbone, regime_logits]
        # regime_logits has 3 values (before softmax)
        self.visc_head = nn.Sequential(
            nn.Linear(feature_dim + 3, 128),  # +3 for regime logits
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.diff_head = nn.Sequential(
            nn.Linear(feature_dim + 3, 128),  # +3 for regime logits
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, backbone_features: torch.Tensor):
        """
        Forward pass with regime as intermediary.
        
        Flow: backbone → ri_pred → regime_logits → coefficients
        
        Returns: (visc, diff, richardson, regime_logits)
        """
        # Step 1: Predict Richardson number
        ri_pred = self.richardson_head(backbone_features)
        
        # Step 2: Predict regime using Ri
        features_with_ri = torch.cat([backbone_features, ri_pred], dim=1)
        regime_logits = self.regime_head(features_with_ri)
        
        # Step 3: Predict coefficients using regime logits (Q2's key feature)
        features_with_regime = torch.cat([backbone_features, regime_logits], dim=1)
        visc_pred = self.visc_head(features_with_regime)
        diff_pred = self.diff_head(features_with_regime)
        
        return visc_pred, diff_pred, ri_pred, regime_logits


# ==================== ARCHITECTURE VARIANTS ====================

class Q2_MLP(nn.Module):
    """Q2-MLP: Regime as intermediary between Ri and coefficients."""
    
    def __init__(self, n_features=54):
        super().__init__()
        
        # Backbone: EXACT match to train_new_coeff.py
        self.backbone = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Q2 prediction heads
        self.heads = Q2_PredictionHeads(feature_dim=256)
        
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Created Q2-MLP with {total_params:,} parameters")
    
    def forward(self, x):
        features = self.backbone(x)
        return self.heads(features)


class Q2_ResMLP(nn.Module):
    """Q2-ResMLP: Regime as intermediary with residual connections."""
    
    def __init__(self, n_features=54, embed_size=256, num_blocks=4, dropout=0.3):
        super().__init__()
        
        from train_resmlp import ResidualBlock
        
        # Backbone: EXACT match to train_resmlp.py
        self.backbone = nn.Sequential(
            nn.Linear(n_features, embed_size),
            nn.ReLU(),
            *[ResidualBlock(embed_size, dropout) for _ in range(num_blocks)]
        )
        
        # Q2 prediction heads
        self.heads = Q2_PredictionHeads(feature_dim=embed_size)
        
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Created Q2-ResMLP with {total_params:,} parameters")
    
    def forward(self, x):
        features = self.backbone(x)
        return self.heads(features)


class Q2_TabTransformer(nn.Module):
    """Q2-TabTransformer: Regime as intermediary with attention."""
    
    def __init__(self, n_features=54, embed_dim=32, num_layers=4,
                 num_heads=8, ff_dim=128, dropout=0.2):
        super().__init__()
        
        self.n_features = n_features
        self.embed_dim = embed_dim
        
        from train_tab_transformer import TransformerBlock
        
        # Feature embedding: EXACT match
        self.feature_embedder = nn.Linear(1, embed_dim)
        
        # Transformer backbone: EXACT match
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
             for _ in range(num_layers)]
        )
        
        # Q2 prediction heads
        final_embedding_size = n_features * embed_dim
        self.heads = Q2_PredictionHeads(feature_dim=final_embedding_size)
        
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Created Q2-TabTransformer with {total_params:,} parameters")
    
    def forward(self, x):
        # Embed each feature
        x = x.unsqueeze(-1)
        embeddings = self.feature_embedder(x)
        
        # Transformer layers
        for block in self.transformer_blocks:
            embeddings = block(embeddings)
        
        # Flatten for heads
        features = embeddings.view(-1, self.n_features * self.embed_dim)
        
        return self.heads(features)


# ==================== FACTORY FUNCTION ====================

def create_q2_model(architecture: str, n_features: int = 54):
    """
    Factory function to create Q2 models.
    
    Args:
        architecture: 'MLP', 'ResMLP', or 'TabTransformer'
        n_features: Number of input features
        
    Returns:
        Q2 model instance
    """
    if architecture == 'MLP':
        return Q2_MLP(n_features)
    elif architecture == 'ResMLP':
        return Q2_ResMLP(n_features)
    elif architecture == 'TabTransformer':
        return Q2_TabTransformer(n_features)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


# ==================== TESTING ====================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*70)
    print("Testing Q2 Models: Regime as Intermediary")
    print("="*70 + "\n")
    
    batch_size = 32
    n_features = 54
    x = torch.randn(batch_size, n_features)
    
    models = {
        'Q2-MLP': create_q2_model('MLP', n_features),
        'Q2-ResMLP': create_q2_model('ResMLP', n_features),
        'Q2-TabTransformer': create_q2_model('TabTransformer', n_features)
    }
    
    for name, model in models.items():
        print(f"\n{name}:")
        print("-" * 50)
        
        visc, diff, ri, regime = model(x)
        
        print(f"  Viscosity shape: {visc.shape}")
        print(f"  Diffusivity shape: {diff.shape}")
        print(f"  Richardson shape: {ri.shape}")
        print(f"  Regime logits shape: {regime.shape}")
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Total parameters: {total_params:,}")
    
    print("\n✅ All Q2 models tested successfully!")
    print("\nQ2 Key Feature: Coefficients receive regime logits, not Ri directly")
    print("Information flow: Input → Ri → Regime(Ri) → Coefficients(regime_logits)")
