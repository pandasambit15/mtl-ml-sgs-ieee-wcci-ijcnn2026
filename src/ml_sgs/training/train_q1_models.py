#!/usr/bin/env python3
"""
Q1 Multi-Task Models: Richardson Number Conditions ALL Heads
=============================================================

Key difference from Ri-conditioned baseline:
- Regime head NOW also receives Richardson number as input
- Architecture matches train_*.py exactly

Q1 Hypothesis: Conditioning regime classification on predicted Ri will help
               the model learn better representations, even if Ri prediction
               contains errors that cascade to downstream tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


# ==================== SHARED COMPONENTS ====================

class Q1_PredictionHeads(nn.Module):
    """
    Q1 Prediction heads: ALL heads (including regime) receive Richardson conditioning.
    Matches EXACT head dimensions from your train_*.py scripts.
    """
    def __init__(self, feature_dim: int):
        super().__init__()
        
        # Richardson prediction head (predicts FROM backbone features only)
        # Architecture: feature_dim → 128 → 64 → 1 (matching your baseline)
        self.richardson_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # KEY Q1 CHANGE: Regime head NOW takes [backbone, ri_pred]
        # Architecture: (feature_dim + 1) → 128 → 64 → 3
        self.regime_head = nn.Sequential(
            nn.Linear(feature_dim + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        
        # Coefficient heads (conditioned on Ri, same as Ri-baseline)
        # Architecture: (feature_dim + 1) → 128 → 64 → 1
        self.visc_head = nn.Sequential(
            nn.Linear(feature_dim + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.diff_head = nn.Sequential(
            nn.Linear(feature_dim + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, backbone_features: torch.Tensor):
        """
        Forward pass with Richardson conditioning on ALL heads.
        
        Returns: (visc, diff, richardson, regime_logits)
        """
        # Step 1: Predict Richardson number from backbone only
        ri_pred = self.richardson_head(backbone_features)
        
        # Step 2: Concatenate Ri with backbone for ALL downstream predictions
        conditioned_features = torch.cat([backbone_features, ri_pred], dim=1)
        
        # Step 3: ALL heads receive Ri conditioning (Q1's key feature)
        regime_pred = self.regime_head(conditioned_features)  # Q1: regime gets Ri
        visc_pred = self.visc_head(conditioned_features)
        diff_pred = self.diff_head(conditioned_features)
        
        return visc_pred, diff_pred, ri_pred, regime_pred


# ==================== ARCHITECTURE VARIANTS ====================

class Q1_MLP(nn.Module):
    """
    Q1-MLP: Richardson conditions ALL heads including regime.
    Backbone EXACTLY matches train_new_coeff.py
    """
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
        
        # Q1 prediction heads
        self.heads = Q1_PredictionHeads(feature_dim=256)
        
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Created Q1-MLP with {total_params:,} parameters")
    
    def forward(self, x):
        features = self.backbone(x)
        return self.heads(features)


class Q1_ResMLP(nn.Module):
    """
    Q1-ResMLP: Richardson conditions ALL heads.
    Backbone EXACTLY matches train_resmlp.py
    """
    def __init__(self, n_features=54, embed_size=256, num_blocks=4, dropout=0.3):
        super().__init__()
        
        # Import ResidualBlock from train_resmlp
        from train_resmlp import ResidualBlock
        
        # Backbone: EXACT match to train_resmlp.py
        self.backbone = nn.Sequential(
            nn.Linear(n_features, embed_size),
            nn.ReLU(),
            *[ResidualBlock(embed_size, dropout) for _ in range(num_blocks)]
        )
        
        # Q1 prediction heads
        self.heads = Q1_PredictionHeads(feature_dim=embed_size)
        
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Created Q1-ResMLP with {total_params:,} parameters")
    
    def forward(self, x):
        features = self.backbone(x)
        return self.heads(features)


class Q1_TabTransformer(nn.Module):
    """
    Q1-TabTransformer: Richardson conditions ALL heads.
    Backbone EXACTLY matches train_tab_transformer.py
    """
    def __init__(self, n_features=54, embed_dim=32, num_layers=4, 
                 num_heads=8, ff_dim=128, dropout=0.2):
        super().__init__()
        
        self.n_features = n_features
        self.embed_dim = embed_dim
        
        # Import TransformerBlock from train_tab_transformer
        from train_tab_transformer import TransformerBlock
        
        # Feature embedding: EXACT match
        self.feature_embedder = nn.Linear(1, embed_dim)
        
        # Transformer backbone: EXACT match
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, ff_dim, dropout) 
             for _ in range(num_layers)]
        )
        
        # Q1 prediction heads
        final_embedding_size = n_features * embed_dim
        self.heads = Q1_PredictionHeads(feature_dim=final_embedding_size)
        
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Created Q1-TabTransformer with {total_params:,} parameters")
    
    def forward(self, x):
        # Embed each feature: (batch, 54) → (batch, 54, 32)
        x = x.unsqueeze(-1)
        embeddings = self.feature_embedder(x)
        
        # Transformer layers
        for block in self.transformer_blocks:
            embeddings = block(embeddings)
        
        # Flatten for heads
        features = embeddings.view(-1, self.n_features * self.embed_dim)
        
        return self.heads(features)


# ==================== FACTORY FUNCTION ====================

def create_q1_model(architecture: str, n_features: int = 54):
    """
    Factory function to create Q1 models.
    
    Args:
        architecture: 'MLP', 'ResMLP', or 'TabTransformer'
        n_features: Number of input features
        
    Returns:
        Q1 model instance
    """
    if architecture == 'MLP':
        return Q1_MLP(n_features)
    elif architecture == 'ResMLP':
        return Q1_ResMLP(n_features)
    elif architecture == 'TabTransformer':
        return Q1_TabTransformer(n_features)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


# ==================== TESTING ====================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*70)
    print("Testing Q1 Models: Richardson Conditions ALL Heads")
    print("="*70 + "\n")
    
    batch_size = 32
    n_features = 54
    x = torch.randn(batch_size, n_features)
    
    models = {
        'Q1-MLP': create_q1_model('MLP', n_features),
        'Q1-ResMLP': create_q1_model('ResMLP', n_features),
        'Q1-TabTransformer': create_q1_model('TabTransformer', n_features)
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
    
    print("\n✅ All Q1 models tested successfully!")
    print("\nQ1 Key Feature: Regime head receives [backbone; predicted_Ri]")
