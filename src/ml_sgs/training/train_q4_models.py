#!/usr/bin/env python3
"""
Q4 Multi-Task Models: Direct Coefficient Prediction Only
=========================================================

Essential control experiment: NO auxiliary tasks.
- No Richardson number prediction
- No regime classification
- Only viscosity and diffusivity coefficients

Q4 Hypothesis: This will perform worse than multi-task models,
               validating that auxiliary tasks provide useful inductive bias.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


# ==================== ARCHITECTURE VARIANTS ====================

class Q4_MLP(nn.Module):
    """
    Q4-MLP: Direct coefficient prediction without auxiliary tasks.
    Backbone EXACTLY matches train_new_coeff.py for fair comparison.
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
        
        # Only coefficient heads - matching baseline dimensions
        # Architecture: 256 → 128 → 64 → 1
        self.visc_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.diff_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Created Q4-MLP with {total_params:,} parameters")
    
    def forward(self, x):
        h = self.backbone(x)
        
        visc_pred = self.visc_head(h)
        diff_pred = self.diff_head(h)
        
        # Return tuple format: (visc, diff, None, None)
        # None values for Richardson and regime since Q4 doesn't predict them
        return (
            visc_pred.squeeze(-1),
            diff_pred.squeeze(-1),
            None,
            None
        )


class Q4_ResMLP(nn.Module):
    """
    Q4-ResMLP: Direct coefficient prediction with residual connections.
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
        
        # Only coefficient heads
        self.visc_head = nn.Sequential(
            nn.Linear(embed_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.diff_head = nn.Sequential(
            nn.Linear(embed_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Created Q4-ResMLP with {total_params:,} parameters")
    
    def forward(self, x):
        h = self.backbone(x)
        
        visc_pred = self.visc_head(h)
        diff_pred = self.diff_head(h)
        
        return (
            visc_pred.squeeze(-1),
            diff_pred.squeeze(-1),
            None,
            None
        )


class Q4_TabTransformer(nn.Module):
    """
    Q4-TabTransformer: Direct coefficient prediction with attention.
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
        
        # Only coefficient heads
        final_embedding_size = n_features * embed_dim
        
        self.visc_head = nn.Sequential(
            nn.Linear(final_embedding_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.diff_head = nn.Sequential(
            nn.Linear(final_embedding_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Created Q4-TabTransformer with {total_params:,} parameters")
    
    def forward(self, x):
        # Embed each feature: (batch, 54) → (batch, 54, 32)
        x = x.unsqueeze(-1)
        embeddings = self.feature_embedder(x)
        
        # Transformer layers
        for block in self.transformer_blocks:
            embeddings = block(embeddings)
        
        # Flatten for heads
        h = embeddings.view(-1, self.n_features * self.embed_dim)
        
        visc_pred = self.visc_head(h)
        diff_pred = self.diff_head(h)
        
        return (
            visc_pred.squeeze(-1),
            diff_pred.squeeze(-1),
            None,
            None
        )


# ==================== FACTORY FUNCTION ====================

def create_q4_model(architecture: str, n_features: int = 54):
    """
    Factory function to create Q4 models.
    
    Args:
        architecture: 'MLP', 'ResMLP', or 'TabTransformer'
        n_features: Number of input features
        
    Returns:
        Q4 model instance
    """
    if architecture == 'MLP':
        return Q4_MLP(n_features)
    elif architecture == 'ResMLP':
        return Q4_ResMLP(n_features)
    elif architecture == 'TabTransformer':
        return Q4_TabTransformer(n_features)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


# ==================== TESTING ====================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*70)
    print("Testing Q4 Models: Direct Coefficient Prediction Only")
    print("="*70 + "\n")
    
    batch_size = 32
    n_features = 54
    x = torch.randn(batch_size, n_features)
    
    models = {
        'Q4-MLP': create_q4_model('MLP', n_features),
        'Q4-ResMLP': create_q4_model('ResMLP', n_features),
        'Q4-TabTransformer': create_q4_model('TabTransformer', n_features)
    }
    
    for name, model in models.items():
        print(f"\n{name}:")
        print("-" * 50)
        
        visc, diff, ri, regime = model(x)
        
        print(f"  Viscosity shape: {visc.shape}")
        print(f"  Diffusivity shape: {diff.shape}")
        print(f"  Richardson: {ri} (None - not predicted)")
        print(f"  Regime: {regime} (None - not predicted)")
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Total parameters: {total_params:,}")
    
    print("\n✅ All Q4 models tested successfully!")
    print("\nQ4 Key Feature: No auxiliary tasks, direct coefficient prediction only")
