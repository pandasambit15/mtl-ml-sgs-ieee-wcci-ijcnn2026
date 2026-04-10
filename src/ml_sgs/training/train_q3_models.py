#!/usr/bin/env python3
"""
Q3 Multi-Task Models: Sequential Hierarchical Training
=======================================================

Architecture: Same as Q2 (Ri → Regime → Coefficients)
Training: Three-stage curriculum learning
  - Stage 1: Train Richardson head only
  - Stage 2: Freeze Ri, train Regime head
  - Stage 3: Freeze Ri+Regime, train coefficient heads

Q3 Hypothesis: Sequential training reduces error cascading and allows each
               component to learn optimal representations before adding
               downstream dependencies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


# ==================== SHARED COMPONENTS ====================

class Q3_PredictionHeads(nn.Module):
    """
    Q3 Prediction heads: Same architecture as Q2 but designed for sequential training.
    Architecture: Ri → Regime(Ri) → Coefficients(regime_logits)
    """
    def __init__(self, feature_dim: int):
        super().__init__()
        
        # Richardson prediction head (Stage 1)
        self.richardson_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Regime head (Stage 2) - takes [backbone, ri_pred]
        self.regime_head = nn.Sequential(
            nn.Linear(feature_dim + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        
        # Coefficient heads (Stage 3) - take [backbone, regime_logits]
        self.visc_head = nn.Sequential(
            nn.Linear(feature_dim + 3, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.diff_head = nn.Sequential(
            nn.Linear(feature_dim + 3, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, backbone_features: torch.Tensor, stage: int = 3):
        """
        Forward pass with stage-aware computation.
        
        Args:
            backbone_features: Features from backbone
            stage: Training stage (1, 2, or 3)
                  Stage 1: Only Richardson
                  Stage 2: Richardson + Regime
                  Stage 3: Full pipeline
        
        Returns: (visc, diff, richardson, regime_logits)
        """
        # Stage 1: Predict Richardson
        ri_pred = self.richardson_head(backbone_features)
        
        if stage == 1:
            # Only Richardson is being trained
            return None, None, ri_pred, None
        
        # Stage 2+: Predict regime using Ri
        features_with_ri = torch.cat([backbone_features, ri_pred], dim=1)
        regime_logits = self.regime_head(features_with_ri)
        
        if stage == 2:
            # Only Richardson and Regime trained
            return None, None, ri_pred, regime_logits
        
        # Stage 3: Full pipeline - predict coefficients
        features_with_regime = torch.cat([backbone_features, regime_logits], dim=1)
        visc_pred = self.visc_head(features_with_regime)
        diff_pred = self.diff_head(features_with_regime)
        
        return visc_pred, diff_pred, ri_pred, regime_logits
    
    def freeze_richardson(self):
        """Freeze Richardson head parameters (for Stage 2)."""
        for param in self.richardson_head.parameters():
            param.requires_grad = False
        logger.info("Froze Richardson head")
    
    def freeze_regime(self):
        """Freeze Regime head parameters (for Stage 3)."""
        for param in self.regime_head.parameters():
            param.requires_grad = False
        logger.info("Froze Regime head")
    
    def get_trainable_params(self, stage: int):
        """Get trainable parameters for specific stage."""
        if stage == 1:
            return self.richardson_head.parameters()
        elif stage == 2:
            return self.regime_head.parameters()
        elif stage == 3:
            params = list(self.visc_head.parameters()) + list(self.diff_head.parameters())
            return iter(params)
        else:
            raise ValueError(f"Invalid stage: {stage}")


# ==================== ARCHITECTURE VARIANTS ====================

class Q3_MLP(nn.Module):
    """Q3-MLP: Sequential hierarchical training."""
    
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
        
        # Q3 prediction heads with stage support
        self.heads = Q3_PredictionHeads(feature_dim=256)
        self.current_stage = 3  # Default to full pipeline
        
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Created Q3-MLP with {total_params:,} parameters")
    
    def forward(self, x):
        features = self.backbone(x)
        return self.heads(features, stage=self.current_stage)
    
    def set_stage(self, stage: int):
        """Set training stage and freeze appropriate components."""
        self.current_stage = stage
        if stage == 2:
            self.heads.freeze_richardson()
        elif stage == 3:
            self.heads.freeze_richardson()
            self.heads.freeze_regime()
        logger.info(f"Set training stage to {stage}")


class Q3_ResMLP(nn.Module):
    """Q3-ResMLP: Sequential hierarchical training with residual connections."""
    
    def __init__(self, n_features=54, embed_size=256, num_blocks=4, dropout=0.3):
        super().__init__()
        
        from train_resmlp import ResidualBlock
        
        # Backbone: EXACT match to train_resmlp.py
        self.backbone = nn.Sequential(
            nn.Linear(n_features, embed_size),
            nn.ReLU(),
            *[ResidualBlock(embed_size, dropout) for _ in range(num_blocks)]
        )
        
        # Q3 prediction heads
        self.heads = Q3_PredictionHeads(feature_dim=embed_size)
        self.current_stage = 3
        
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Created Q3-ResMLP with {total_params:,} parameters")
    
    def forward(self, x):
        features = self.backbone(x)
        return self.heads(features, stage=self.current_stage)
    
    def set_stage(self, stage: int):
        """Set training stage and freeze appropriate components."""
        self.current_stage = stage
        if stage == 2:
            self.heads.freeze_richardson()
        elif stage == 3:
            self.heads.freeze_richardson()
            self.heads.freeze_regime()
        logger.info(f"Set training stage to {stage}")


class Q3_TabTransformer(nn.Module):
    """Q3-TabTransformer: Sequential hierarchical training with attention."""
    
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
        
        # Q3 prediction heads
        final_embedding_size = n_features * embed_dim
        self.heads = Q3_PredictionHeads(feature_dim=final_embedding_size)
        self.current_stage = 3
        
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Created Q3-TabTransformer with {total_params:,} parameters")
    
    def forward(self, x):
        # Embed each feature
        x = x.unsqueeze(-1)
        embeddings = self.feature_embedder(x)
        
        # Transformer layers
        for block in self.transformer_blocks:
            embeddings = block(embeddings)
        
        # Flatten for heads
        features = embeddings.view(-1, self.n_features * self.embed_dim)
        
        return self.heads(features, stage=self.current_stage)
    
    def set_stage(self, stage: int):
        """Set training stage and freeze appropriate components."""
        self.current_stage = stage
        if stage == 2:
            self.heads.freeze_richardson()
        elif stage == 3:
            self.heads.freeze_richardson()
            self.heads.freeze_regime()
        logger.info(f"Set training stage to {stage}")


# ==================== FACTORY FUNCTION ====================

def create_q3_model(architecture: str, n_features: int = 54):
    """
    Factory function to create Q3 models.
    
    Args:
        architecture: 'MLP', 'ResMLP', or 'TabTransformer'
        n_features: Number of input features
        
    Returns:
        Q3 model instance
    """
    if architecture == 'MLP':
        return Q3_MLP(n_features)
    elif architecture == 'ResMLP':
        return Q3_ResMLP(n_features)
    elif architecture == 'TabTransformer':
        return Q3_TabTransformer(n_features)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


# ==================== TESTING ====================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*70)
    print("Testing Q3 Models: Sequential Hierarchical Training")
    print("="*70 + "\n")
    
    batch_size = 32
    n_features = 54
    x = torch.randn(batch_size, n_features)
    
    models = {
        'Q3-MLP': create_q3_model('MLP', n_features),
        'Q3-ResMLP': create_q3_model('ResMLP', n_features),
        'Q3-TabTransformer': create_q3_model('TabTransformer', n_features)
    }
    
    for name, model in models.items():
        print(f"\n{name}:")
        print("-" * 50)
        
        # Test Stage 1
        model.set_stage(1)
        visc, diff, ri, regime = model(x)
        print(f"  Stage 1 - Richardson only: ri={ri.shape}, visc={visc}, regime={regime}")
        
        # Test Stage 2
        model.set_stage(2)
        visc, diff, ri, regime = model(x)
        print(f"  Stage 2 - Ri+Regime: ri={ri.shape}, regime={regime.shape}, visc={visc}")
        
        # Test Stage 3
        model.set_stage(3)
        visc, diff, ri, regime = model(x)
        print(f"  Stage 3 - Full: visc={visc.shape}, diff={diff.shape}, ri={ri.shape}, regime={regime.shape}")
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Total parameters: {total_params:,}")
    
    print("\n✅ All Q3 models tested successfully!")
    print("\nQ3 Key Feature: Three-stage sequential training")
    print("Stage 1: Train Ri | Stage 2: Freeze Ri, train Regime | Stage 3: Freeze both, train coefficients")
