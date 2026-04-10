"""
ml_sgs.models
=============

Neural network architectures for SGS coefficient prediction.

Baseline (no physics conditioning)
-----------------------------------
- ``UnifiedSGSCoefficientNetwork``  – 4-head MLP (visc, diff, Ri, regime)
- ``ResMLP``                         – residual variant
- ``TabTransformer``                 – attention-based tabular variant

Richardson-number conditioned
------------------------------
- ``RiConditionedMLP``
- ``RiConditionedResMLP``
- ``RiConditionedTabTransformer``
- ``RichardsonConditionedHeads``    – shared head module

All Ri-conditioned variants predict the Richardson number first and then
feed it back into the coefficient heads (cascade conditioning).
"""

from ml_sgs.models.ri_conditioned import (
    RiConditionedMLP,
    RiConditionedResMLP,
    RiConditionedTabTransformer,
    RichardsonConditionedHeads,
    ResidualBlock,
    TransformerBlock,
)

__all__ = [
    "RiConditionedMLP",
    "RiConditionedResMLP",
    "RiConditionedTabTransformer",
    "RichardsonConditionedHeads",
    "ResidualBlock",
    "TransformerBlock",
]
