"""
ml_sgs.inference
================

Unified and physics-constrained inference engines.

Classes
-------
UnifiedInferenceEngine          – Single interface for baseline, Ri, and Q1-Q4 models
Q1Q4UnifiedEngine               – Quadrant-conditioned inference with regime routing
ConstrainedInferenceEngine      – Post-hoc physical constraint enforcement (Ri-based)
"""

from ml_sgs.inference.engine import UnifiedInferenceEngine
from ml_sgs.inference.q1q4_engine import Q1Q4UnifiedEngine

__all__ = ["UnifiedInferenceEngine", "Q1Q4UnifiedEngine"]
