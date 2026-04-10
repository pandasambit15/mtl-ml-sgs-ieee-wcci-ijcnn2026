"""
ml_sgs.evaluation
=================

Metrics, statistical tests, visualisation, and prediction storage.

Modules
-------
metrics     – R², RMSE, MAE, bias, variance ratio, skill scores, Taylor diagrams,
              bootstrap confidence intervals, paired significance tests
analysis    – Cross-model comparison utilities, ranking tables
plotting    – Publication-quality bar charts, scatter grids, vertical profiles,
              distribution plots, summary figures
storage     – HDF5-backed prediction cache (save / load / compress)
"""

from ml_sgs.evaluation.metrics import (
    paired_ttest,
    wilcoxon_test,
    bootstrap_confidence_interval,
    nash_sutcliffe_efficiency,
    kling_gupta_efficiency,
    index_of_agreement,
)
from ml_sgs.evaluation.storage import PredictionStorage

__all__ = [
    "paired_ttest",
    "wilcoxon_test",
    "bootstrap_confidence_interval",
    "nash_sutcliffe_efficiency",
    "kling_gupta_efficiency",
    "index_of_agreement",
    "PredictionStorage",
]
