# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- _Nothing yet._

### Changed
- _Nothing yet._

### Fixed
- _Nothing yet._

---

## [1.0.0] – 2026-01-01

### Added
- **`ml_sgs.models`**: Baseline MLP, ResMLP, and TabTransformer architectures for
  multi-task SGS coefficient prediction (viscosity, diffusivity, Richardson number,
  regime classification).
- **`ml_sgs.models.ri_conditioned`**: Richardson-number cascade conditioning for all
  three backbone architectures (`RiConditionedMLP`, `RiConditionedResMLP`,
  `RiConditionedTabTransformer`).
- **`ml_sgs.data.processor`**: Parallelised MONC NetCDF processor (`FastUnifiedProcessor`)
  supporting ARM and RCE simulation types with automatic time-dimension detection.
- **`ml_sgs.data.enhanced_processor`**: Extended data processor with additional
  diagnostic outputs and stratified sampling.
- **`ml_sgs.training`**: Training scripts for all model variants with early stopping,
  learning-rate scheduling, gradient clipping, and multi-task loss weighting.
- **`ml_sgs.inference.engine`**: `UnifiedInferenceEngine` providing a single interface
  for baseline, Ri-conditioned, and Q1–Q4 models.
- **`ml_sgs.inference.q1q4_engine`**: `Q1Q4UnifiedEngine` with regime-based routing
  of samples to quadrant-specific models.
- **`ml_sgs.inference.constrained`**: Post-hoc physical constraint enforcement
  (positivity, Ri-based bounds).
- **`ml_sgs.evaluation.metrics`**: Comprehensive skill scores: R², RMSE, MAE, bias,
  variance ratio, Nash-Sutcliffe efficiency, Kling-Gupta efficiency, Index of Agreement.
- **`ml_sgs.evaluation.metrics`**: Statistical significance: paired t-test, Wilcoxon
  signed-rank test, bootstrap confidence intervals, Taylor diagrams.
- **`ml_sgs.evaluation.analysis`**: Cross-model ranking and comparison tables.
- **`ml_sgs.evaluation.plotting`**: Publication-quality figures: bar charts, scatter
  grids, vertical profiles, distribution plots, Taylor diagrams, summary panels.
- **`ml_sgs.evaluation.storage`**: HDF5-backed prediction cache with gzip compression.
- **`scripts/compare_models.py`**: Master CLI for end-to-end model comparison.
- **`scripts/run_analysis.py`**: Post-hoc analysis on saved predictions.
- **`scripts/run_ri_comparison.py`**: Focused Ri-conditioning vs. baseline comparison.
- **`configs/default.yaml`**: Documented default configuration for all pipeline stages.
- Full test suite (`tests/`) covering models, metrics, data loading, and inference.
- GitHub Actions CI with linting, type-checking, and multi-Python-version testing.
- Pre-commit hooks (ruff, mypy).
- Comprehensive documentation (`docs/`).

[Unreleased]: https://github.com/your-org/ml-sgs-turbulence/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/your-org/ml-sgs-turbulence/releases/tag/v1.0.0
