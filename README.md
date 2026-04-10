# ml-sgs-turbulence

**Physics-Guided Multi-Task Learning for Subgrid-Scale Turbulence Parameterisation**

[![CI](https://github.com/your-org/ml-sgs-turbulence/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/ml-sgs-turbulence/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/your-org/ml-sgs-turbulence/branch/main/graph/badge.svg)](https://codecov.io/gh/your-org/ml-sgs-turbulence)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

---

A research framework accompanying the paper:

> **Panda, S.K., Jones, T., Shahzad, M., Lawrence, B. & Ellis, A.-L. (2026).**
> *Physics-Guided Multi-Task Learning for Subgrid Scale Turbulence
> Parameterization: A Comparative Study of Physics Integration Strategies.*
> IEEE WCCI IJCNN 2026, Maastricht, Netherlands.

The framework trains and evaluates machine learning models that predict
**SGS eddy viscosity and diffusivity coefficients** from resolved-scale
flow variables in the [Met Office NERC Cloud model (MONC)](https://github.com/EPIC-model/monc),
comparing three **physics integration strategies**:

| Strategy | Description |
|---|---|
| **Baseline** | Standard multi-task MLP with no explicit physics inductive bias |
| **Ri-conditioned** | Cascade conditioning: predict Richardson number first, then feed it into coefficient heads |
| **Q1–Q4** | Train separate models per stability quadrant; route at inference time |

Each strategy is implemented for three backbone architectures: **MLP**, **ResMLP**, and **TabTransformer**.

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [1. Process Data](#1-process-data)
  - [2. Train Models](#2-train-models)
  - [3. Run Comparison](#3-run-comparison)
  - [4. Analyse Results](#4-analyse-results)
- [Configuration](#configuration)
- [Output Structure](#output-structure)
- [Documentation](#documentation)
- [Testing](#testing)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

---

## Features

- **Unified inference** – single `UnifiedInferenceEngine` interface for all 18 model configurations
- **Constrained inference** – post-hoc enforcement of physical positivity constraints
- **Efficient storage** – HDF5-backed prediction cache (gzip compressed) to avoid re-running inference
- **Comprehensive metrics** – R², RMSE, MAE, bias, variance ratio, Nash-Sutcliffe efficiency,
  Kling-Gupta efficiency, Index of Agreement, Taylor diagrams
- **Statistical rigour** – paired t-tests, Wilcoxon signed-rank tests, bootstrap confidence intervals
- **Publication plots** – bar charts, scatter grids, vertical profiles, distributions, summary panels
- **Reproducible** – seeded training, config files, versioned checkpoints
- **Parallel data processing** – multi-process MONC NetCDF ingestion

---

## Project Structure

```
ml-sgs-turbulence/
├── src/
│   └── ml_sgs/
│       ├── __init__.py
│       ├── models/
│       │   ├── __init__.py
│       │   └── ri_conditioned.py       # All Ri-conditioned architectures
│       ├── data/
│       │   ├── __init__.py
│       │   ├── processor.py            # Parallel MONC NetCDF processor
│       │   └── enhanced_processor.py  # Extended processor with diagnostics
│       ├── training/
│       │   ├── __init__.py
│       │   ├── train_baseline.py       # Baseline MLP training
│       │   ├── train_resmlp.py
│       │   ├── train_tab_transformer.py
│       │   ├── train_ri_conditioned.py # Ri-conditioned training
│       │   └── train_q{1,2,3,4}_models.py
│       ├── inference/
│       │   ├── __init__.py
│       │   ├── engine.py               # UnifiedInferenceEngine
│       │   ├── q1q4_engine.py          # Q1–Q4 routing engine
│       │   └── constrained.py         # Physics-constrained inference
│       └── evaluation/
│           ├── __init__.py
│           ├── metrics.py             # Skill scores & significance tests
│           ├── analysis.py            # Cross-model comparison utilities
│           ├── plotting.py            # Publication-quality figures
│           └── storage.py            # HDF5 prediction cache
├── scripts/
│   ├── compare_models.py              # Master end-to-end comparison CLI
│   ├── run_analysis.py                # Post-hoc analysis on saved predictions
│   └── run_ri_comparison.py           # Baseline vs. Ri-conditioned deep-dive
├── configs/
│   └── default.yaml                   # Documented default configuration
├── tests/
│   ├── test_models.py
│   ├── test_metrics.py
│   ├── test_data.py
│   └── test_inference.py
├── docs/
│   ├── architecture.md
│   ├── data_format.md
│   └── training_guide.md
├── .github/
│   └── workflows/ci.yml
├── pyproject.toml
├── requirements.txt
├── CHANGELOG.md
├── CONTRIBUTING.md
└── LICENSE
```

---

## Installation

### Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0 (CPU or CUDA)

### Install from source

```bash
# Clone the repository
git clone https://github.com/your-org/ml-sgs-turbulence.git
cd ml-sgs-turbulence

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install runtime dependencies
pip install -r requirements.txt

# Install the package in editable mode
pip install -e .
```

For GPU training, replace the PyTorch install:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## Quick Start

### Compare all models end-to-end

```bash
python scripts/compare_models.py \
    --data-dir /path/to/netcdf_files/ \
    --output   results/ \
    --baseline-mlp   models/baseline_mlp/best_model.pt \
    --ri-mlp         models/ri_conditioned/mlp/best_model.pt \
    --q1-mlp         models/q1/best_model.pt \
    --q2-mlp         models/q2/best_model.pt \
    --q3-mlp         models/q3/best_model.pt \
    --q4-mlp         models/q4/best_model.pt \
    --scaler-dir     models/baseline_mlp/ \
    --config         configs/default.yaml \
    --save-predictions
```

### Re-analyse from saved predictions (no re-inference)

```bash
python scripts/compare_models.py \
    --data-dir         /path/to/netcdf_files/ \
    --output           results_v2/ \
    --load-predictions \
    --prediction-dir   results/predictions/
```

### Smoke test on 10 % of files

```bash
python scripts/compare_models.py \
    --data-dir       /path/to/netcdf_files/ \
    --output         test_results/ \
    --baseline-mlp   models/baseline_mlp/best_model.pt \
    --scaler-dir     models/baseline_mlp/ \
    --sample-rate    0.1
```

---

## Usage

### 1. Process Data

```bash
python -m ml_sgs.data.processor \
    --data-dir  /path/to/netcdf_files/ \
    --output-dir processed_data/ \
    --sampling-strategy stratified \
    --sampling-fraction 0.5 \
    --n-workers 8
```

See [docs/data_format.md](docs/data_format.md) for the expected NetCDF variables
and the produced feature engineering pipeline.

### 2. Train Models

```bash
# Baseline MLP
python -m ml_sgs.training.train_baseline \
    --data-dir processed_data/ \
    --output-dir models/baseline_mlp/

# Ri-conditioned MLP
python -m ml_sgs.training.train_ri_conditioned \
    --data-dir processed_data/ \
    --output-dir models/ri_conditioned/ \
    --arch mlp

# Quadrant models (run in parallel)
for q in 1 2 3 4; do
    python -m ml_sgs.training.train_q${q}_models \
        --data-dir processed_data/ \
        --output-dir models/q${q}/ &
done; wait
```

See [docs/training_guide.md](docs/training_guide.md) for full details and
hyperparameter reference.

### 3. Run Comparison

```bash
python scripts/compare_models.py --help
```

### 4. Analyse Results

```python
from pathlib import Path
from ml_sgs.evaluation import PredictionStorage, nash_sutcliffe_efficiency
from ml_sgs.evaluation.plotting import plot_metrics_bar_chart

# Load cached predictions
store = PredictionStorage("results/predictions/predictions.h5")
visc_true  = store.load("ground_truth/visc_coeff")
visc_base  = store.load("baseline/MLP/visc_coeff")
visc_ri    = store.load("ri_conditioned/MLP/visc_coeff")

# Compute skill scores
print("Baseline NSE :", nash_sutcliffe_efficiency(visc_base, visc_true))
print("Ri-cond  NSE :", nash_sutcliffe_efficiency(visc_ri,   visc_true))

# Publication figure
plot_metrics_bar_chart(
    metrics={"Baseline MLP": visc_base, "Ri-MLP": visc_ri},
    truth=visc_true,
    output_path=Path("results/plots/nse_comparison.png"),
)
```

---

## Configuration

All parameters are documented in [`configs/default.yaml`](configs/default.yaml).
Pass a custom config with `--config path/to/config.yaml`; missing keys fall back
to the defaults.

Key sections:

| Section      | Controls                                          |
|--------------|---------------------------------------------------|
| `data`       | Input paths, worker count, vertical range         |
| `scalers`    | Scaler directory, fallback behaviour              |
| `training`   | Batch size, epochs, LR, dropout, loss weights     |
| `model`      | Architecture hyperparameters for all variants     |
| `inference`  | Device, batch size, physical constraints          |
| `evaluation` | Metrics, significance level, bootstrap resamples  |
| `output`     | Paths, plot DPI and format, compression           |

---

## Output Structure

```
results/
├── metrics_domain.csv            # Domain-averaged metrics table
├── metrics_nonzero.csv           # Non-zero-only metrics
├── comparison_table.csv          # Publication-ready comparison table
├── metrics_all.json              # Full metrics + best models per variable
├── prediction_storage_info.json  # Cache statistics
├── plots/
│   ├── metrics_comparison_r2.png
│   ├── scatter_grid_visc_coeff_part1.png
│   ├── distribution_visc_coeff_log_part1.png
│   ├── vertical_profiles_visc_coeff_MLP.png
│   └── summary_figure.png
└── predictions/
    ├── predictions.h5            # Compressed HDF5 prediction cache
    └── metadata.json
```

---

## Documentation

| Document | Contents |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Network architectures, head design, parameter counts |
| [docs/data_format.md](docs/data_format.md) | NetCDF variables, feature engineering, HDF5 storage schema |
| [docs/training_guide.md](docs/training_guide.md) | Step-by-step training walkthrough, hyperparameter reference |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Development setup, code style, PR workflow |
| [CHANGELOG.md](CHANGELOG.md) | Version history |

---

## Testing

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run full test suite
pytest

# With coverage
pytest --cov=ml_sgs --cov-report=term-missing

# Specific module
pytest tests/test_models.py -v
```

The CI pipeline runs tests on Python 3.10, 3.11, and 3.12.

---

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) before
opening a pull request. Use the [bug report](.github/ISSUE_TEMPLATE/bug_report.md)
or [feature request](.github/ISSUE_TEMPLATE/feature_request.md) templates for issues.

---

## Citation

If you use this framework in your research, please cite:

```bibtex
@inproceedings{panda2026physics,
  title     = {Physics-Guided Multi-Task Learning for Subgrid Scale Turbulence
               Parameterization: A Comparative Study of Physics Integration Strategies},
  author    = {Panda, Sambit Kumar and Jones, Todd and Shahzad, Muhammad and
               Lawrence, Bryan and Ellis, Anna-Louise},
  booktitle = {Proceedings of the IEEE World Congress on Computational Intelligence
               (WCCI), International Joint Conference on Neural Networks (IJCNN)},
  year      = {2026},
  address   = {Maastricht, Netherlands},
}
```

---

## License

This project is released under the [MIT License](LICENSE).
