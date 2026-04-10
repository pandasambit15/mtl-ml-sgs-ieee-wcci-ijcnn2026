# Contributing

Thank you for considering a contribution to **ml-sgs-turbulence**! This document
explains how to set up a development environment, run the test suite, and submit
a pull request.

---

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Setting Up a Development Environment](#setting-up-a-development-environment)
3. [Running Tests](#running-tests)
4. [Code Style](#code-style)
5. [Submitting a Pull Request](#submitting-a-pull-request)
6. [Adding a New Model Variant](#adding-a-new-model-variant)
7. [Reporting Bugs](#reporting-bugs)

---

## Code of Conduct

This project follows the [Contributor Covenant](https://www.contributor-covenant.org/)
v2.1. By participating you agree to uphold a respectful and inclusive environment.

---

## Setting Up a Development Environment

```bash
# 1. Fork and clone the repository
git clone https://github.com/your-org/ml-sgs-turbulence.git
cd ml-sgs-turbulence

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install in editable mode with development dependencies
pip install -e ".[dev]"

# 4. Install pre-commit hooks
pre-commit install
```

> **GPU note**: The CI pipeline uses a CPU-only PyTorch build. For local GPU
> development, install the appropriate CUDA wheel from
> https://pytorch.org/get-started/locally/

---

## Running Tests

```bash
# Run the full suite
pytest

# Run with coverage report
pytest --cov=ml_sgs --cov-report=term-missing

# Run a single test file
pytest tests/test_models.py -v

# Run tests in parallel (requires pytest-xdist)
pytest -n auto
```

All tests must pass before a PR can be merged.

---

## Code Style

We use [**ruff**](https://docs.astral.sh/ruff/) for linting and formatting, and
[**mypy**](https://mypy.readthedocs.io/) for type checking.

```bash
# Lint and auto-fix
ruff check --fix src/ scripts/ tests/

# Format
ruff format src/ scripts/ tests/

# Type check
mypy src/ml_sgs
```

Pre-commit hooks run these automatically on every commit if you have run
`pre-commit install`.

Key style conventions:
- Line length: **100 characters**
- Docstrings: [NumPy style](https://numpydoc.readthedocs.io/en/latest/format.html)
- Type annotations required for all public functions and methods
- Variable names in scientific code may use standard notation (e.g. `Ri`, `dU_dz`)

---

## Submitting a Pull Request

1. Create a feature branch from `develop`:
   ```bash
   git checkout -b feat/my-new-model develop
   ```
2. Make your changes, write tests, and update `CHANGELOG.md` under `[Unreleased]`.
3. Push your branch and open a PR against `develop` (not `main`).
4. Fill in the [PR template](.github/pull_request_template.md).
5. Ensure all CI checks pass.

PRs are squash-merged to keep a clean history.

---

## Adding a New Model Variant

1. Add the architecture class to `src/ml_sgs/models/` (create a new file or extend
   an existing one).
2. Export it from `src/ml_sgs/models/__init__.py`.
3. Add a corresponding training script to `src/ml_sgs/training/`.
4. Register the model in `src/ml_sgs/inference/engine.py` (`add_*_models` method).
5. Add unit tests to `tests/test_models.py`.
6. Update `configs/default.yaml` with any new hyperparameters.
7. Document the variant in `docs/architecture.md`.

---

## Reporting Bugs

Please use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md) and
include a minimal reproducible example.
