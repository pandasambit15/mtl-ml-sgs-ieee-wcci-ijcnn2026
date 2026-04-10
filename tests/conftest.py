"""
conftest.py – shared pytest fixtures for ml-sgs-turbulence
===========================================================

Fixtures here are automatically available to all test modules.
"""

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Random-state fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def rng():
    """Seeded NumPy default RNG for reproducible random data."""
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def torch_rng():
    """Set the global PyTorch seed and return the generator."""
    gen = torch.Generator()
    gen.manual_seed(42)
    return gen


# ---------------------------------------------------------------------------
# Common tensor fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def feature_batch():
    """Random feature batch of shape (32, 54) — the standard SGS input."""
    return torch.randn(32, 54)


@pytest.fixture
def small_batch():
    """Tiny batch (4, 54) for fast shape-only checks."""
    return torch.randn(4, 54)


# ---------------------------------------------------------------------------
# Common NumPy array fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def prediction_pair(rng):
    """
    Returns (truth, model_a_preds, model_b_preds) as 1-D float arrays.

    model_a is a good predictor (low noise), model_b is a poor predictor.
    """
    n = 1000
    truth = rng.normal(0.0, 1.0, n).astype(np.float32)
    model_a = truth + rng.normal(0.0, 0.1, n).astype(np.float32)   # good
    model_b = truth + rng.normal(0.0, 1.5, n).astype(np.float32)   # bad
    return truth, model_a, model_b


@pytest.fixture
def perfect_prediction(rng):
    """Returns (truth, perfect_preds) – identical arrays."""
    truth = rng.normal(0.0, 2.0, 500).astype(np.float32)
    return truth, truth.copy()
