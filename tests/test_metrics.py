"""
Tests for ml_sgs.evaluation.metrics
=====================================

Verifies numerical correctness of key skill scores and statistical tests
using analytically known results.
"""

import numpy as np
import pytest

from ml_sgs.evaluation import (
    paired_ttest,
    wilcoxon_test,
    bootstrap_confidence_interval,
    nash_sutcliffe_efficiency,
    kling_gupta_efficiency,
    index_of_agreement,
)


# ---------------------------------------------------------------------------
# Perfect prediction baseline
# ---------------------------------------------------------------------------

@pytest.fixture
def perfect():
    rng = np.random.default_rng(42)
    truth = rng.normal(0, 1, 500)
    return truth, truth.copy()   # (truth, perfect prediction)


@pytest.fixture
def noisy(perfect):
    truth, _ = perfect
    rng = np.random.default_rng(0)
    preds = truth + rng.normal(0, 0.1, len(truth))
    return truth, preds


# ---------------------------------------------------------------------------
# Nash-Sutcliffe Efficiency
# ---------------------------------------------------------------------------

class TestNSE:
    def test_perfect_nse_is_one(self, perfect):
        truth, preds = perfect
        nse = nash_sutcliffe_efficiency(preds, truth)
        assert abs(nse - 1.0) < 1e-9

    def test_nse_is_bounded(self, noisy):
        truth, preds = noisy
        nse = nash_sutcliffe_efficiency(preds, truth)
        assert nse <= 1.0
        assert nse > 0.0   # noisy but close

    def test_mean_predictor_gives_zero(self):
        rng = np.random.default_rng(1)
        truth = rng.normal(5, 2, 300)
        mean_pred = np.full_like(truth, truth.mean())
        nse = nash_sutcliffe_efficiency(mean_pred, truth)
        assert abs(nse) < 1e-9


# ---------------------------------------------------------------------------
# Index of Agreement
# ---------------------------------------------------------------------------

class TestIndexOfAgreement:
    def test_perfect_ioa_is_one(self, perfect):
        truth, preds = perfect
        ioa = index_of_agreement(preds, truth)
        assert abs(ioa - 1.0) < 1e-9

    def test_ioa_range(self, noisy):
        truth, preds = noisy
        ioa = index_of_agreement(preds, truth)
        assert 0.0 <= ioa <= 1.0


# ---------------------------------------------------------------------------
# Paired t-test
# ---------------------------------------------------------------------------

class TestPairedTtest:
    def test_identical_models_not_significant(self):
        rng = np.random.default_rng(7)
        truth = rng.normal(0, 1, 400)
        preds = truth + rng.normal(0, 0.5, 400)
        result = paired_ttest(preds, preds, truth)
        assert not result["significant"], "Identical predictions should not be significantly different"

    def test_result_keys(self):
        rng = np.random.default_rng(9)
        truth = rng.normal(0, 1, 200)
        p1 = truth + rng.normal(0, 0.3, 200)
        p2 = truth + rng.normal(0, 1.5, 200)
        result = paired_ttest(p1, p2, truth)
        for key in ("statistic", "p_value", "significant", "mean_diff", "ci_lower", "ci_upper"):
            assert key in result, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

class TestBootstrapCI:
    def test_ci_contains_true_mean(self):
        rng = np.random.default_rng(3)
        data = rng.normal(5.0, 1.0, 1000)
        lo, hi = bootstrap_confidence_interval(data, n_bootstrap=500, ci=0.95)
        assert lo < 5.0 < hi

    def test_ci_ordered(self):
        rng = np.random.default_rng(4)
        data = rng.exponential(2.0, 500)
        lo, hi = bootstrap_confidence_interval(data, n_bootstrap=300)
        assert lo < hi
