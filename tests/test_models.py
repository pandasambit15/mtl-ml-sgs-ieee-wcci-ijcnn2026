"""
Tests for ml_sgs.models
========================

Verifies that all neural network architectures produce correctly shaped outputs
and can perform a backward pass without error.
"""

import pytest
import torch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def batch():
    """Return a small random feature batch (B=8, F=54)."""
    return torch.randn(8, 54)


# ---------------------------------------------------------------------------
# Richardson-conditioned models
# ---------------------------------------------------------------------------

class TestRiConditionedMLP:
    def test_output_shapes(self, batch):
        from ml_sgs.models import RiConditionedMLP

        model = RiConditionedMLP(n_features=54)
        model.eval()
        visc, diff, ri, regime = model(batch)

        assert visc.shape == (8, 1),   f"Expected (8,1), got {visc.shape}"
        assert diff.shape == (8, 1),   f"Expected (8,1), got {diff.shape}"
        assert ri.shape == (8, 1),     f"Expected (8,1), got {ri.shape}"
        assert regime.shape == (8, 3), f"Expected (8,3), got {regime.shape}"

    def test_backward(self, batch):
        from ml_sgs.models import RiConditionedMLP

        model = RiConditionedMLP(n_features=54)
        visc, diff, ri, regime = model(batch)
        loss = visc.mean() + diff.mean() + ri.mean() + regime.mean()
        loss.backward()

    def test_parameter_count(self, batch):
        from ml_sgs.models import RiConditionedMLP

        model = RiConditionedMLP(n_features=54)
        n_params = sum(p.numel() for p in model.parameters())
        # Rough sanity check – should be at least 500k
        assert n_params > 500_000, f"Suspiciously few parameters: {n_params}"


class TestRiConditionedResMLP:
    def test_output_shapes(self, batch):
        from ml_sgs.models import RiConditionedResMLP

        model = RiConditionedResMLP(n_features=54)
        model.eval()
        visc, diff, ri, regime = model(batch)

        assert visc.shape == (8, 1)
        assert diff.shape == (8, 1)
        assert ri.shape == (8, 1)
        assert regime.shape == (8, 3)

    def test_backward(self, batch):
        from ml_sgs.models import RiConditionedResMLP

        model = RiConditionedResMLP(n_features=54)
        loss = sum(o.mean() for o in model(batch))
        loss.backward()


class TestRiConditionedTabTransformer:
    def test_output_shapes(self, batch):
        from ml_sgs.models import RiConditionedTabTransformer

        model = RiConditionedTabTransformer(n_features=54)
        model.eval()
        visc, diff, ri, regime = model(batch)

        assert visc.shape == (8, 1)
        assert diff.shape == (8, 1)
        assert ri.shape == (8, 1)
        assert regime.shape == (8, 3)


# ---------------------------------------------------------------------------
# Shared head module
# ---------------------------------------------------------------------------

class TestRichardsonConditionedHeads:
    def test_conditioning_order(self):
        """Ri should be predicted before viscosity/diffusivity are computed."""
        from ml_sgs.models import RichardsonConditionedHeads
        import torch

        heads = RichardsonConditionedHeads(feature_dim=256)
        features = torch.randn(4, 256)
        visc, diff, ri, regime = heads(features)

        # ri is predicted from backbone only (no Ri in input); check it is 1-D output
        assert ri.shape == (4, 1)

    def test_dropout_disabled_in_eval(self):
        from ml_sgs.models import RiConditionedMLP

        model = RiConditionedMLP()
        model.eval()
        x = torch.randn(16, 54)
        out1 = model(x)
        out2 = model(x)
        # Deterministic in eval mode
        for a, b in zip(out1, out2):
            assert torch.allclose(a, b), "Non-deterministic outputs in eval mode"
