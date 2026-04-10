"""
Tests for ml_sgs.inference
===========================

Smoke tests verifying the inference engines can be imported and instantiated.
Full integration tests require model checkpoints and NetCDF data.
"""

import pytest


class TestUnifiedInferenceEngine:
    def test_import(self):
        from ml_sgs.inference import UnifiedInferenceEngine
        assert UnifiedInferenceEngine is not None

    def test_instantiation(self):
        from ml_sgs.inference import UnifiedInferenceEngine
        engine = UnifiedInferenceEngine(device="cpu")
        assert hasattr(engine, "add_baseline_models")
        assert hasattr(engine, "add_ri_models")
        assert hasattr(engine, "predict_all")

    def test_empty_engine_predict_raises(self):
        """Calling predict_all on an empty engine should raise a meaningful error."""
        from ml_sgs.inference import UnifiedInferenceEngine
        from pathlib import Path

        engine = UnifiedInferenceEngine(device="cpu")
        with pytest.raises(Exception):
            engine.predict_all(Path("nonexistent.nc"), time_idx=0, k_min=2, k_max=10)


class TestQ1Q4Engine:
    def test_import(self):
        from ml_sgs.inference import Q1Q4UnifiedEngine
        assert Q1Q4UnifiedEngine is not None
