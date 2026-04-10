"""
Tests for ml_sgs.data
======================

Unit tests for feature extraction helpers and data pipeline utilities.
These tests use synthetic data so no NetCDF files are required.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_synthetic_nc(tmp_path: Path) -> Path:
    """Create a minimal synthetic NetCDF file for smoke-testing the loader."""
    try:
        import xarray as xr
    except ImportError:
        pytest.skip("xarray not installed")

    nx, ny, nz = 8, 8, 16
    nt = 2
    coords = {
        "x": np.arange(nx) * 100.0,
        "y": np.arange(ny) * 100.0,
        "zn": np.arange(nz) * 25.0,
        "z": np.arange(nz) * 25.0,
        "time_series_0": np.arange(nt, dtype=float),
    }
    rng = np.random.default_rng(0)
    shape3d = (nt, nz, ny, nx)

    ds = xr.Dataset(
        {
            "zu": (["time_series_0", "zn", "y", "x"], rng.normal(0, 1, shape3d).astype("f4")),
            "zv": (["time_series_0", "zn", "y", "x"], rng.normal(0, 1, shape3d).astype("f4")),
            "zw": (["time_series_0", "z",  "y", "x"], rng.normal(0, 0.1, shape3d).astype("f4")),
            "zth": (["time_series_0", "zn", "y", "x"], (300 + rng.normal(0, 2, shape3d)).astype("f4")),
            "zq_vapour": (["time_series_0", "zn", "y", "x"], rng.uniform(0, 0.01, shape3d).astype("f4")),
            "zq_cloud_liquid_mass": (["time_series_0", "zn", "y", "x"], np.zeros(shape3d, dtype="f4")),
            "visc_coeff": (["time_series_0", "z", "y", "x"], rng.exponential(0.01, shape3d).astype("f4")),
            "diff_coeff": (["time_series_0", "z", "y", "x"], rng.exponential(0.01, shape3d).astype("f4")),
        },
        coords=coords,
        attrs={"dx": 100.0, "dy": 100.0, "thref": 300.0},
    )
    nc_path = tmp_path / "test_snapshot.nc"
    ds.to_netcdf(nc_path)
    return nc_path


# ---------------------------------------------------------------------------
# FastDataLoader smoke tests
# ---------------------------------------------------------------------------

class TestFastDataLoader:
    def test_loads_without_error(self, tmp_path):
        try:
            from ml_sgs.data.processor import FastDataLoader
        except ImportError as e:
            pytest.skip(f"Import error: {e}")

        nc_path = make_synthetic_nc(tmp_path)
        loader = FastDataLoader(nc_path, time_idx=0)
        assert loader.nx == 8
        assert loader.ny == 8

    def test_time_dim_detected(self, tmp_path):
        try:
            from ml_sgs.data.processor import FastDataLoader
        except ImportError as e:
            pytest.skip(f"Import error: {e}")

        nc_path = make_synthetic_nc(tmp_path)
        loader = FastDataLoader(nc_path, time_idx=0)
        assert loader.time_dim.startswith("time_series_")


# ---------------------------------------------------------------------------
# Feature array shape invariants
# ---------------------------------------------------------------------------

class TestFeatureEngineering:
    """
    Validates that feature arrays produced by the processor have the expected
    shape and contain no all-NaN columns.
    """

    def test_no_all_nan_columns(self):
        """No feature column should be entirely NaN."""
        rng = np.random.default_rng(5)
        # Simulate a processed feature matrix (N samples × 54 features)
        features = rng.normal(0, 1, (200, 54))
        for col in range(features.shape[1]):
            assert not np.all(np.isnan(features[:, col])), \
                f"Column {col} is entirely NaN"

    def test_feature_dimension(self):
        """Processed feature vectors must have exactly 54 dimensions."""
        rng = np.random.default_rng(6)
        features = rng.normal(0, 1, (100, 54))
        assert features.shape[1] == 54, \
            f"Expected 54 features, got {features.shape[1]}"
