"""
ml_sgs.data
===========

MONC NetCDF data processing and feature engineering.

Classes
-------
FastDataLoader          – Loads a single NetCDF snapshot into RAM
FastUnifiedProcessor    – Parallel multi-file processor (standard)
EnhancedProcessor       – Extended processor with additional diagnostics
"""

from ml_sgs.data.processor import FastDataLoader
from ml_sgs.data.enhanced_processor import EnhancedDataProcessor

__all__ = ["FastDataLoader", "EnhancedDataProcessor"]
