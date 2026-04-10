#!/usr/bin/env python3
"""
Unified Inference Engine for All ML-SGS Models
==============================================

Provides a consistent interface for:
- Baseline models (MLP, ResMLP, TabTransformer)
- Ri-conditioned models (MLP, ResMLP, TabTransformer)
- Q1-Q4 configurations (all architectures)

Usage:
    engine = UnifiedInferenceEngine()
    engine.add_baseline_models(baseline_paths, scaler_dir)
    engine.add_ri_models(ri_paths, scaler_dir)
    engine.add_q1q4_models(q1q4_paths, config, scaler_dir)
    
    predictions = engine.predict_all(nc_file, time_idx, k_min, k_max)
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging
from collections import defaultdict

# Import existing engines
from run_best_models_analysis import BestModelsInferenceEngine
from run_models_comparison_with_ri_v2 import UnifiedInferenceEngine as RiEngine
from q1_q4_unified_engine import Q1Q4UnifiedEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnifiedInferenceEngine:
    """
    Universal inference engine wrapping all model types.
    
    Provides consistent interface and output format for:
    - Baseline models
    - Richardson-conditioned models  
    - Q1-Q4 configurations
    """
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.engines = {}
        self.model_registry = {}  # Maps model_name -> (engine_type, engine_key)
        
        logger.info("\n" + "="*80)
        logger.info("UNIFIED INFERENCE ENGINE INITIALIZED")
        logger.info("="*80)
    
    def add_baseline_models(self, model_paths: Dict[str, Path], 
                           scaler_dir: Path, n_workers: Optional[int] = None):
        """
        Add baseline models (no Richardson conditioning).
        
        Parameters
        ----------
        model_paths : Dict[str, Path]
            {'MLP': path, 'ResMLP': path, 'TabTransformer': path}
        scaler_dir : Path
            Directory with 54-feature scalers
        """
        if not model_paths:
            return
        
        logger.info("\nAdding Baseline Models...")
        
        self.engines['baseline'] = BestModelsInferenceEngine(
            model_paths=model_paths,
            scaler_dir=scaler_dir,
            n_workers=n_workers
        )
        
        for arch in model_paths.keys():
            model_name = f"Baseline-{arch}"
            self.model_registry[model_name] = ('baseline', arch)
            logger.info(f"  ✓ Registered: {model_name}")
    
    def add_ri_models(self, model_paths: Dict[str, Path], 
                      scaler_dir: Path, n_workers: Optional[int] = None):
        """
        Add Richardson-conditioned models.
        
        Parameters
        ----------
        model_paths : Dict[str, Path]
            {'MLP': path, 'ResMLP': path, 'TabTransformer': path}
        scaler_dir : Path
            Directory with 54-feature scalers
        """
        if not model_paths:
            return
        
        logger.info("\nAdding Ri-Conditioned Models...")
        
        self.engines['ri'] = RiEngine(
            baseline_paths=None,
            ri_paths=model_paths,
            scaler_dir=scaler_dir,
            n_workers=n_workers
        )
        
        for arch in model_paths.keys():
            model_name = f"Ri-{arch}"
            self.model_registry[model_name] = ('ri', arch)
            logger.info(f"  ✓ Registered: {model_name}")
    
    def add_q1q4_models(self, model_paths: Dict[str, Dict[str, Path]], 
                        config: Dict, scaler_dir: Path, 
                        n_workers: Optional[int] = None):
        """
        Add Q1-Q4 configuration models.
        
        Parameters
        ----------
        model_paths : Dict[str, Dict[str, Path]]
            Nested: {'Q1': {'MLP': path, ...}, 'Q2': {...}, ...}
        config : Dict
            Configuration dictionary
        scaler_dir : Path
            Directory with scalers
        """
        if not model_paths:
            return
        
        logger.info("\nAdding Q1-Q4 Models...")
        
        self.engines['q1q4'] = Q1Q4UnifiedEngine(
            model_paths=model_paths,
            config=config,
            scaler_dir=scaler_dir,
            n_workers=n_workers,
            device=self.device
        )
        
        for config_name in model_paths.keys():
            for arch in model_paths[config_name].keys():
                model_name = f"{config_name}-{arch}"
                self.model_registry[model_name] = ('q1q4', model_name)
                logger.info(f"  ✓ Registered: {model_name}")
    
    def get_all_model_names(self) -> List[str]:
        """Get list of all registered model names."""
        return sorted(self.model_registry.keys())
    
    def get_models_by_architecture(self) -> Dict[str, List[str]]:
        """Group models by architecture type."""
        by_arch = defaultdict(list)
        
        for model_name in self.model_registry.keys():
            # Extract architecture (last component)
            arch = model_name.split('-')[-1]
            by_arch[arch].append(model_name)
        
        return dict(by_arch)
    
    def get_models_by_configuration(self) -> Dict[str, List[str]]:
        """Group models by configuration type."""
        by_config = defaultdict(list)
        
        for model_name in self.model_registry.keys():
            # Extract configuration (first component)
            config = model_name.split('-')[0]
            by_config[config].append(model_name)
        
        return dict(by_config)
    
    @torch.no_grad()
    def predict_all(self, nc_file: Path, time_idx: int = 0,
                    k_min: int = 0, k_max: int = 219,
                    models: Optional[List[str]] = None) -> Dict[str, Dict]:
        """
        Run inference for all (or specified) models.
        
        Parameters
        ----------
        nc_file : Path
            NetCDF input file
        time_idx : int
            Time index within file
        k_min, k_max : int
            Vertical level range
        models : Optional[List[str]]
            Specific models to run (None = all)
        
        Returns
        -------
        predictions : Dict[str, Dict]
            Nested dict: {
                'Baseline-MLP': {'visc_coeff': array, 'diff_coeff': array, ...},
                'Ri-MLP': {...},
                'Q1-MLP': {...},
                ...
            }
        """
        if models is None:
            models = self.get_all_model_names()
        
        logger.info(f"\nRunning inference on {len(models)} models...")
        logger.info(f"File: {nc_file.name}")
        
        all_predictions = {}
        
        # Group models by engine type for efficiency
        by_engine = defaultdict(list)
        for model_name in models:
            engine_type, engine_key = self.model_registry[model_name]
            by_engine[engine_type].append((model_name, engine_key))
        
        # Run each engine once
        for engine_type, model_list in by_engine.items():
            logger.info(f"\n  Processing {engine_type} models...")
            
            if engine_type == 'baseline':
                raw_preds = self.engines['baseline'].predict_3d_domain(
                    nc_file, time_idx, k_min, k_max
                )
                
                for model_name, arch in model_list:
                    all_predictions[model_name] = raw_preds[arch]
            
            elif engine_type == 'ri':
                raw_preds = self.engines['ri'].predict_3d_domain(
                    nc_file, time_idx, k_min, k_max
                )
                
                for model_name, arch in model_list:
                    # Ri engine returns with 'Ri-' prefix
                    all_predictions[model_name] = raw_preds[f'Ri-{arch}']
            
            elif engine_type == 'q1q4':
                raw_preds = self.engines['q1q4'].predict_3d_domain(
                    nc_file, time_idx, k_min, k_max
                )
                
                for model_name, q1q4_key in model_list:
                    all_predictions[model_name] = raw_preds[q1q4_key]
        
        logger.info(f"\n✓ Inference complete for {len(all_predictions)} models")
        
        return all_predictions
    
    def get_model_info(self, model_name: str) -> Dict:
        """Get metadata about a model."""
        if model_name not in self.model_registry:
            raise ValueError(f"Unknown model: {model_name}")
        
        engine_type, engine_key = self.model_registry[model_name]
        
        parts = model_name.split('-')
        
        return {
            'full_name': model_name,
            'engine_type': engine_type,
            'configuration': parts[0],
            'architecture': parts[-1],
            'has_richardson': engine_type in ['ri', 'q1q4'] and parts[0] != 'Q4',
            'has_regime': engine_type == 'q1q4' and parts[0] != 'Q4'
        }
    
    def summary(self):
        """Print summary of registered models."""
        print("\n" + "="*80)
        print("UNIFIED ENGINE SUMMARY")
        print("="*80)
        
        print(f"\nTotal Models: {len(self.model_registry)}")
        
        print("\nBy Engine Type:")
        by_type = defaultdict(list)
        for name in self.model_registry.keys():
            engine_type, _ = self.model_registry[name]
            by_type[engine_type].append(name)
        
        for engine_type, models in sorted(by_type.items()):
            print(f"  {engine_type}: {len(models)} models")
            for model in sorted(models):
                print(f"    - {model}")
        
        print("\nBy Architecture:")
        by_arch = self.get_models_by_architecture()
        for arch, models in sorted(by_arch.items()):
            print(f"  {arch}: {len(models)} models")
        
        print("="*80 + "\n")


# ==================== TESTING ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Unified Engine')
    parser.add_argument('--baseline-mlp', type=Path)
    parser.add_argument('--ri-mlp', type=Path)
    parser.add_argument('--q1-mlp', type=Path)
    parser.add_argument('--scaler-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--nc-file', type=Path, required=True)
    
    args = parser.parse_args()
    
    # Initialize engine
    engine = UnifiedInferenceEngine()
    
    # Add models
    if args.baseline_mlp:
        engine.add_baseline_models({'MLP': args.baseline_mlp}, args.scaler_dir)
    
    if args.ri_mlp:
        engine.add_ri_models({'MLP': args.ri_mlp}, args.scaler_dir)
    
    if args.q1_mlp and args.config:
        import yaml
        with open(args.config) as f:
            config = yaml.safe_load(f)
        engine.add_q1q4_models({'Q1': {'MLP': args.q1_mlp}}, config, args.scaler_dir)
    
    # Show summary
    engine.summary()
    
    # Test inference
    predictions = engine.predict_all(args.nc_file, k_min=0, k_max=10)
    
    print("\nPrediction Results:")
    for model_name, preds in predictions.items():
        print(f"\n{model_name}:")
        for key, value in preds.items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: shape={value.shape}, mean={value.mean():.6f}")
