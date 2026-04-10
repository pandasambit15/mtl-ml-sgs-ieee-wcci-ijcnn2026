#!/usr/bin/env python3
"""
Q1-Q4 Unified Inference Engine
===============================

Multi-model inference engine that loads ALL Q1-Q4 models simultaneously
and provides unified interface for batch predictions.

Usage:
    model_paths = {
        'Q1': {'MLP': 'q1_mlp.pt', 'ResMLP': 'q1_resmlp.pt'},
        'Q2': {'MLP': 'q2_mlp.pt'},
        'Q4': {'MLP': 'q4_mlp.pt', 'TabTransformer': 'q4_tabtrans.pt'}
    }
    
    engine = Q1Q4UnifiedEngine(
        model_paths=model_paths,
        config=config_dict,
        scaler_dir='scalers/'
    )
    
    # Get all model names
    models = engine.get_model_names()  # ['Q1-MLP', 'Q1-ResMLP', 'Q2-MLP', ...]
    
    # Run inference on all models
    predictions = engine.predict_3d_domain('data.nc', time_idx=0, k_min=0, k_max=219)
"""

import torch
import torch.nn as nn
import numpy as np
import netCDF4 as nc
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import joblib
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Import model creation functions
from train_q1_models import create_q1_model
from train_q2_models import create_q2_model
from train_q3_models import create_q3_model
from train_q4_models import create_q4_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Q1Q4UnifiedEngine:
    """
    Multi-model inference engine for Q1-Q4 configurations.
    
    Loads all specified models and provides unified batch inference.
    """
    
    def __init__(self, 
                 model_paths: Dict[str, Dict[str, Path]],
                 config: Dict,
                 scaler_dir: Path,
                 n_workers: Optional[int] = None,
                 device: str = 'cuda'):
        """
        Initialize the unified engine with multiple models.
        
        Args:
            model_paths: Nested dict: {'Q1': {'MLP': path, 'ResMLP': path}, 'Q2': {...}}
            config: Configuration dictionary
            scaler_dir: Directory containing scaler files
            n_workers: Number of parallel workers (None = no parallelization)
            device: Device for inference ('cuda' or 'cpu')
        """
        self.model_paths = model_paths
        self.config = config
        self.scaler_dir = Path(scaler_dir)
        self.n_workers = n_workers
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load scalers
        self.scalers = self._load_scalers()
        
        # Load all models
        self.models = {}
        self._load_all_models()
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Q1-Q4 Unified Engine Initialized")
        logger.info(f"{'='*70}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Models loaded: {len(self.models)}")
        logger.info(f"Model names: {', '.join(self.get_model_names())}")
        logger.info(f"{'='*70}\n")
    
    def _load_scalers(self) -> Dict:
        """Load feature and target scalers."""
        scaler_files = {
            'features': self.scaler_dir / 'feature_scaler.pkl',
            'visc': self.scaler_dir / 'visc_scaler.pkl',
            'diff': self.scaler_dir / 'diff_scaler.pkl',
            'ri': self.scaler_dir / 'richardson_scaler.pkl'
        }
        
        scalers = {}
        for name, path in scaler_files.items():
            if path.exists():
                scalers[name] = joblib.load(path)  # Changed to joblib
                logger.info(f"Loaded scaler: {path.name}")
            else:
                logger.warning(f"Scaler not found: {path}")
        
        return scalers
    
    def _load_all_models(self):
        """Load all models from model_paths."""
        for config_name, arch_dict in self.model_paths.items():
            for arch_name, model_path in arch_dict.items():
                model_key = f"{config_name}-{arch_name}"
                
                logger.info(f"Loading {model_key} from {model_path}")
                
                try:
                    # Create model
                    if config_name == 'Q1':
                        model = create_q1_model(arch_name)
                    elif config_name == 'Q2':
                        model = create_q2_model(arch_name)
                    elif config_name == 'Q3':
                        model = create_q3_model(arch_name)
                        model.set_stage(3)  # Full pipeline
                    elif config_name == 'Q4':
                        model = create_q4_model(arch_name)
                    else:
                        raise ValueError(f"Unknown config: {config_name}")
                    
                    # Load checkpoint
                    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                    if 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        model.load_state_dict(checkpoint)
                    
                    model = model.to(self.device)
                    model.eval()
                    
                    self.models[model_key] = {
                        'model': model,
                        'config': config_name,
                        'architecture': arch_name
                    }
                    
                    logger.info(f"  ✓ Loaded successfully")
                
                except Exception as e:
                    logger.error(f"  ✗ Failed to load {model_key}: {e}")
                    raise
    
    def get_model_names(self) -> List[str]:
        """Get list of all loaded model names."""
        return sorted(self.models.keys())
    
    def _extract_54_features_from_netcdf(self, nc_file: Path, time_idx: int,
                                        k_min: int, k_max: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Extract 54 features matching the training pipeline exactly.
        
        Features (54 total):
        - Local values (6): zu, zv, zw, zth, zq_vapour, zq_cloud
        - Spatial neighbors 5 vars × 6 directions (30)
        - W-field neighbors (6)
        - Grid parameters (5): dx, dy, dz, height, thref
        - Normalized position (7)
        """
        import xarray as xr
        
        ds = xr.open_dataset(nc_file).load()
        
        # Find time dimension
        time_dims = [dim for dim in ds.dims if dim.startswith('time_series_')]
        time_dim = time_dims[0] if time_dims else 'time'
        
        nx, ny = ds.sizes['x'], ds.sizes['y']
        nz_zn, nz_z = ds.sizes['zn'], ds.sizes['z']
        
        # Load variables
        def load_var(var_name, coord):
            for name in [var_name, var_name[1:] if var_name.startswith('z') else 'z' + var_name]:
                if name in ds:
                    var = ds[name]
                    if time_dim in var.dims and coord in var.dims:
                        return var.isel({time_dim: time_idx}).values
            shape = (nx, ny, nz_zn if coord == 'zn' else nz_z)
            logger.warning(f"Variable '{var_name}' not found. Using zeros.")
            return np.zeros(shape, dtype=np.float32)
        
        zu = load_var('zu', 'zn')
        zv = load_var('zv', 'zn')
        zth = load_var('zth', 'zn')
        zq_vapour = load_var('zq_vapour', 'zn')
        zq_cloud = load_var('zq_cloud_liquid_mass', 'zn')
        zw = load_var('zw', 'z')
        
        # Grid metadata
        heights = ds['zn'].values if 'zn' in ds.coords else np.arange(nz_zn) * 50.0
        dx = float(ds['x_resolution'].isel({time_dim: time_idx}).values) if 'x_resolution' in ds else 100.0
        dy = float(ds['y_resolution'].isel({time_dim: time_idx}).values) if 'y_resolution' in ds else 100.0
        thref = float(ds.attrs['thref']) if 'thref' in ds.attrs else 300.0
        
        # Clamp k_max to valid range
        k_max_actual = min(k_max, nz_zn - 1)
        
        # Extract features for all points in slice
        all_features = []
        
        for k in range(k_min, k_max_actual + 1):
            for j in range(ny):
                for i in range(nx):
                    features = []
                    
                    # Neighbor indices
                    i_p, i_m = (i + 1) % nx, (i - 1) % nx
                    j_p, j_m = (j + 1) % ny, (j - 1) % ny
                    k_p, k_m = min(k + 1, nz_zn - 1), max(k - 1, 0)
                    
                    # 1. Local values (6)
                    features.extend([
                        zu[i, j, k],
                        zv[i, j, k],
                        zw[i, j, min(k, nz_z - 1)],
                        zth[i, j, k],
                        zq_vapour[i, j, k],
                        zq_cloud[i, j, k]
                    ])
                    
                    # 2. Spatial neighbors (30): 5 vars × 6 directions
                    for var in [zu, zv, zth, zq_vapour, zq_cloud]:
                        features.extend([
                            var[i_p, j, k], var[i_m, j, k],
                            var[i, j_p, k], var[i, j_m, k],
                            var[i, j, k_p], var[i, j, k_m]
                        ])
                    
                    # 3. W-field neighbors (6)
                    k_z = min(k, nz_z - 1)
                    k_pz = min(k_p, nz_z - 1)
                    k_mz = min(k_m, nz_z - 1)
                    features.extend([
                        zw[i_p, j, k_z], zw[i_m, j, k_z],
                        zw[i, j_p, k_z], zw[i, j_m, k_z],
                        zw[i, j, k_pz], zw[i, j, k_mz]
                    ])
                    
                    # 4. Grid parameters (5)
                    height = heights[k]
                    dz = heights[k_p] - height if k < nz_zn - 1 else height - heights[k_m]
                    features.extend([dx, dy, dz, height, thref])
                    
                    # 5. Normalized position (7)
                    z_max = heights[-1]
                    features.extend([
                        k / nz_zn,
                        height,
                        z_max - height,
                        i / nx,
                        j / ny,
                        min(i, nx - 1 - i) / nx,
                        min(j, ny - 1 - j) / ny
                    ])
                    
                    all_features.append(features)
        
        # Shape: (nz, ny, nx, 54)
        nz_actual = k_max_actual - k_min + 1
        features_array = np.array(all_features, dtype=np.float32).reshape(nz_actual, ny, nx, 54)
        
        # Load targets
        targets = {}
        if 'visc_coeff' in ds:
            visc = ds['visc_coeff'].isel({time_dim: time_idx}).values
            targets['visc_coeff'] = np.transpose(visc[:, :, k_min:k_max_actual+1], (2, 1, 0))
        
        if 'diff_coeff' in ds:
            diff = ds['diff_coeff'].isel({time_dim: time_idx}).values
            targets['diff_coeff'] = np.transpose(diff[:, :, k_min:k_max_actual+1], (2, 1, 0))
        
        if 'ri_smag' in ds:
            ri = ds['ri_smag'].isel({time_dim: time_idx}).values
            targets['ri'] = np.transpose(ri[:, :, k_min:k_max_actual+1], (2, 1, 0))
        
        ds.close()
        
        return features_array, targets
    
    def _load_netcdf_slice(self, nc_file: Path, time_idx: int, 
                          k_min: int, k_max: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Load NetCDF and extract 54 features - wrapper for the main extraction."""
        return self._extract_54_features_from_netcdf(nc_file, time_idx, k_min, k_max)
    
    def _parse_model_output(self, model_output: Tuple, config: str) -> Dict[str, np.ndarray]:
        """Parse model output tuple into dictionary."""
        visc, diff, ri, regime = model_output
        
        predictions = {}
        
        # All configs predict viscosity and diffusivity
        if visc is not None:
            predictions['visc_coeff'] = visc.cpu().numpy()
        if diff is not None:
            predictions['diff_coeff'] = diff.cpu().numpy()
        
        # Q1, Q2, Q3 also predict richardson and regime
        if config != 'Q4':
            if ri is not None:
                if ri.dim() > 1:
                    predictions['ri'] = ri.squeeze(-1).cpu().numpy()
                else:
                    predictions['ri'] = ri.cpu().numpy()
            
            if regime is not None:
                predictions['regime_class'] = regime.argmax(dim=1).cpu().numpy()
                predictions['regime_logits'] = regime.cpu().numpy()
        
        return predictions
    
    @torch.no_grad()
    #@torch.no_grad()
    def predict_3d_domain(self, nc_file: Path, time_idx: int = 0,
                         k_min: int = 0, k_max: int = 219) -> Dict[str, Dict[str, np.ndarray]]:
        """Run inference sequentially to save memory."""
    
        # Load data ONCE
        features, targets = self._load_netcdf_slice(nc_file, time_idx, k_min, k_max)
        nz, ny, nx, n_features = features.shape
    
        # Scale features ONCE
        if 'features' in self.scalers:
            features_flat = features.reshape(-1, n_features)
            features_scaled = self.scalers['features'].transform(features_flat)
            features_scaled = features_scaled.reshape(nz, ny, nx, n_features)
        else:
            features_scaled = features
    
        # Convert to tensor ONCE (keep on CPU initially)
        x_cpu = torch.from_numpy(features_scaled.reshape(-1, n_features)).float()
    
        all_predictions = {'shared': {'features': features}}
    
        # Process each model sequentially
        for model_name, model_info in self.models.items():
            model = model_info['model']
            config = model_info['config']
        
            # Move model to GPU
            model = model.to(self.device)
        
            # Run inference in batches
            batch_size = 8192  # Adjust based on your GPU
            preds_list = {}  # ✅ Use empty dict, will populate dynamically
        
            for i in range(0, len(x_cpu), batch_size):
                batch_cpu = x_cpu[i:i+batch_size]
                batch_gpu = batch_cpu.to(self.device)
            
                model_output = model(batch_gpu)
            
                # Parse and store (move to CPU to free GPU memory)
                preds_batch = self._parse_model_output(model_output, config)
            
                # ✅ Dynamically append to lists
                for key, value in preds_batch.items():
                    if value is not None:
                        if key not in preds_list:
                            preds_list[key] = []
                        preds_list[key].append(value)
            
                # Clear batch from GPU
                del batch_gpu
        
            # Concatenate batches
            preds_flat = {}
            for key, values in preds_list.items():
                if values:
                    preds_flat[key] = np.concatenate(values)
        
            # Reshape and inverse scale
            preds_3d = self._reshape_and_inverse_scale(
                preds_flat, nz, ny, nx, config
            )
        
            all_predictions[model_name] = preds_3d
        
            # Free GPU memory
            model = model.cpu()
            torch.cuda.empty_cache()
    
        return all_predictions


    def _reshape_and_inverse_scale(self, preds_flat: Dict, nz: int, ny: int, 
                                nx: int, config: str) -> Dict:
        """Helper to reshape and inverse scale predictions."""
        preds_3d = {}
    
        for key, value in preds_flat.items():
            if value is None:
                continue
        
            # Handle different output types
            if key in ['visc_coeff', 'diff_coeff', 'ri', 'regime_class']:
                # Single-channel outputs
                if value.ndim == 1 or (value.ndim == 2 and value.shape[1] == 1):
                    if value.ndim == 2:
                        value = value.squeeze(-1)
                    # Reshape (nz, ny, nx) and transpose to (nx, ny, nz)
                    preds_3d[key] = value.reshape(nz, ny, nx).transpose(2, 1, 0)
        
            elif key == 'regime_logits':
                # Multi-channel output (3 classes)
                n_classes = value.shape[1]
                # Reshape and transpose
                preds_3d[key] = value.reshape(nz, ny, nx, n_classes).transpose(2, 1, 0, 3)
    
        # Inverse scale coefficients
        if 'visc_coeff' in preds_3d and 'visc' in self.scalers:
            visc_flat = preds_3d['visc_coeff'].flatten().reshape(-1, 1)
            visc_unscaled = self.scalers['visc'].inverse_transform(visc_flat)
            preds_3d['visc_coeff'] = visc_unscaled.reshape(nx, ny, nz)
    
        if 'diff_coeff' in preds_3d and 'diff' in self.scalers:
            diff_flat = preds_3d['diff_coeff'].flatten().reshape(-1, 1)
            diff_unscaled = self.scalers['diff'].inverse_transform(diff_flat)
            preds_3d['diff_coeff'] = diff_unscaled.reshape(nx, ny, nz)
    
        if 'ri' in preds_3d and 'ri' in self.scalers:
            ri_flat = preds_3d['ri'].flatten().reshape(-1, 1)
            ri_unscaled = self.scalers['ri'].inverse_transform(ri_flat)
            preds_3d['ri'] = ri_unscaled.reshape(nx, ny, nz)
    
        return preds_3d


    def predict_3d_domain_old(self, nc_file: Path, time_idx: int = 0,
                         k_min: int = 0, k_max: int = 219) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Run inference on full 3D domain for ALL models.
        
        Args:
            nc_file: Path to NetCDF file
            time_idx: Time index to use
            k_min: Minimum vertical level
            k_max: Maximum vertical level
            
        Returns:
            Nested dict: {
                'Q1-MLP': {'visc_coeff': array, 'diff_coeff': array, ...},
                'Q2-MLP': {...},
                'shared': {'features': array}  # Shared input features
            }
        """
        # Load data
        features, targets = self._load_netcdf_slice(nc_file, time_idx, k_min, k_max)
        
        # features shape: (nz, ny, nx, n_features)
        nz, ny, nx, n_features = features.shape
        
        # Scale features
        if 'features' in self.scalers:
            features_flat = features.reshape(-1, n_features)
            features_scaled = self.scalers['features'].transform(features_flat)
            features_scaled = features_scaled.reshape(nz, ny, nx, n_features)
        else:
            features_scaled = features
        
        # Convert to tensor and flatten spatial dimensions
        # Shape: (nz*ny*nx, n_features)
        x = torch.from_numpy(features_scaled.reshape(-1, n_features)).float().to(self.device)
        
        # Run inference for all models
        all_predictions = {'shared': {'features': features}}
        
        for model_name, model_info in self.models.items():
            model = model_info['model']
            config = model_info['config']
            
            # Run model
            model_output = model(x)
            
            # Parse output
            preds_flat = self._parse_model_output(model_output, config)
            
            # Reshape back to 3D
            preds_3d = {}
            for key, value in preds_flat.items():
                if value.ndim == 1:
                    # Reshape and transpose to match truth format (nx, ny, nz)
                    preds_3d[key] = value.reshape(nz, ny, nx).transpose(2, 1, 0) # (nz,ny,nx) -> (nx,ny,nz) 
                elif value.ndim == 2:
                    # For multi-dimensional outputs (e.g., regime_logits)
                    n_classes = value.shape[1]
                    preds_3d[key] = value.reshape(nz, ny, nx, n_classes).transpose(2, 1, 0, 3)  # (nz,ny,nx,classes) -> (nx,ny,nz,classes)
            
            # Inverse scale coefficients
            if 'visc_coeff' in preds_3d and 'visc' in self.scalers:
                visc_flat = preds_3d['visc_coeff'].flatten().reshape(-1, 1)
                visc_unscaled = self.scalers['visc'].inverse_transform(visc_flat)
                preds_3d['visc_coeff'] = visc_unscaled.reshape(nx, ny, nz)  # Changed from (nz, ny, nx) visc_unscaled.reshape(nz, ny, nx)
            
            if 'diff_coeff' in preds_3d and 'diff' in self.scalers:
                diff_flat = preds_3d['diff_coeff'].flatten().reshape(-1, 1)
                diff_unscaled = self.scalers['diff'].inverse_transform(diff_flat)
                preds_3d['diff_coeff'] = diff_unscaled.reshape(nx, ny, nz)  # Changed from (nz, ny, nx) diff_unscaled.reshape(nz, ny, nx)
            
            if 'ri' in preds_3d and 'ri' in self.scalers:
                ri_flat = preds_3d['ri'].flatten().reshape(-1, 1)
                ri_unscaled = self.scalers['ri'].inverse_transform(ri_flat)
                preds_3d['ri'] = ri_unscaled.reshape(nx, ny, nz)  # Changed from (nz, ny, nx) ri_unscaled.reshape(nz, ny, nx)
            
            all_predictions[model_name] = preds_3d
        
        return all_predictions
    
    def predict_batch_files(self, nc_files: List[Path], time_idx: int = 0,
                           k_min: int = 0, k_max: int = 219,
                           show_progress: bool = True) -> Dict[str, List[Dict]]:
        """
        Run inference on multiple files with optional parallelization.
        
        Returns:
            Dict mapping model names to lists of predictions for each file
        """
        results = {name: [] for name in self.get_model_names()}
        
        if self.n_workers and self.n_workers > 1:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                futures = {
                    executor.submit(self.predict_3d_domain, nc_file, time_idx, k_min, k_max): nc_file
                    for nc_file in nc_files
                }
                
                iterator = as_completed(futures)
                if show_progress:
                    iterator = tqdm(iterator, total=len(nc_files), desc="Processing files")
                
                for future in iterator:
                    try:
                        predictions = future.result()
                        for model_name in self.get_model_names():
                            results[model_name].append(predictions[model_name])
                    except Exception as e:
                        nc_file = futures[future]
                        logger.error(f"Error processing {nc_file}: {e}")
        else:
            # Sequential processing
            iterator = nc_files
            if show_progress:
                iterator = tqdm(nc_files, desc="Processing files")
            
            for nc_file in iterator:
                try:
                    predictions = self.predict_3d_domain(nc_file, time_idx, k_min, k_max)
                    for model_name in self.get_model_names():
                        results[model_name].append(predictions[model_name])
                except Exception as e:
                    logger.error(f"Error processing {nc_file}: {e}")
        
        return results


# ==================== UTILITY FUNCTIONS ====================

def aggregate_predictions(predictions_list: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """
    Aggregate predictions from multiple files.
    
    Args:
        predictions_list: List of prediction dicts from different files
        
    Returns:
        Aggregated predictions with flattened arrays
    """
    aggregated = {}
    
    if not predictions_list:
        return aggregated
    
    # Get all keys from first prediction
    keys = predictions_list[0].keys()
    
    for key in keys:
        arrays = [pred[key].flatten() for pred in predictions_list if key in pred]
        if arrays:
            aggregated[key] = np.concatenate(arrays)
    
    return aggregated


# ==================== TESTING ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Q1-Q4 Unified Engine')
    parser.add_argument('--q1-mlp', type=Path, help='Q1 MLP model path')
    parser.add_argument('--q2-mlp', type=Path, help='Q2 MLP model path')
    parser.add_argument('--q3-mlp', type=Path, help='Q3 MLP model path')
    parser.add_argument('--q4-mlp', type=Path, help='Q4 MLP model path')
    parser.add_argument('--config', type=Path, required=True, help='Config YAML')
    parser.add_argument('--scaler-dir', type=Path, required=True, help='Scaler directory')
    parser.add_argument('--nc-file', type=Path, required=True, help='NetCDF file for testing')
    
    args = parser.parse_args()
    
    # Build model paths
    model_paths = {}
    if args.q1_mlp:
        model_paths['Q1'] = {'MLP': args.q1_mlp}
    if args.q2_mlp:
        model_paths['Q2'] = {'MLP': args.q2_mlp}
    if args.q3_mlp:
        model_paths['Q3'] = {'MLP': args.q3_mlp}
    if args.q4_mlp:
        model_paths['Q4'] = {'MLP': args.q4_mlp}
    
    if not model_paths:
        parser.error("Must provide at least one model")
    
    # Load config
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("\n" + "="*70)
    print("Testing Q1-Q4 Unified Engine")
    print("="*70 + "\n")
    
    # Initialize engine
    engine = Q1Q4UnifiedEngine(
        model_paths=model_paths,
        config=config,
        scaler_dir=args.scaler_dir,
        device='cuda'
    )
    
    # Run prediction
    print(f"\nRunning inference on {args.nc_file}...")
    predictions = engine.predict_3d_domain(args.nc_file, time_idx=0, k_min=0, k_max=10)
    
    print("\nPrediction results:")
    for model_name, preds in predictions.items():
        if model_name == 'shared':
            continue
        print(f"\n{model_name}:")
        for key, value in preds.items():
            print(f"  {key}: shape={value.shape}, mean={value.mean():.6f}, std={value.std():.6f}")
    
    print("\n✅ Engine test completed successfully!")
