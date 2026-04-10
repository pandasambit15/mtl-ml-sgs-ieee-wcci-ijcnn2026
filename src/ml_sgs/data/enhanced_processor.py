#!/usr/bin/env python3
"""
Enhanced MONC Data Processor with Multiple Sampling Strategies
==============================================================

NEW FEATURES:
✅ Regular grid subsampling (original)
✅ Random sampling with fixed sample count
✅ Stratified sampling by Richardson number regime
✅ Safety fallbacks for imbalanced regimes
✅ Automatic regime balancing

Sampling Modes:
1. 'regular': Regular grid subsampling (--subsample N)
2. 'random': Random sampling (--n-samples N)
3. 'stratified': Stratified by regime (--n-samples-per-regime N)
"""

import numpy as np
import xarray as xr
import torch
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter
from tqdm import tqdm
import json
import logging
import gc
from sklearn.preprocessing import StandardScaler, RobustScaler
import joblib
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedMONCDataProcessor:
    """
    Enhanced MONC data processor with multiple sampling strategies
    """
    
    def __init__(self, config: Dict):
        """
        Initialize processor
        
        Args:
            config: Configuration dictionary with keys:
                - halo_size: Number of halo cells (default: 2)
                - chunk_size: Processing chunk size
                - use_robust_scaling: Use RobustScaler
                - sampling_mode: 'regular', 'random', or 'stratified'
                - random_seed: Random seed for reproducibility
        """
        self.config = config
        self.halo_size = config.get('halo_size', 2)
        self.chunk_size = config.get('chunk_size', 100000)
        self.use_robust_scaling = config.get('use_robust_scaling', False)
        self.sampling_mode = config.get('sampling_mode', 'regular')
        
        # Set random seed for reproducibility
        random_seed = config.get('random_seed', 42)
        np.random.seed(random_seed)
        
        # Feature structure definition
        self.feature_names = self._define_feature_names()
        
        # Richardson number regime thresholds
        self.regime_thresholds = {
            'unstable': (-np.inf, 0.0),      # Ri < 0
            'stable': (0.0, 0.25),            # 0 <= Ri < 0.25
            'supercritical': (0.25, np.inf)   # Ri >= 0.25
        }
        
        logger.info("Enhanced MONC Data Processor initialized")
        logger.info(f"Sampling mode: {self.sampling_mode}")
        logger.info(f"Random seed: {random_seed}")
    
    def _define_feature_names(self) -> List[str]:
        """Define human-readable feature names"""
        names = []
        names.extend(['zu', 'zv', 'zw', 'zth', 'zq_vapour', 'zq_cloud_liquid'])
        
        for var in ['zu', 'zv', 'zw', 'zth', 'zq_vapour', 'zq_cloud_liquid']:
            for direction in ['i_plus', 'i_minus', 'j_plus', 'j_minus', 'k_plus', 'k_minus']:
                names.append(f'{var}_{direction}')
        
        names.extend(['dx', 'dy', 'dz', 'height', 'thref'])
        names.extend(['k_normalized', 'z_from_bottom', 'z_from_top'])
        names.extend(['x_normalized', 'y_normalized', 'x_edge_distance', 'y_edge_distance'])
        
        return names
    
    def detect_time_dimension(self, dataset: xr.Dataset) -> str:
        """Auto-detect time dimension name"""
        time_dims = [dim for dim in dataset.dims if dim.startswith('time_series_')]
        
        if len(time_dims) == 0:
            if 'time' in dataset.dims:
                return 'time'
            raise ValueError(f"No time dimension found: {list(dataset.dims)}")
        
        return time_dims[0]
    
    def classify_regime(self, ri_value: float) -> str:
        """
        Classify atmospheric regime based on Richardson number
        
        Args:
            ri_value: Richardson number
            
        Returns:
            Regime name: 'unstable', 'stable', or 'supercritical'
        """
        for regime_name, (lower, upper) in self.regime_thresholds.items():
            if lower <= ri_value < upper:
                return regime_name
        return 'supercritical'  # Fallback for Ri >= 0.25
    
    def get_variable_safe(self, ds: xr.Dataset, var_name: str, 
                          time_dim: str, t: int, i: int, j: int, 
                          k: int, coord: str = 'zn') -> Optional[float]:
        """Safely extract variable value"""
        if var_name not in ds:
            return None
        
        try:
            var = ds[var_name]
            if time_dim not in var.dims:
                return None
            
            sel_dict = {time_dim: t, 'x': i, 'y': j}
            
            if coord in var.dims:
                sel_dict[coord] = k
            elif coord == 'z' and 'zn' in var.dims:
                sel_dict['zn'] = k
            elif coord == 'zn' and 'z' in var.dims:
                sel_dict['z'] = k
            else:
                return None
            
            value = var.isel(sel_dict).values
            return float(value)
            
        except (KeyError, IndexError) as e:
            return None
    
    def extract_point_features(self, ds: xr.Dataset, time_idx: int, 
                               i: int, j: int, k: int, 
                               time_dim: Optional[str] = None) -> Optional[np.ndarray]:
        """Extract 54 features for a single point"""
        if time_dim is None:
            time_dim = self.detect_time_dimension(ds)
        
        nx = ds.sizes['x']
        ny = ds.sizes['y']
        nz_zn = ds.sizes['zn']
        nz_z = ds.sizes['z']
        
        features = []
        
        # Local state (6)
        local_vars_zn = ['zu', 'zv', 'zth', 'zq_vapour', 'zq_cloud_liquid_mass']
        local_vars_z = ['zw']
        
        for var_name in local_vars_zn:
            value = self.get_variable_safe(ds, var_name, time_dim, time_idx, i, j, k, coord='zn')
            if value is None:
                alt_name = var_name[1:] if var_name.startswith('z') else 'z' + var_name
                value = self.get_variable_safe(ds, alt_name, time_dim, time_idx, i, j, k, coord='zn')
            features.append(value if value is not None else 0.0)
        
        for var_name in local_vars_z:
            value = self.get_variable_safe(ds, var_name, time_dim, time_idx, i, j, k, coord='z')
            features.append(value if value is not None else 0.0)
        
        # Neighbors (36)
        neighbors = {
            'i_plus': ((i + 1) % nx, j, k),
            'i_minus': ((i - 1) % nx, j, k),
            'j_plus': (i, (j + 1) % ny, k),
            'j_minus': (i, (j - 1) % ny, k),
            'k_plus': (i, j, min(k + 1, nz_zn - 1)),
            'k_minus': (i, j, max(k - 1, 0)),
        }
        
        for var_name in local_vars_zn:
            for direction, (ni, nj, nk) in neighbors.items():
                value = self.get_variable_safe(ds, var_name, time_dim, time_idx, ni, nj, nk, coord='zn')
                features.append(value if value is not None else 0.0)
        
        for var_name in local_vars_z:
            for direction, (ni, nj, nk) in neighbors.items():
                nk_z = min(nk, nz_z - 1)
                value = self.get_variable_safe(ds, var_name, time_dim, time_idx, ni, nj, nk_z, coord='z')
                features.append(value if value is not None else 0.0)
        
        # Grid info (5)
        dx = 100.0
        dy = 100.0
        if 'zn' in ds.coords:
            heights_zn = ds['zn'].values
            if k < len(heights_zn) - 1:
                dz = heights_zn[k + 1] - heights_zn[k]
            else:
                dz = heights_zn[k] - heights_zn[k - 1]
            height = heights_zn[k]
        else:
            dz = 50.0
            height = k * 50.0
        
        thref = 300.0
        features.extend([dx, dy, dz, height, thref])
        
        # Vertical position (3)
        k_normalized = k / nz_zn
        z_from_bottom = height
        if 'zn' in ds.coords:
            z_max = float(ds['zn'].values[-1])
            z_from_top = z_max - height
        else:
            z_from_top = (nz_zn - k) * dz
        
        features.extend([k_normalized, z_from_bottom, z_from_top])
        
        # Horizontal position (4)
        x_normalized = i / nx
        y_normalized = j / ny
        x_edge_distance = min(i, nx - 1 - i) / nx
        y_edge_distance = min(j, ny - 1 - j) / ny
        
        features.extend([x_normalized, y_normalized, x_edge_distance, y_edge_distance])
        
        if len(features) != 54:
            return None
        
        return np.array(features, dtype=np.float32)
    
    def extract_targets(self, ds: xr.Dataset, time_idx: int, i: int, j: int, k: int,
                       target_type: str, time_dim: Optional[str] = None) -> Optional[Dict[str, float]]:
        """Extract target variables"""
        if time_dim is None:
            time_dim = self.detect_time_dimension(ds)
        
        targets = {}
        
        if target_type == 'coefficients':
            visc = self.get_variable_safe(ds, 'visc_coeff', time_dim, time_idx, i, j, k, coord='z')
            diff = self.get_variable_safe(ds, 'diff_coeff', time_dim, time_idx, i, j, k, coord='z')
            ri = self.get_variable_safe(ds, 'ri_smag', time_dim, time_idx, i, j, k, coord='zn')
            
            if visc is None or diff is None or ri is None:
                return None
            
            targets['visc_coeff'] = visc
            targets['diff_coeff'] = diff
            targets['richardson'] = ri
            
            # Classify regime (only store numeric ID, not string name)
            regime = self.classify_regime(ri)
            regime_map = {'unstable': 0, 'stable': 1, 'supercritical': 2}
            targets['regime'] = regime_map[regime]
            # Don't store regime_name - it's a string and will cause conversion error
        
        elif target_type == 'momentum_tendencies':
            tend_u = self.get_variable_safe(ds, 'tend_u_viscosity_3d_local', time_dim, time_idx, i, j, k, coord='zn')
            tend_v = self.get_variable_safe(ds, 'tend_v_viscosity_3d_local', time_dim, time_idx, i, j, k, coord='zn')
            tend_w = self.get_variable_safe(ds, 'tend_w_viscosity_3d_local', time_dim, time_idx, i, j, k, coord='z')
            
            if tend_u is None or tend_v is None or tend_w is None:
                return None
            
            targets['tend_u'] = tend_u
            targets['tend_v'] = tend_v
            targets['tend_w'] = tend_w
        
        elif target_type == 'scalar_tendencies':
            tend_th = self.get_variable_safe(ds, 'tend_th_diffusion_3d_local', time_dim, time_idx, i, j, k, coord='zn')
            tend_qv = self.get_variable_safe(ds, 'tend_qv_diffusion_3d_local', time_dim, time_idx, i, j, k, coord='zn')
            tend_ql = self.get_variable_safe(ds, 'tend_ql_diffusion_3d_local', time_dim, time_idx, i, j, k, coord='zn')
            
            if tend_th is None or tend_qv is None or tend_ql is None:
                return None
            
            targets['tend_th'] = tend_th
            targets['tend_qv'] = tend_qv
            targets['tend_ql'] = tend_ql
        
        else:
            raise ValueError(f"Unknown target_type: {target_type}")
        
        return targets
    
    def first_pass_regime_classification(
        self, 
        ds: xr.Dataset, 
        time_dim: str,
        k_levels: Tuple[int, int],
        spatial_subsample: int = 1
    ) -> Dict[str, List[Tuple[int, int, int, int]]]:
        """
        FIRST PASS: Classify all points by regime without extracting features.
        
        This is much faster than extracting features, allowing us to:
        1. Count samples per regime
        2. Determine if we need fallback strategy
        3. Create balanced sampling strategy
        
        Args:
            ds: Dataset
            time_dim: Time dimension name
            k_levels: (k_min, k_max) vertical range
            spatial_subsample: Spatial subsampling for initial scan
            
        Returns:
            Dictionary mapping regime names to list of (t, i, j, k) coordinates
        """
        logger.info("  First pass: Classifying all points by regime...")
        
        n_time = len(ds[time_dim])
        nx = ds.sizes['x']
        ny = ds.sizes['y']
        
        k_min, k_max = k_levels
        
        # Store indices by regime
        regime_indices = {
            'unstable': [],
            'stable': [],
            'supercritical': []
        }
        
        total_points = n_time * (nx // spatial_subsample) * (ny // spatial_subsample) * (k_max - k_min + 1)
        
        with tqdm(total=total_points, desc="  Classifying", unit="pts") as pbar:
            for t in range(n_time):
                for i in range(0, nx, spatial_subsample):
                    for j in range(0, ny, spatial_subsample):
                        for k in range(k_min, k_max + 1):
                            # Only extract Richardson number (fast!)
                            ri = self.get_variable_safe(
                                ds, 'ri_smag', time_dim, t, i, j, k, coord='zn'
                            )
                            
                            if ri is not None:
                                regime = self.classify_regime(ri)
                                regime_indices[regime].append((t, i, j, k))
                            
                            pbar.update(1)
        
        # Log regime distribution
        logger.info("  Regime distribution:")
        for regime, indices in regime_indices.items():
            logger.info(f"    {regime}: {len(indices):,} points")
        
        return regime_indices
    
    def stratified_sampling_strategy(
        self,
        regime_indices: Dict[str, List],
        n_samples_per_regime: int,
        min_samples_threshold: int = 100
    ) -> Tuple[Dict[str, List], Dict[str, str]]:
        """
        Determine stratified sampling strategy with safety fallbacks.
        
        Args:
            regime_indices: Dict mapping regime to list of coordinates
            n_samples_per_regime: Target samples per regime
            min_samples_threshold: Minimum samples needed per regime
            
        Returns:
            (selected_indices, messages): Selected indices and warning messages
        """
        selected_indices = {}
        messages = {}
        
        for regime, indices in regime_indices.items():
            n_available = len(indices)
            
            if n_available == 0:
                # FALLBACK 1: No samples in this regime
                selected_indices[regime] = []
                messages[regime] = f"⚠️  {regime}: NO SAMPLES FOUND - skipping"
                logger.warning(messages[regime])
                
            elif n_available < min_samples_threshold:
                # FALLBACK 2: Very few samples - take all
                selected_indices[regime] = indices
                messages[regime] = (
                    f"⚠️  {regime}: Only {n_available} samples available "
                    f"(< threshold {min_samples_threshold}) - using all"
                )
                logger.warning(messages[regime])
                
            elif n_available < n_samples_per_regime:
                # FALLBACK 3: Fewer samples than requested - take all
                selected_indices[regime] = indices
                messages[regime] = (
                    f"ℹ️  {regime}: {n_available} samples available "
                    f"(requested {n_samples_per_regime}) - using all"
                )
                logger.info(messages[regime])
                
            else:
                # NORMAL CASE: Randomly sample requested amount
                selected = np.random.choice(
                    len(indices), 
                    size=n_samples_per_regime, 
                    replace=False
                )
                selected_indices[regime] = [indices[i] for i in selected]
                messages[regime] = (
                    f"✓ {regime}: {n_samples_per_regime:,} samples "
                    f"(from {n_available:,} available)"
                )
                logger.info(messages[regime])
        
        return selected_indices, messages
    
    def process_single_file_stratified(
        self,
        nc_file: Path,
        k_levels: Tuple[int, int],
        target_type: str,
        n_samples_per_regime: int,
        spatial_subsample: int = 1,
        min_samples_threshold: int = 100
    ) -> Dict:
        """
        Process file with STRATIFIED sampling by Richardson number regime.
        
        Args:
            nc_file: Path to NetCDF file
            k_levels: (k_min, k_max) vertical range
            target_type: Type of targets
            n_samples_per_regime: Target samples per regime
            spatial_subsample: Initial spatial subsampling for classification
            min_samples_threshold: Minimum samples needed per regime
            
        Returns:
            Dictionary with features, targets, and metadata
        """
        logger.info(f"Processing file (STRATIFIED): {nc_file.name}")
        
        try:
            ds = xr.open_dataset(nc_file)
            time_dim = self.detect_time_dimension(ds)
            
            nx = ds.sizes['x']
            ny = ds.sizes['y']
            nz = ds.sizes['zn']
            
            k_min, k_max = k_levels
            k_min = max(k_min, 0)
            k_max = min(k_max, nz - 1)
            
            logger.info(f"  Grid: {len(ds[time_dim])} times × {nx} x × {ny} y × {nz} z")
            logger.info(f"  Vertical range: k=[{k_min}, {k_max}]")
            logger.info(f"  Target per regime: {n_samples_per_regime:,}")
            
            # FIRST PASS: Classify all points by regime
            regime_indices = self.first_pass_regime_classification(
                ds, time_dim, k_levels, spatial_subsample
            )
            
            # Determine sampling strategy with fallbacks
            selected_indices, messages = self.stratified_sampling_strategy(
                regime_indices, 
                n_samples_per_regime,
                min_samples_threshold
            )
            
            # Flatten all selected indices
            all_coords = []
            regime_labels = []
            for regime, coords in selected_indices.items():
                all_coords.extend(coords)
                regime_labels.extend([regime] * len(coords))
            
            if len(all_coords) == 0:
                logger.error("  No samples selected!")
                ds.close()
                return {'features': np.array([]), 'targets': {}, 'n_samples': 0}
            
            logger.info(f"  Total selected: {len(all_coords):,} samples")
            
            # SECOND PASS: Extract features and targets
            logger.info("  Second pass: Extracting features...")
            features_list = []
            targets_dict = defaultdict(list)
            
            n_valid = 0
            n_skipped = 0
            
            with tqdm(all_coords, desc=f"  {nc_file.name}", unit="pts") as pbar:
                for t, i, j, k in pbar:
                    # Extract features
                    features = self.extract_point_features(ds, t, i, j, k, time_dim=time_dim)
                    
                    if features is None:
                        n_skipped += 1
                        continue
                    
                    # Extract targets
                    targets = self.extract_targets(ds, t, i, j, k, target_type, time_dim=time_dim)
                    
                    if targets is None:
                        n_skipped += 1
                        continue
                    
                    # Store
                    features_list.append(features)
                    for key, value in targets.items():
                        targets_dict[key].append(value)
                    
                    n_valid += 1
            
            ds.close()
            
            # Final regime distribution
            if 'regime' in targets_dict:
                regime_counts = Counter(targets_dict['regime'])
                logger.info("  Final regime distribution:")
                regime_names = {0: 'unstable', 1: 'stable', 2: 'supercritical'}
                for regime_id, count in sorted(regime_counts.items()):
                    logger.info(f"    {regime_names[regime_id]}: {count:,} samples")
            
            logger.info(f"  Valid samples: {n_valid:,}")
            if n_skipped > 0:
                logger.warning(f"  Skipped samples: {n_skipped:,}")
            
            # Convert to arrays
            if len(features_list) == 0:
                return {'features': np.array([]), 'targets': {}, 'n_samples': 0}
            
            features_array = np.array(features_list, dtype=np.float32)
            
            # Convert targets to arrays, filtering out non-numeric fields
            targets_arrays = {}
            for key, vals in targets_dict.items():
                # Skip non-numeric fields (shouldn't have any now, but safety check)
                if key == 'regime_name':
                    continue
                try:
                    targets_arrays[key] = np.array(vals, dtype=np.float32)
                except (ValueError, TypeError) as e:
                    logger.warning(f"  Skipping target '{key}' - cannot convert to float: {e}")
            
            return {
                'features': features_array,
                'targets': targets_arrays,
                'n_samples': len(features_array),
                'file': str(nc_file.name),
                'sampling_messages': messages
            }
            
        except Exception as e:
            logger.error(f"Error processing {nc_file.name}: {e}", exc_info=True)
            return {'features': np.array([]), 'targets': {}, 'n_samples': 0}
    
    def process_single_file_random(
        self,
        nc_file: Path,
        k_levels: Tuple[int, int],
        target_type: str,
        n_samples_target: int
    ) -> Dict:
        """
        Process file with RANDOM sampling.
        
        Args:
            nc_file: Path to NetCDF file
            k_levels: (k_min, k_max) vertical range
            target_type: Type of targets
            n_samples_target: Total number of random samples
            
        Returns:
            Dictionary with features, targets, and metadata
        """
        logger.info(f"Processing file (RANDOM): {nc_file.name}")
        
        try:
            ds = xr.open_dataset(nc_file)
            time_dim = self.detect_time_dimension(ds)
            
            n_time = len(ds[time_dim])
            nx = ds.sizes['x']
            ny = ds.sizes['y']
            nz = ds.sizes['zn']
            
            k_min, k_max = k_levels
            k_min = max(k_min, 0)
            k_max = min(k_max, nz - 1)
            
            # Calculate total available points
            total_available = n_time * nx * ny * (k_max - k_min + 1)
            
            logger.info(f"  Grid: {n_time} times × {nx} x × {ny} y × {nz} z")
            logger.info(f"  Total available: {total_available:,}")
            logger.info(f"  Requested samples: {n_samples_target:,}")
            
            # Determine actual samples
            n_samples = min(n_samples_target, total_available)
            
            if n_samples < n_samples_target:
                logger.warning(
                    f"  Only {total_available:,} points available "
                    f"(requested {n_samples_target:,})"
                )
            
            # Generate random indices
            logger.info("  Generating random coordinates...")
            random_indices = np.random.choice(
                total_available, 
                size=n_samples, 
                replace=False
            )
            random_indices.sort()
            
            # Convert to coordinates
            coords = []
            ny_nk = ny * (k_max - k_min + 1)
            nk = k_max - k_min + 1
            
            for flat_idx in random_indices:
                t = flat_idx // (nx * ny_nk)
                remainder = flat_idx % (nx * ny_nk)
                
                i = remainder // ny_nk
                remainder = remainder % ny_nk
                
                j = remainder // nk
                k = k_min + (remainder % nk)
                
                coords.append((t, i, j, k))
            
            # Extract features and targets
            logger.info(f"  Extracting {len(coords):,} samples...")
            features_list = []
            targets_dict = defaultdict(list)
            
            n_valid = 0
            n_skipped = 0
            
            with tqdm(coords, desc=f"  {nc_file.name}", unit="pts") as pbar:
                for t, i, j, k in pbar:
                    features = self.extract_point_features(ds, t, i, j, k, time_dim=time_dim)
                    
                    if features is None:
                        n_skipped += 1
                        continue
                    
                    targets = self.extract_targets(ds, t, i, j, k, target_type, time_dim=time_dim)
                    
                    if targets is None:
                        n_skipped += 1
                        continue
                    
                    features_list.append(features)
                    for key, value in targets.items():
                        targets_dict[key].append(value)
                    
                    n_valid += 1
            
            ds.close()
            
            logger.info(f"  Valid samples: {n_valid:,}")
            if n_skipped > 0:
                logger.warning(f"  Skipped samples: {n_skipped:,}")
            
            if len(features_list) == 0:
                return {'features': np.array([]), 'targets': {}, 'n_samples': 0}
            
            features_array = np.array(features_list, dtype=np.float32)
            
            # Convert targets to arrays, filtering out non-numeric fields
            targets_arrays = {}
            for key, vals in targets_dict.items():
                if key == 'regime_name':
                    continue
                try:
                    targets_arrays[key] = np.array(vals, dtype=np.float32)
                except (ValueError, TypeError) as e:
                    logger.warning(f"  Skipping target '{key}' - cannot convert to float: {e}")
            
            return {
                'features': features_array,
                'targets': targets_arrays,
                'n_samples': len(features_array),
                'file': str(nc_file.name)
            }
            
        except Exception as e:
            logger.error(f"Error processing {nc_file.name}: {e}", exc_info=True)
            return {'features': np.array([]), 'targets': {}, 'n_samples': 0}
    
    # NOTE: Add the regular process_single_file method here from previous version
    # (The one with regular grid subsampling - omitted for brevity)


def main():
    """Enhanced main with multiple sampling modes"""
    import argparse
    import time
    
    parser = argparse.ArgumentParser(
        description='Process MONC NetCDF files with multiple sampling strategies'
    )
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--output-dir', type=str, default='./processed_data')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--k-min', type=int, default=2)
    parser.add_argument('--k-max', type=int, default=30)
    
    # Sampling mode selection
    sampling_group = parser.add_mutually_exclusive_group()
    sampling_group.add_argument('--subsample', type=int, default=None,
                               help='Regular grid subsampling (1=all points, 2=every 2nd, etc.)')
    sampling_group.add_argument('--n-samples', type=int, default=None,
                               help='Random sampling: total number of samples')
    sampling_group.add_argument('--n-samples-per-regime', type=int, default=None,
                               help='Stratified sampling: samples per regime')
    
    parser.add_argument('--min-samples-threshold', type=int, default=100,
                       help='Minimum samples per regime for stratified sampling')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--target-type', type=str, default='coefficients',
                       choices=['coefficients', 'momentum_tendencies', 'scalar_tendencies'])
    parser.add_argument('--test', action='store_true',
                       help='Quick test mode')
    
    args = parser.parse_args()
    
    # Determine sampling mode and parameters
    if args.n_samples_per_regime is not None:
        sampling_mode = 'stratified'
        sampling_params = {
            'n_samples_per_regime': args.n_samples_per_regime,
            'min_samples_threshold': args.min_samples_threshold
        }
    elif args.n_samples is not None:
        sampling_mode = 'random'
        sampling_params = {'n_samples_target': args.n_samples}
    else:
        sampling_mode = 'regular'
        subsample = args.subsample if args.subsample is not None else 1
        sampling_params = {'spatial_subsample': subsample}
    
    # Test mode overrides
    if args.test:
        logger.info("🧪 TEST MODE ENABLED")
        args.k_max = min(args.k_max, 10)
        if sampling_mode == 'regular':
            sampling_params['spatial_subsample'] = 4
        elif sampling_mode == 'random':
            sampling_params['n_samples_target'] = 5000
        elif sampling_mode == 'stratified':
            sampling_params['n_samples_per_regime'] = 500
    
    # Initialize processor
    config = {
        'halo_size': 2,
        'chunk_size': 100000,
        'use_robust_scaling': False,
        'sampling_mode': sampling_mode,
        'random_seed': args.random_seed
    }
    
    processor = EnhancedMONCDataProcessor(config)
    
    # Find NetCDF files
    data_dir = Path(args.data_dir)
    nc_files = sorted(data_dir.glob('*.nc'))
    
    if len(nc_files) == 0:
        logger.error(f"No NetCDF files found in {data_dir}")
        return
    
    logger.info(f"Found {len(nc_files)} NetCDF files")
    logger.info(f"Sampling mode: {sampling_mode}")
    logger.info(f"Sampling params: {sampling_params}")
    
    # Process files
    output_dir = Path(args.output_dir) / args.target_type / sampling_mode
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Processing: {args.target_type}")
    logger.info(f"Sampling: {sampling_mode}")
    logger.info(f"{'='*70}\n")
    
    all_features = []
    all_targets = defaultdict(list)
    all_messages = []
    
    start_time = time.time()
    
    for nc_file in nc_files:
        # Select processing method based on mode
        if sampling_mode == 'stratified':
            result = processor.process_single_file_stratified(
                nc_file=nc_file,
                k_levels=(args.k_min, args.k_max),
                target_type=args.target_type,
                n_samples_per_regime=sampling_params['n_samples_per_regime'],
                spatial_subsample=1,  # Can adjust for faster classification
                min_samples_threshold=sampling_params['min_samples_threshold']
            )
            if 'sampling_messages' in result:
                all_messages.append({
                    'file': nc_file.name,
                    'messages': result['sampling_messages']
                })
        
        elif sampling_mode == 'random':
            result = processor.process_single_file_random(
                nc_file=nc_file,
                k_levels=(args.k_min, args.k_max),
                target_type=args.target_type,
                n_samples_target=sampling_params['n_samples_target']
            )
        
        else:  # regular
            # Note: You would call the regular processing method here
            # processor.process_single_file_regular(...)
            logger.warning("Regular mode not implemented in this snippet - use original processor")
            continue
        
        if result['n_samples'] > 0:
            all_features.append(result['features'])
            for key, values in result['targets'].items():
                all_targets[key].append(values)
    
    # Combine results
    if len(all_features) == 0:
        logger.error("No valid samples extracted!")
        return
    
    combined_features = np.vstack(all_features)
    combined_targets = {key: np.concatenate(all_targets[key]) 
                       for key in all_targets}
    
    # Save processed data
    np.save(output_dir / 'features.npy', combined_features)
    for key, values in combined_targets.items():
        np.save(output_dir / f'{key}.npy', values)
    
    # Save configuration and messages
    config_dict = {
        'sampling_mode': sampling_mode,
        'sampling_params': sampling_params,
        'k_levels': (args.k_min, args.k_max),
        'target_type': args.target_type,
        'n_files': len(nc_files),
        'n_samples': len(combined_features),
        'random_seed': args.random_seed,
        'files': [str(f.name) for f in nc_files]
    }
    
    if all_messages:
        config_dict['sampling_messages'] = all_messages
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    elapsed = time.time() - start_time
    
    # Final summary
    logger.info(f"\n{'='*70}")
    logger.info("✅ PROCESSING COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"Sampling mode: {sampling_mode}")
    logger.info(f"Total samples: {len(combined_features):,}")
    logger.info(f"Features shape: {combined_features.shape}")
    logger.info(f"Targets: {list(combined_targets.keys())}")
    
    # Show final regime distribution for stratified sampling
    if sampling_mode == 'stratified' and 'regime' in combined_targets:
        regime_counts = Counter(combined_targets['regime'])
        regime_names = {0: 'unstable', 1: 'stable', 2: 'supercritical'}
        logger.info("\nFinal regime distribution across all files:")
        for regime_id, count in sorted(regime_counts.items()):
            pct = 100 * count / len(combined_features)
            logger.info(f"  {regime_names[regime_id]}: {count:,} ({pct:.1f}%)")
    
    logger.info(f"\nOutput: {output_dir}")
    logger.info(f"⏱️  Processing time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    logger.info(f"{'='*70}")


if __name__ == '__main__':
    main()
