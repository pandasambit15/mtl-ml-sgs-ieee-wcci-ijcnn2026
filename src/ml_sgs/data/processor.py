#!/usr/bin/env python3
"""
Fast, Unified, and Enhanced MONC Data Processor (Final Version)
================================================================

This single script combines the best features of all previous processors:

- ✅ FAST: Uses multiprocessing to process multiple NetCDF files in parallel.
- ✅ UNIFIED & CORRECT: Robustly handles different data formats by auto-detecting
     time dimensions and dynamically extracting dx, dy, and thref.
- ✅ ADAPTIVE: Automatically detects simulation type (ARM vs. RCE) and adjusts
     the vertical processing range, with a flag to use the full vertical domain.
- ✅ ENHANCED: Includes 'regular', 'random', and 'stratified' sampling to create
     smaller, more balanced datasets for efficient model training.
- ✅ SELF-CONTAINED: Creates and saves the necessary scaler .pkl files.
"""

import numpy as np
import xarray as xr
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter
from tqdm import tqdm
import json
import logging
import gc
import joblib
import argparse
import time
from multiprocessing import Pool, cpu_count
from sklearn.preprocessing import StandardScaler, RobustScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FastDataLoader:
    """Fast data loader with pre-loaded arrays and consistent parameter extraction."""
    
    def __init__(self, nc_file: Path, time_idx: int):
        logger.info(f"  Pre-loading {nc_file.name} into memory...")
        start_time = time.time()
        
        # Open and load the entire dataset into RAM
        ds = xr.open_dataset(nc_file).load()
        
        # Detect time dimension
        time_dims = [dim for dim in ds.dims if dim.startswith('time_series_')]
        self.time_dim = time_dims[0] if time_dims else 'time'
        
        self.nx, self.ny, self.nz_zn, self.nz_z = ds.sizes['x'], ds.sizes['y'], ds.sizes['zn'], ds.sizes['z']
        
        # Load state variables
        self.zu = self._load_var(ds, 'zu', time_idx, 'zn')
        self.zv = self._load_var(ds, 'zv', time_idx, 'zn')
        self.zth = self._load_var(ds, 'zth', time_idx, 'zn')
        self.zq_vapour = self._load_var(ds, 'zq_vapour', time_idx, 'zn')
        self.zq_cloud = self._load_var(ds, 'zq_cloud_liquid_mass', time_idx, 'zn')
        self.zw = self._load_var(ds, 'zw', time_idx, 'z')
        
        # Load targets
        self.visc_coeff = self._load_var(ds, 'visc_coeff', time_idx, 'z')
        self.diff_coeff = self._load_var(ds, 'diff_coeff', time_idx, 'z')
        self.ri_smag = self._load_var(ds, 'ri_smag', time_idx, 'zn')
        
        self.heights = ds['zn'].values if 'zn' in ds.coords else np.arange(self.nz_zn) * 50.0

        # ========== CORRECTED & CONSISTENT PARAMETER EXTRACTION ==========
        self.dx = float(ds['x_resolution'].isel({self.time_dim: time_idx}).values) if 'x_resolution' in ds else 100.0
        self.dy = float(ds['y_resolution'].isel({self.time_dim: time_idx}).values) if 'y_resolution' in ds else 100.0
        self.thref = float(ds.attrs['thref']) if 'thref' in ds.attrs else 300.0
        # =================================================================
        
        ds.close()
        logger.info(f"    Loaded in {time.time() - start_time:.1f}s")

    def _load_var(self, ds, var_name, time_idx, coord):
        """Safely load a variable into a NumPy array."""
        # Check for primary and alternative variable names
        for name in [var_name, var_name[1:] if var_name.startswith('z') else 'z' + var_name]:
            if name in ds:
                var = ds[name]
                if self.time_dim in var.dims and coord in var.dims:
                    return var.isel({self.time_dim: time_idx}).values
        # Return zeros if variable is not found
        shape = (self.nx, self.ny, self.nz_zn if coord == 'zn' else self.nz_z)
        logger.warning(f"Variable '{var_name}' not found. Using array of zeros.")
        return np.zeros(shape, dtype=np.float32)


class FastFeatureExtractor:
    """Extracts features and targets from pre-loaded data."""
    def __init__(self, data_loader: FastDataLoader):
        self.data = data_loader
        self.regime_thresholds = {'unstable': (-np.inf, 0.0), 'stable': (0.0, 0.25), 'supercritical': (0.25, np.inf)}

    def classify_regime(self, ri_value: float) -> str:
        """Classifies the atmospheric regime based on the Richardson number."""
        if np.isnan(ri_value): return 'stable' # Default for NaN
        for regime, (lower, upper) in self.regime_thresholds.items():
            if lower <= ri_value < upper: return regime
        return 'supercritical'
    
    def extract_point_features(self, i: int, j: int, k: int) -> Optional[np.ndarray]:
        """Extracts the 54-feature vector for a single grid point."""
        features = []
        d = self.data
        nx, ny, nz_zn, nz_z = d.nx, d.ny, d.nz_zn, d.nz_z

        # Local state (6 features)
        features.extend([d.zu[i, j, k], d.zv[i, j, k], d.zw[i, j, min(k, nz_z - 1)], d.zth[i, j, k], d.zq_vapour[i, j, k], d.zq_cloud[i, j, k]])
        
        # Neighbors (36 features)
        i_p, i_m = (i + 1) % nx, (i - 1) % nx
        j_p, j_m = (j + 1) % ny, (j - 1) % ny
        k_p, k_m = min(k + 1, nz_zn - 1), max(k - 1, 0)
        
        for var in [d.zu, d.zv, d.zth, d.zq_vapour, d.zq_cloud]:
            features.extend([var[i_p, j, k], var[i_m, j, k], var[i, j_p, k], var[i, j_m, k], var[i, j, k_p], var[i, j, k_m]])
        
        k_z, k_pz, k_mz = min(k, nz_z - 1), min(k_p, nz_z - 1), min(k_m, nz_z - 1)
        features.extend([d.zw[i_p, j, k_z], d.zw[i_m, j, k_z], d.zw[i, j_p, k_z], d.zw[i, j_m, k_z], d.zw[i, j, k_pz], d.zw[i, j, k_mz]])
        
        # Grid info (5 features)
        height = d.heights[k]
        dz = d.heights[k_p] - height if k < nz_zn - 1 else height - d.heights[k_m]
        features.extend([d.dx, d.dy, dz, height, d.thref])
        
        # Positional info (7 features)
        z_max = d.heights[-1]
        features.extend([k / nz_zn, height, z_max - height]) # Vertical
        features.extend([i / nx, j / ny, min(i, nx - 1 - i) / nx, min(j, ny - 1 - j) / ny]) # Horizontal
        
        return np.array(features, dtype=np.float32) if len(features) == 54 else None

    def extract_targets(self, i: int, j: int, k: int, target_type: str) -> Optional[Dict[str, float]]:
        """Extracts the ground truth target values for a single grid point."""
        if target_type == 'coefficients':
            k_z = min(k, self.data.nz_z - 1)
            visc, diff, ri = self.data.visc_coeff[i, j, k_z], self.data.diff_coeff[i, j, k_z], self.data.ri_smag[i, j, k]
            if any(np.isnan(v) for v in [visc, diff, ri]): return None
            return {
                'visc_coeff': visc, 'diff_coeff': diff, 'richardson': ri,
                'regime': {'unstable': 0, 'stable': 1, 'supercritical': 2}[self.classify_regime(ri)]
            }
        return None
    
    def extract_chunk(self, coords_list: List[Tuple], target_type: str) -> Tuple[List, List]:
        """Processes a list (chunk) of coordinates."""
        features_list, targets_list = [], []
        for i, j, k in coords_list:
            features = self.extract_point_features(i, j, k)
            targets = self.extract_targets(i, j, k, target_type)
            if features is not None and targets is not None:
                features_list.append(features)
                targets_list.append(targets)
        return features_list, targets_list


def extract_chunk_worker(args):
    """Wrapper function for each parallel worker process."""
    data_loader, coords_list, target_type = args
    extractor = FastFeatureExtractor(data_loader)
    return extractor.extract_chunk(coords_list, target_type)


class FastMONCDataProcessor:
    """Main class to orchestrate the parallel data processing."""
    def __init__(self, config: Dict):
        self.config = config
        self.sampling_mode = config.get('sampling_mode', 'regular')
        self.auto_detect_k_levels = config.get('auto_detect_k_levels', True)
        self.use_full_vertical_range = config.get('use_full_vertical_range', False)
        self.n_workers = config.get('n_workers', max(1, cpu_count() - 2))
        np.random.seed(config.get('random_seed', 42))

    def detect_simulation_type(self, nc_file: Path) -> Dict:
        """Detects simulation type and sets k_levels based on vertical dimension size."""
        with xr.open_dataset(nc_file) as ds:
            nz = ds.sizes.get('zn', ds.sizes.get('z', -1))
        
        if nz >= 200: sim_type, default_k_max = 'ARM', 50
        elif 80 <= nz < 200: sim_type, default_k_max = 'RCE', 30
        else: sim_type, default_k_max = 'OTHER', 30
        
        if self.use_full_vertical_range:
            k_min, k_max = 0, nz - 1
            logger.info(f"  ✓ Detected: {sim_type} (nz={nz}), using FULL RANGE k={k_min}:{k_max}")
        else:
            k_min, k_max = 2, min(default_k_max, nz - 1)
            logger.info(f"  ✓ Detected: {sim_type} (nz={nz}), using k={k_min}:{k_max} (boundary layer focus)")
            
        return {'sim_type': sim_type, 'nz': nz, 'k_min': k_min, 'k_max': k_max}

    def process_directory(self, data_dir: Path, output_dir: Path, target_type: str, sampling_params: Dict, k_levels_fixed: Optional[Tuple] = None):
        """Processes all files in a directory, aggregates, and saves the final dataset."""
        nc_files = sorted(data_dir.glob('*.nc'))
        if not nc_files: return logger.error(f"No NetCDF files found in {data_dir}")

        all_features, all_targets_dict, file_info_list = [], defaultdict(list), []
        total_time = 0

        for nc_file in nc_files:
            if self.auto_detect_k_levels and k_levels_fixed is None:
                file_info = self.detect_simulation_type(nc_file)
                k_levels = (file_info['k_min'], file_info['k_max'])
            else:
                k_levels = k_levels_fixed
            
            result = self._process_file_parallel(nc_file, 0, k_levels, target_type, sampling_params)
            
            if result['n_samples'] > 0:
                all_features.append(result['features'])
                for key, values in result['targets'].items(): all_targets_dict[key].append(values)
                total_time += result['processing_time']
                file_info_list.append({'file': nc_file.name, 'k_levels': k_levels, 'samples': result['n_samples']})

        if not all_features: return logger.error("No valid samples extracted!")
        
        logger.info("\nCombining results and saving...")
        combined_features = np.vstack(all_features)
        combined_targets = {key: np.concatenate(vals) for key, vals in all_targets_dict.items()}
        
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / 'features.npy', combined_features)
        for key, values in combined_targets.items(): np.save(output_dir / f'{key}.npy', values)
        
        # Create and save scalers
        logger.info("Creating and saving scalers using RobustScaler...")
        joblib.dump(RobustScaler().fit(combined_features), output_dir / 'feature_scaler.pkl')
        for key in ['visc_coeff', 'diff_coeff', 'richardson']:
            if key in combined_targets:
                scaler_name = f'{key.split("_")[0]}_scaler.pkl'
                joblib.dump(StandardScaler().fit(combined_targets[key].reshape(-1, 1)), output_dir / scaler_name)
        
        # Save metadata
        config_dict = self.config.copy()
        config_dict.update({
            'sampling_params': sampling_params,
            'n_files': len(file_info_list), 'n_samples': len(combined_features),
            'total_processing_time_s': total_time, 'file_info': file_info_list
        })
        with open(output_dir / 'config.json', 'w') as f: json.dump(config_dict, f, indent=2)
        logger.info(f"✅ Processing complete. Data saved to {output_dir}")

    def _process_file_parallel(self, nc_file: Path, time_idx: int, k_levels: Tuple, target_type: str, sampling_params: Dict) -> Dict:
        """Orchestrates the parallel processing of a single file."""
        start_time = time.time()
        data_loader = FastDataLoader(nc_file, time_idx)
        k_min, k_max = max(k_levels[0], 0), min(k_levels[1], data_loader.nz_zn - 1)
        
        # Generate coordinates to process based on sampling mode
        coords = self._generate_coordinates(data_loader, k_min, k_max, sampling_params)
        logger.info(f"  Total points to process from {nc_file.name}: {len(coords):,}")

        # Split coordinates into chunks for workers
        chunk_size = max(100, len(coords) // (self.n_workers * 4))
        chunks = [coords[i:i + chunk_size] for i in range(0, len(coords), chunk_size)]
        tasks = [(data_loader, chunk, target_type) for chunk in chunks]
        
        all_features, all_targets = [], defaultdict(list)
        with Pool(self.n_workers) as pool:
            results = list(tqdm(pool.imap(extract_chunk_worker, tasks), total=len(tasks), desc=f"  Extracting from {nc_file.name}"))

        for features_list, targets_list in results:
            all_features.extend(features_list)
            for targets in targets_list:
                for key, value in targets.items(): all_targets[key].append(value)
        
        features_array = np.array(all_features, dtype=np.float32) if all_features else np.array([])
        targets_arrays = {key: np.array(vals, dtype=np.float32) for key, vals in all_targets.items()}
        
        return {'features': features_array, 'targets': targets_arrays, 'n_samples': len(features_array), 'processing_time': time.time() - start_time}

    def _generate_coordinates(self, data_loader: FastDataLoader, k_min: int, k_max: int, sampling_params: Dict) -> List[Tuple]:
        """Generates a list of (i, j, k) coordinates based on the sampling strategy."""
        nx, ny = data_loader.nx, data_loader.ny
        
        if self.sampling_mode == 'stratified':
            regime_indices = defaultdict(list)
            extractor = FastFeatureExtractor(data_loader)
            logger.info("  Stratified mode: performing first pass classification...")
            for i in range(nx):
                for j in range(ny):
                    for k in range(k_min, k_max + 1):
                        regime = extractor.classify_regime(data_loader.ri_smag[i, j, k])
                        regime_indices[regime].append((i, j, k))
            
            selected_coords = []
            n_per_regime = sampling_params.get('n_samples_per_regime', 10000)
            for regime, indices in regime_indices.items():
                n_available = len(indices)
                n_to_sample = min(n_per_regime, n_available)
                if n_to_sample > 0:
                    selected_idx = np.random.choice(n_available, size=n_to_sample, replace=False)
                    selected_coords.extend([indices[idx] for idx in selected_idx])
            return selected_coords
        
        elif self.sampling_mode == 'random':
            total_points = nx * ny * (k_max - k_min + 1)
            n_samples = min(sampling_params.get('n_samples_target', 10000), total_points)
            flat_indices = np.random.choice(total_points, size=n_samples, replace=False)
            
            nk = k_max - k_min + 1
            k_coords = k_min + (flat_indices % nk)
            j_coords = (flat_indices // nk) % ny
            i_coords = (flat_indices // (nk * ny))
            return list(zip(i_coords, j_coords, k_coords))
            
        else: # Regular
            subsample = sampling_params.get('spatial_subsample', 1)
            return [(i, j, k) for i in range(0, nx, subsample) for j in range(0, ny, subsample) for k in range(k_min, k_max + 1)]

def main():
    """Main function to parse arguments and run the processor."""
    parser = argparse.ArgumentParser(
        description='Fast, Unified, and Enhanced MONC Data Processor',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  # Stratified sampling with auto-detected boundary-layer focus (default)
  python fast_unified_processor.py --data-dir ./mixed_data --output-dir ./proc --n-samples-per-regime 10000

  # Stratified sampling using the FULL vertical range of each file
  python fast_unified_processor.py --data-dir ./mixed_data --output-dir ./proc_full --use-full-vertical-range --n-samples-per-regime 10000

  # Random sampling with a fixed vertical range for all files
  python fast_unified_processor.py --data-dir ./data --output-dir ./proc_fixed --no-auto-detect --k-min 2 --k-max 40 --n-samples 50000
        """
    )
    
    parser.add_argument('--data-dir', type=str, required=True, help='Directory containing NetCDF files')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for processed data')
    
    # Vertical range options
    parser.add_argument('--k-min', type=int, default=2, help='Minimum vertical level (only used if --no-auto-detect is set)')
    parser.add_argument('--k-max', type=int, default=30, help='Maximum vertical level (only used if --no-auto-detect is set)')
    parser.add_argument('--no-auto-detect', action='store_true', help='Disable auto-detection of k-levels and use the fixed k-min/k-max for all files.')
    parser.add_argument('--use-full-vertical-range', action='store_true', help='If auto-detecting, use all vertical levels (k=0 to nz-1) for each file. The default is a boundary layer focus.')
    
    # Sampling mode selection (mutually exclusive)
    sampling_group = parser.add_mutually_exclusive_group(required=True)
    sampling_group.add_argument('--subsample', type=int, help='Regular grid subsampling (e.g., 2 for every 2nd point).')
    sampling_group.add_argument('--n-samples', type=int, help='Random sampling: total number of samples per file.')
    sampling_group.add_argument('--n-samples-per-regime', type=int, help='Stratified sampling: samples per regime per file (recommended).')
    
    parser.add_argument('--target-type', type=str, default='coefficients', choices=['coefficients'])
    parser.add_argument('--random-seed', type=int, default=42)
    parser.add_argument('--n-workers', type=int, default=None, help='Number of CPU workers (default: auto-detect).')
    
    args = parser.parse_args()
    
    # Determine sampling mode and parameters from arguments
    if args.n_samples_per_regime is not None:
        sampling_mode = 'stratified'
        sampling_params = {'n_samples_per_regime': args.n_samples_per_regime}
    elif args.n_samples is not None:
        sampling_mode = 'random'
        sampling_params = {'n_samples_target': args.n_samples}
    else:
        sampling_mode = 'regular'
        sampling_params = {'spatial_subsample': args.subsample}
    
    # Configure the processor
    n_workers = args.n_workers if args.n_workers is not None else max(1, cpu_count() - 2)
    config = {
        'sampling_mode': sampling_mode,
        'random_seed': args.random_seed,
        'n_workers': n_workers,
        'auto_detect_k_levels': not args.no_auto_detect,
        'use_full_vertical_range': args.use_full_vertical_range
    }
    
    processor = FastMONCDataProcessor(config)
    
    # Set output path and fixed k-levels if specified
    output_dir = Path(args.output_dir) / args.target_type / sampling_mode
    k_levels_fixed = (args.k_min, args.k_max) if args.no_auto_detect else None
    
    # Run the processing
    processor.process_directory(
        data_dir=Path(args.data_dir),
        output_dir=output_dir,
        target_type=args.target_type,
        sampling_params=sampling_params,
        k_levels_fixed=k_levels_fixed
    )

if __name__ == '__main__':
    main()
