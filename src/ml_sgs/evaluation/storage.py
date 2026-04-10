#!/usr/bin/env python3
"""
Prediction Storage and Management
==================================

Save and load model predictions efficiently for future analysis.
Supports HDF5 format with compression and metadata.
"""

import h5py
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class PredictionStorage:
    """
    Manage storage and retrieval of model predictions.
    
    Features:
    - HDF5 format with compression
    - Metadata tracking (model info, dataset info, timestamps)
    - Efficient batch operations
    - Incremental updates
    """
    
    def __init__(self, storage_dir: Path):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata_file = self.storage_dir / 'metadata.json'
        self.predictions_file = self.storage_dir / 'predictions.h5'
        
        # Load or initialize metadata
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                'created': datetime.now().isoformat(),
                'models': {},
                'datasets': {},
                'files_processed': []
            }
    
    def save_predictions(self, 
                        model_name: str,
                        nc_file: Path,
                        predictions: Dict[str, np.ndarray],
                        model_info: Optional[Dict] = None,
                        overwrite: bool = False):
        """
        Save predictions for a single model and file.
        
        Parameters
        ----------
        model_name : str
            Model identifier
        nc_file : Path
            NetCDF file that was processed
        predictions : Dict[str, np.ndarray]
            Prediction arrays {'visc_coeff': array, 'diff_coeff': array, ...}
        model_info : Optional[Dict]
            Model metadata (configuration, architecture, etc.)
        overwrite : bool
            If True, overwrite existing predictions
        """
        nc_name = nc_file.name
        
        # Create dataset key
        dataset_key = f"{model_name}/{nc_name}"
        
        with h5py.File(self.predictions_file, 'a') as f:
            # Check if exists
            if dataset_key in f and not overwrite:
                logger.warning(f"Predictions already exist for {dataset_key}. Use overwrite=True.")
                return
            
            # Create group if needed
            if model_name not in f:
                f.create_group(model_name)
            
            # Save each variable
            for var_name, array in predictions.items():
                full_key = f"{dataset_key}/{var_name}"
                
                if full_key in f:
                    del f[full_key]
                
                # Save with compression
                f.create_dataset(
                    full_key,
                    data=array,
                    compression='gzip',
                    compression_opts=4,
                    chunks=True
                )
        
        # Update metadata
        if model_name not in self.metadata['models']:
            self.metadata['models'][model_name] = {
                'files_processed': [],
                'info': model_info or {}
            }
        
        if nc_name not in self.metadata['models'][model_name]['files_processed']:
            self.metadata['models'][model_name]['files_processed'].append(nc_name)
        
        if nc_name not in self.metadata['files_processed']:
            self.metadata['files_processed'].append(nc_name)
        
        self.metadata['datasets'][nc_name] = {
            'processed': datetime.now().isoformat(),
            'path': str(nc_file),
            'variables': list(predictions.keys()),
            'shapes': {k: list(v.shape) for k, v in predictions.items()}
        }
        
        self._save_metadata()
        
        logger.info(f"✓ Saved predictions: {dataset_key}")
    
    def load_predictions(self, 
                        model_name: str,
                        nc_file: Optional[Path] = None) -> Dict[str, np.ndarray]:
        """
        Load predictions for a model.
        
        Parameters
        ----------
        model_name : str
            Model identifier
        nc_file : Optional[Path]
            Specific file (None = load all files for model)
        
        Returns
        -------
        predictions : Dict
            If nc_file specified: {'visc_coeff': array, ...}
            If nc_file is None: {'file1.nc': {'visc_coeff': array, ...}, ...}
        """
        if not self.predictions_file.exists():
            raise FileNotFoundError(f"No predictions file: {self.predictions_file}")
        
        with h5py.File(self.predictions_file, 'r') as f:
            if model_name not in f:
                raise KeyError(f"Model not found: {model_name}")
            
            if nc_file is not None:
                # Load specific file
                nc_name = nc_file.name if isinstance(nc_file, Path) else nc_file
                dataset_key = f"{model_name}/{nc_name}"
                
                if dataset_key not in f:
                    raise KeyError(f"File not found: {dataset_key}")
                
                predictions = {}
                for var_name in f[dataset_key].keys():
                    predictions[var_name] = f[dataset_key][var_name][:]
                
                return predictions
            
            else:
                # Load all files for model
                all_predictions = {}
                for nc_name in f[model_name].keys():
                    predictions = {}
                    for var_name in f[f"{model_name}/{nc_name}"].keys():
                        predictions[var_name] = f[f"{model_name}/{nc_name}/{var_name}"][:]
                    all_predictions[nc_name] = predictions
                
                return all_predictions
    
    def aggregate_predictions(self, 
                             model_names: Optional[List[str]] = None,
                             nc_files: Optional[List[Path]] = None) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Load and aggregate predictions across multiple files.
        
        Returns
        -------
        aggregated : Dict[str, Dict[str, np.ndarray]]
            {'model1': {'visc_coeff': concatenated_array, ...}, ...}
        """
        if model_names is None:
            model_names = list(self.metadata['models'].keys())
        
        aggregated = {}
        
        with h5py.File(self.predictions_file, 'r') as f:
            for model_name in model_names:
                if model_name not in f:
                    logger.warning(f"Model not found: {model_name}")
                    continue
                
                # Get file list
                if nc_files is not None:
                    nc_names = [nf.name if isinstance(nf, Path) else nf for nf in nc_files]
                else:
                    nc_names = list(f[model_name].keys())
                
                # Aggregate arrays
                var_arrays = {}
                
                for nc_name in nc_names:
                    dataset_key = f"{model_name}/{nc_name}"
                    
                    if dataset_key not in f:
                        continue
                    
                    for var_name in f[dataset_key].keys():
                        array = f[dataset_key][var_name][:]
                        
                        if var_name not in var_arrays:
                            var_arrays[var_name] = []
                        
                        var_arrays[var_name].append(array.flatten())
                
                # Concatenate
                aggregated[model_name] = {
                    var: np.concatenate(arrays) 
                    for var, arrays in var_arrays.items()
                }
                
                logger.info(f"✓ Aggregated {model_name}: {len(nc_names)} files")
        
        return aggregated
    
    def get_model_list(self) -> List[str]:
        """Get list of all models with saved predictions."""
        return list(self.metadata['models'].keys())
    
    def get_file_list(self, model_name: Optional[str] = None) -> List[str]:
        """
        Get list of processed files.
        
        Parameters
        ----------
        model_name : Optional[str]
            If specified, return files for this model only
        """
        if model_name is not None:
            if model_name in self.metadata['models']:
                return self.metadata['models'][model_name]['files_processed']
            else:
                return []
        else:
            return self.metadata['files_processed']
    
    def get_storage_info(self) -> Dict:
        """Get storage statistics."""
        info = {
            'storage_dir': str(self.storage_dir),
            'n_models': len(self.metadata['models']),
            'n_files': len(self.metadata['files_processed']),
            'created': self.metadata['created']
        }
        
        if self.predictions_file.exists():
            size_mb = self.predictions_file.stat().st_size / (1024**2)
            info['storage_size_mb'] = f"{size_mb:.2f}"
        
        return info
    
    def delete_predictions(self, model_name: str, nc_file: Optional[Path] = None):
        """
        Delete predictions.
        
        Parameters
        ----------
        model_name : str
            Model identifier
        nc_file : Optional[Path]
            If specified, delete only this file. Otherwise delete all files for model.
        """
        with h5py.File(self.predictions_file, 'a') as f:
            if nc_file is not None:
                nc_name = nc_file.name if isinstance(nc_file, Path) else nc_file
                dataset_key = f"{model_name}/{nc_name}"
                
                if dataset_key in f:
                    del f[dataset_key]
                    logger.info(f"✓ Deleted: {dataset_key}")
                
                # Update metadata
                if model_name in self.metadata['models']:
                    files = self.metadata['models'][model_name]['files_processed']
                    if nc_name in files:
                        files.remove(nc_name)
            
            else:
                # Delete entire model
                if model_name in f:
                    del f[model_name]
                    logger.info(f"✓ Deleted model: {model_name}")
                
                # Update metadata
                if model_name in self.metadata['models']:
                    del self.metadata['models'][model_name]
        
        self._save_metadata()
    
    def _save_metadata(self):
        """Save metadata to JSON."""
        self.metadata['last_updated'] = datetime.now().isoformat()
        
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def summary(self):
        """Print storage summary."""
        print("\n" + "="*70)
        print("PREDICTION STORAGE SUMMARY")
        print("="*70)
        
        info = self.get_storage_info()
        
        print(f"\nStorage Directory: {info['storage_dir']}")
        print(f"Created: {info['created']}")
        
        if 'storage_size_mb' in info:
            print(f"Storage Size: {info['storage_size_mb']} MB")
        
        print(f"\nModels: {info['n_models']}")
        for model_name in sorted(self.metadata['models'].keys()):
            n_files = len(self.metadata['models'][model_name]['files_processed'])
            print(f"  - {model_name}: {n_files} files")
        
        print(f"\nTotal Files Processed: {info['n_files']}")
        
        print("="*70 + "\n")


# ==================== UTILITY FUNCTIONS ====================

def export_to_netcdf(storage: PredictionStorage,
                     model_name: str,
                     nc_file: Path,
                     output_file: Path):
    """
    Export predictions to NetCDF format for visualization.
    
    Parameters
    ----------
    storage : PredictionStorage
        Storage instance
    model_name : str
        Model identifier
    nc_file : Path
        Original NetCDF file (for dimensions)
    output_file : Path
        Output NetCDF file
    """
    import xarray as xr
    
    # Load predictions
    predictions = storage.load_predictions(model_name, nc_file)
    
    # Open original file for dimensions
    with xr.open_dataset(nc_file) as ds:
        # Create dataset
        data_vars = {}
        
        for var_name, array in predictions.items():
            data_vars[f"{model_name}_{var_name}"] = (
                ['x', 'y', 'zn'],
                array
            )
        
        # Create output dataset
        ds_out = xr.Dataset(
            data_vars=data_vars,
            coords={
                'x': ds.coords['x'],
                'y': ds.coords['y'],
                'zn': ds.coords['zn']
            },
            attrs={
                'model': model_name,
                'source_file': str(nc_file),
                'created': datetime.now().isoformat()
            }
        )
        
        # Save
        ds_out.to_netcdf(output_file)
        logger.info(f"✓ Exported to NetCDF: {output_file}")


# ==================== TESTING ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Prediction Storage')
    parser.add_argument('--storage-dir', type=Path, default='prediction_storage')
    parser.add_argument('--action', choices=['summary', 'list', 'info'], default='summary')
    
    args = parser.parse_args()
    
    storage = PredictionStorage(args.storage_dir)
    
    if args.action == 'summary':
        storage.summary()
    
    elif args.action == 'list':
        print("\nStored Models:")
        for model in storage.get_model_list():
            files = storage.get_file_list(model)
            print(f"  {model}: {len(files)} files")
    
    elif args.action == 'info':
        info = storage.get_storage_info()
        print(json.dumps(info, indent=2))
