#!/usr/bin/env python3
"""
Unified Analysis Utilities
===========================

Common functions for metrics, comparisons, and diagnostics.
Works with all model types.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score
import json
import logging

logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def calculate_domain_metrics(predictions: Dict, truth: Dict) -> Dict:
    """
    Calculate comprehensive domain-averaged metrics.
    
    Parameters
    ----------
    predictions : Dict
        Model predictions: {'visc_coeff': array, 'diff_coeff': array, ...}
    truth : Dict
        Ground truth: {'visc_coeff': array, 'diff_coeff': array, ...}
    
    Returns
    -------
    metrics : Dict
        Complete metrics dictionary
    """
    metrics = {}
    
    for var in ['visc_coeff', 'diff_coeff', 'richardson']:
        if var not in predictions or var not in truth:
            continue
        
        pred_flat = predictions[var].flatten()
        true_flat = truth[var].flatten()
        
        # Filter valid
        valid = ~(np.isnan(pred_flat) | np.isnan(true_flat) | 
                 np.isinf(pred_flat) | np.isinf(true_flat))
        
        pred_valid = pred_flat[valid]
        true_valid = true_flat[valid]
        
        if len(pred_valid) == 0:
            metrics[var] = {
                'r2': np.nan, 'rmse': np.nan, 'mae': np.nan,
                'bias': np.nan, 'variance_ratio': np.nan, 'n_valid': 0
            }
            continue
        
        # Calculate metrics
        r2 = r2_score(true_valid, pred_valid)
        rmse = np.sqrt(mean_squared_error(true_valid, pred_valid))
        mae = mean_absolute_error(true_valid, pred_valid)
        bias = np.mean(pred_valid - true_valid)
        
        var_pred = np.var(pred_valid)
        var_true = np.var(true_valid)
        var_ratio = var_pred / var_true if var_true > 0 else np.nan
        
        corr = np.corrcoef(true_valid, pred_valid)[0, 1]
        
        metrics[var] = {
            'r2': float(r2),
            'rmse': float(rmse),
            'mae': float(mae),
            'bias': float(bias),
            'variance_ratio': float(var_ratio),
            'correlation': float(corr),
            'n_valid': int(len(pred_valid)),
            'mean_pred': float(np.mean(pred_valid)),
            'mean_truth': float(np.mean(true_valid)),
            'std_pred': float(np.std(pred_valid)),
            'std_truth': float(np.std(true_valid))
        }
    
    # Regime classification (if available)
    if 'regime' in predictions and 'regime' in truth:
        pred_regime = predictions['regime'].flatten()
        true_regime = truth['regime'].flatten()
        
        valid = (pred_regime >= 0) & (true_regime >= 0)
        
        if np.sum(valid) > 0:
            accuracy = accuracy_score(true_regime[valid], pred_regime[valid])
            metrics['regime'] = {
                'accuracy': float(accuracy),
                'n_valid': int(np.sum(valid))
            }
    
    return metrics


def calculate_nonzero_metrics(predictions: Dict, truth: Dict,
                              threshold: float = 1e-10) -> Dict:
    """
    Calculate metrics for all data vs. non-zero data.
    
    Important for coefficients which can be zero in quiescent regions.
    """
    results = {}
    
    for var in ['visc_coeff', 'diff_coeff']:
        if var not in predictions or var not in truth:
            continue
        
        pred_flat = predictions[var].flatten()
        true_flat = truth[var].flatten()
        
        # All data
        valid_all = ~(np.isnan(pred_flat) | np.isnan(true_flat) | 
                     np.isinf(pred_flat) | np.isinf(true_flat))
        
        pred_all = pred_flat[valid_all]
        true_all = true_flat[valid_all]
        
        if len(pred_all) == 0:
            results[var] = {
                'all': {'r2': np.nan, 'rmse': np.nan, 'n': 0},
                'nonzero': {'r2': np.nan, 'rmse': np.nan, 'n': 0},
                'zero_fraction': np.nan
            }
            continue
        
        # All data metrics
        r2_all = r2_score(true_all, pred_all)
        rmse_all = np.sqrt(mean_squared_error(true_all, pred_all))
        mae_all = mean_absolute_error(true_all, pred_all)
        
        # Non-zero data
        nonzero_mask = np.abs(true_all) > threshold
        true_nonzero = true_all[nonzero_mask]
        pred_nonzero = pred_all[nonzero_mask]
        
        zero_fraction = 1 - (len(true_nonzero) / len(true_all))
        
        if len(true_nonzero) > 10:
            r2_nz = r2_score(true_nonzero, pred_nonzero)
            rmse_nz = np.sqrt(mean_squared_error(true_nonzero, pred_nonzero))
            mae_nz = mean_absolute_error(true_nonzero, pred_nonzero)
            rel_rmse = rmse_nz / np.mean(np.abs(true_nonzero))
        else:
            r2_nz = rmse_nz = mae_nz = rel_rmse = np.nan
        
        results[var] = {
            'all': {
                'r2': float(r2_all),
                'rmse': float(rmse_all),
                'mae': float(mae_all),
                'n_valid': int(len(pred_all))
            },
            'nonzero': {
                'r2': float(r2_nz),
                'rmse': float(rmse_nz),
                'mae': float(mae_nz),
                'relative_rmse': float(rel_rmse),
                'n_valid': int(len(true_nonzero))
            },
            'zero_fraction': float(zero_fraction),
            'threshold': float(threshold)
        }
    
    return results


def save_metrics_csv(all_metrics: Dict[str, Dict], output_file: Path,
                     metric_type: str = 'domain'):
    """
    Save metrics comparison table as CSV.
    
    Parameters
    ----------
    all_metrics : Dict[str, Dict]
        Nested: {'Model1': {'visc_coeff': {...}, ...}, 'Model2': {...}}
    output_file : Path
        Output CSV file
    metric_type : str
        'domain' or 'nonzero'
    """
    rows = []
    
    for model_name, metrics in all_metrics.items():
        for var in ['visc_coeff', 'diff_coeff', 'richardson']:
            if var not in metrics:
                continue
            
            var_label = var.replace('_coeff', '').replace('richardson', 'Ri').upper()
            
            if metric_type == 'domain':
                row = {
                    'Model': model_name,
                    'Variable': var_label,
                    'R²': metrics[var].get('r2', np.nan),
                    'RMSE': metrics[var].get('rmse', np.nan),
                    'MAE': metrics[var].get('mae', np.nan),
                    'Bias': metrics[var].get('bias', np.nan),
                    'Variance_Ratio': metrics[var].get('variance_ratio', np.nan),
                    'Correlation': metrics[var].get('correlation', np.nan),
                    'N_Points': metrics[var].get('n_valid', 0)
                }
            elif metric_type == 'nonzero':
                all_m = metrics[var].get('all', {})
                nz_m = metrics[var].get('nonzero', {})
                
                row = {
                    'Model': model_name,
                    'Variable': var_label,
                    'R²_All': all_m.get('r2', np.nan),
                    'R²_NonZero': nz_m.get('r2', np.nan),
                    'RMSE_All': all_m.get('rmse', np.nan),
                    'RMSE_NonZero': nz_m.get('rmse', np.nan),
                    'Zero_Fraction_%': metrics[var].get('zero_fraction', 0) * 100,
                    'N_All': all_m.get('n_valid', 0),
                    'N_NonZero': nz_m.get('n_valid', 0)
                }
            
            rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False, float_format='%.6f')
    logger.info(f"✓ Saved metrics CSV: {output_file.name}")
    
    return df


def create_comparison_table(all_metrics: Dict[str, Dict],
                            variables: List[str] = ['visc_coeff', 'diff_coeff']) -> pd.DataFrame:
    """
    Create publication-ready comparison table.
    
    Returns DataFrame with models as rows, metrics as columns.
    """
    rows = []
    
    for model_name in sorted(all_metrics.keys()):
        metrics = all_metrics[model_name]
        
        row = {'Model': model_name}
        
        for var in variables:
            if var not in metrics:
                continue
            
            var_short = var.replace('_coeff', '').upper()
            
            row[f'{var_short}_R²'] = metrics[var].get('r2', np.nan)
            row[f'{var_short}_RMSE'] = metrics[var].get('rmse', np.nan)
            row[f'{var_short}_MAE'] = metrics[var].get('mae', np.nan)
            row[f'{var_short}_Bias'] = metrics[var].get('bias', np.nan)
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


def identify_best_models(all_metrics: Dict[str, Dict],
                         criterion: str = 'r2') -> Dict[str, str]:
    """
    Identify best model for each variable based on criterion.
    
    Parameters
    ----------
    criterion : str
        'r2' (higher better), 'rmse' (lower better), 'mae' (lower better)
    
    Returns
    -------
    best_models : Dict[str, str]
        {'visc_coeff': 'ModelName', 'diff_coeff': 'ModelName', ...}
    """
    best_models = {}
    
    for var in ['visc_coeff', 'diff_coeff', 'richardson']:
        scores = {}
        
        for model_name, metrics in all_metrics.items():
            if var not in metrics:
                continue
            
            score = metrics[var].get(criterion, np.nan)
            if not np.isnan(score):
                scores[model_name] = score
        
        if not scores:
            continue
        
        # Higher is better for R², correlation
        # Lower is better for RMSE, MAE
        if criterion in ['r2', 'correlation']:
            best_model = max(scores.items(), key=lambda x: x[1])
        else:
            best_model = min(scores.items(), key=lambda x: x[1])
        
        best_models[var] = best_model[0]
        logger.info(f"Best {var} ({criterion}): {best_model[0]} = {best_model[1]:.4f}")
    
    return best_models


def print_metrics_summary(all_metrics: Dict[str, Dict]):
    """Print formatted summary of metrics."""
    
    print("\n" + "="*90)
    print("METRICS SUMMARY")
    print("="*90 + "\n")
    
    for model_name in sorted(all_metrics.keys()):
        print(f"{model_name}:")
        metrics = all_metrics[model_name]
        
        for var in ['visc_coeff', 'diff_coeff', 'richardson']:
            if var not in metrics:
                continue
            
            var_label = var.replace('_coeff', '').upper()
            m = metrics[var]
            
            print(f"  {var_label:6s}: R²={m.get('r2', np.nan):7.4f}  "
                  f"RMSE={m.get('rmse', np.nan):8.4f}  "
                  f"MAE={m.get('mae', np.nan):8.4f}  "
                  f"Bias={m.get('bias', np.nan):+8.4f}")
        
        print()
    
    print("="*90 + "\n")
