#!/usr/bin/env python3
"""
Unified Plotting Utilities
===========================

Consistent plotting functions for all model types.
Publication-ready visualizations with proper styling.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde
import logging

logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", context="paper")


# Color schemes
MODEL_COLORS = {
    # Baseline
    'Baseline-MLP': '#2E86AB',
    'Baseline-ResMLP': '#06A77D',
    'Baseline-TabTransformer': '#A23B72',
    
    # Ri-conditioned
    'Ri-MLP': '#E63946',
    'Ri-ResMLP': '#F77F00',
    'Ri-TabTransformer': '#8338EC',
    
    # Q1
    'Q1-MLP': '#118AB2',
    'Q1-ResMLP': '#073B4C',
    'Q1-TabTransformer': '#06D6A0',
    
    # Q2
    'Q2-MLP': '#FFD60A',
    'Q2-ResMLP': '#FFC300',
    'Q2-TabTransformer': '#FFB700',
    
    # Q3
    'Q3-MLP': '#C1121F',
    'Q3-ResMLP': '#780000',
    'Q3-TabTransformer': '#9D0208',
    
    # Q4
    'Q4-MLP': '#6A4C93',
    'Q4-ResMLP': '#8B5FBF',
    'Q4-TabTransformer': '#A75FC9'
}

LINE_STYLES = {
    'MLP': '-',
    'ResMLP': '--',
    'TabTransformer': '-.'
}


def get_model_style(model_name: str) -> Tuple[str, str]:
    """
    Get color and line style for a model.
    
    Returns
    -------
    color : str
        Hex color code
    linestyle : str
        Line style
    """
    # Try exact match
    if model_name in MODEL_COLORS:
        color = MODEL_COLORS[model_name]
    else:
        # Fallback: use hash for consistent color
        import hashlib
        hash_val = int(hashlib.md5(model_name.encode()).hexdigest()[:6], 16)
        color = f"#{hash_val:06x}"
    
    # Extract architecture for line style
    for arch in ['MLP', 'ResMLP', 'TabTransformer']:
        if arch in model_name:
            linestyle = LINE_STYLES[arch]
            break
    else:
        linestyle = '-'
    
    return color, linestyle


def plot_metrics_comparison_bar(all_metrics: Dict[str, Dict], 
                                output_dir: Path,
                                metric: str = 'r2',
                                variables: List[str] = ['visc_coeff', 'diff_coeff']):
    """
    Create bar chart comparing a specific metric across all models.
    
    Parameters
    ----------
    metric : str
        'r2', 'rmse', 'mae', 'bias', 'correlation'
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_names = sorted(all_metrics.keys())
    n_models = len(model_names)
    n_vars = len(variables)
    
    fig, axes = plt.subplots(1, n_vars, figsize=(6*n_vars, max(6, n_models*0.4)))
    
    if n_vars == 1:
        axes = [axes]
    
    for ax, var in zip(axes, variables):
        var_label = var.replace('_coeff', '').upper()
        
        values = []
        colors_list = []
        
        for model_name in model_names:
            if var in all_metrics[model_name]:
                val = all_metrics[model_name][var].get(metric, np.nan)
                values.append(val)
                color, _ = get_model_style(model_name)
                colors_list.append(color)
            else:
                values.append(np.nan)
                colors_list.append('#CCCCCC')
        
        y_pos = np.arange(n_models)
        bars = ax.barh(y_pos, values, color=colors_list, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, values)):
            if not np.isnan(val):
                ax.text(val, bar.get_y() + bar.get_height()/2,
                       f' {val:.3f}', va='center', fontsize=8, fontweight='bold')
        
        # Reference lines
        if metric == 'r2':
            ax.axvline(x=0.7, color='green', linestyle='--', linewidth=1.5, alpha=0.5, label='Target (0.70)')
            ax.axvline(x=0.85, color='gold', linestyle='--', linewidth=1.5, alpha=0.5, label='Excellent (0.85)')
            ax.set_xlim([0, 1.0])
        elif metric == 'correlation':
            ax.axvline(x=0.9, color='green', linestyle='--', linewidth=1.5, alpha=0.5, label='Target (0.90)')
            ax.set_xlim([0, 1.0])
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([name.replace('Baseline-', 'B-').replace('Ri-', 'Ri-') 
                           for name in model_names], fontsize=9)
        ax.set_xlabel(metric.upper(), fontsize=11, fontweight='bold')
        ax.set_title(f'{var_label}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        ax.legend(loc='lower right', fontsize=8)
    
    plt.suptitle(f'{metric.upper()} Comparison - All Models', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_file = output_dir / f'metrics_comparison_{metric}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"✓ Saved: {output_file.name}")


def plot_scatter_comparison_grid(all_predictions: Dict[str, Dict],
                                 truth: Dict,
                                 output_dir: Path,
                                 variable: str = 'visc_coeff',
                                 max_models_per_plot: int = 9,
                                 downsample: int = 100000):
    """
    Create grid of scatter plots comparing all models.
    
    Parameters
    ----------
    max_models_per_plot : int
        Maximum models in one figure (will create multiple figures if needed)
    downsample : int
        Maximum points to plot (for performance)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    var_label = variable.replace('_coeff', '').title()
    model_names = sorted(all_predictions.keys())
    n_models = len(model_names)
    
    # Split into multiple figures if too many models
    n_figures = int(np.ceil(n_models / max_models_per_plot))
    
    true_flat = truth[variable].flatten()
    valid_truth = ~(np.isnan(true_flat) | np.isinf(true_flat))
    true_valid = true_flat[valid_truth]
    
    # Downsample for plotting
    if len(true_valid) > downsample:
        indices = np.random.choice(len(true_valid), downsample, replace=False)
        true_plot = true_valid[indices]
    else:
        indices = slice(None)
        true_plot = true_valid
    
    for fig_idx in range(n_figures):
        start_idx = fig_idx * max_models_per_plot
        end_idx = min(start_idx + max_models_per_plot, n_models)
        models_in_fig = model_names[start_idx:end_idx]
        
        n_models_fig = len(models_in_fig)
        ncols = int(np.ceil(np.sqrt(n_models_fig)))
        nrows = int(np.ceil(n_models_fig / ncols))
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4.5*nrows))
        
        if nrows == 1 and ncols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        data_min = np.percentile(true_plot, 1)
        data_max = np.percentile(true_plot, 99)
        
        for idx, model_name in enumerate(models_in_fig):
            ax = axes[idx]
            
            pred_flat = all_predictions[model_name][variable].flatten()
            pred_flat = pred_flat[valid_truth]
            pred_plot = pred_flat[indices] if isinstance(indices, np.ndarray) else pred_flat
            
            # Filter extremes
            valid_plot = ~(np.isnan(pred_plot) | np.isinf(pred_plot))
            pred_plot = pred_plot[valid_plot]
            true_plot_filtered = true_plot[valid_plot]
            
            if len(pred_plot) == 0:
                ax.text(0.5, 0.5, 'No valid data', ha='center', va='center')
                ax.set_title(model_name, fontsize=10, fontweight='bold')
                continue
            
            # Hexbin
            hexbin = ax.hexbin(true_plot_filtered, pred_plot, 
                             gridsize=50, cmap='Blues',
                             norm=LogNorm(vmin=1, vmax=max(10, len(pred_plot)/50)),
                             mincnt=1, alpha=0.9, extent=(data_min, data_max, data_min, data_max))
            
            # 1:1 line
            ax.plot([data_min, data_max], [data_min, data_max], 
                   'r--', linewidth=2, alpha=0.8, label='1:1', zorder=10)
            
            # Metrics
            from sklearn.metrics import r2_score, mean_squared_error
            r2 = r2_score(true_plot_filtered, pred_plot)
            rmse = np.sqrt(mean_squared_error(true_plot_filtered, pred_plot))
            
            stats_text = f'R² = {r2:.3f}\nRMSE = {rmse:.2f}\nN = {len(pred_plot):,}'
            
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                   fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                   family='monospace')
            
            # Styling
            short_name = model_name.replace('Baseline-', 'B-').replace('Ri-', 'Ri-')
            ax.set_title(short_name, fontsize=10, fontweight='bold')
            ax.set_xlabel('MONC Truth', fontsize=9)
            ax.set_ylabel('Prediction', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlim([data_min, data_max])
            ax.set_ylim([data_min, data_max])
        
        # Hide unused subplots
        for idx in range(len(models_in_fig), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'{var_label} - Scatter Comparison (Part {fig_idx+1}/{n_figures})',
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        output_file = output_dir / f'scatter_grid_{variable}_part{fig_idx+1}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"✓ Saved: {output_file.name}")


def plot_vertical_profiles_comparison(all_predictions: Dict[str, Dict],
                                      truth: Dict,
                                      output_dir: Path,
                                      variable: str = 'visc_coeff',
                                      group_by: str = 'architecture'):
    """
    Plot vertical profiles grouped by architecture or configuration.
    
    Parameters
    ----------
    group_by : str
        'architecture' (MLP, ResMLP, TabTransformer) or 
        'configuration' (Baseline, Ri, Q1, Q2, Q3, Q4)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    var_label = variable.replace('_coeff', '').title()
    
    # Extract heights from truth
    nz = truth[variable].shape[2]
    heights = np.arange(nz)
    
    # Compute profiles
    truth_profile = np.nanmean(truth[variable], axis=(0, 1))
    
    model_profiles = {}
    for model_name, preds in all_predictions.items():
        if variable in preds:
            model_profiles[model_name] = np.nanmean(preds[variable], axis=(0, 1))
    
    # Group models
    if group_by == 'architecture':
        groups = {'MLP': [], 'ResMLP': [], 'TabTransformer': []}
        for model_name in model_profiles.keys():
            for arch in groups.keys():
                if arch in model_name:
                    groups[arch].append(model_name)
                    break
    else:  # group by configuration
        groups = {}
        for model_name in model_profiles.keys():
            config = model_name.split('-')[0]
            if config not in groups:
                groups[config] = []
            groups[config].append(model_name)
    
    # Plot each group
    for group_name, model_list in groups.items():
        if not model_list:
            continue
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 6))
        
        # Panel 1: Absolute values
        ax = axes[0]
        ax.plot(truth_profile, heights, 'k-', linewidth=3.5, 
               label='MONC Truth', alpha=0.9, zorder=100)
        
        for model_name in sorted(model_list):
            color, linestyle = get_model_style(model_name)
            profile = model_profiles[model_name]
            
            short_name = model_name.replace('Baseline-', 'B-').replace('Ri-', 'Ri-')
            ax.plot(profile, heights, linestyle, color=color, linewidth=2.5,
                   label=short_name, alpha=0.85)
        
        ax.set_xlabel(var_label, fontsize=11, fontweight='bold')
        ax.set_ylabel('Vertical Level (k)', fontsize=11, fontweight='bold')
        ax.set_title('Vertical Profiles', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        # Panel 2: Absolute error
        ax = axes[1]
        for model_name in sorted(model_list):
            color, linestyle = get_model_style(model_name)
            error = model_profiles[model_name] - truth_profile
            
            short_name = model_name.replace('Baseline-', 'B-').replace('Ri-', 'Ri-')
            ax.plot(error, heights, linestyle, color=color, linewidth=2.5,
                   label=short_name, alpha=0.85)
        
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('Error (Pred - Truth)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Vertical Level (k)', fontsize=11, fontweight='bold')
        ax.set_title('Error Profile', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        # Panel 3: Relative error (%)
        ax = axes[2]
        for model_name in sorted(model_list):
            color, linestyle = get_model_style(model_name)
            rel_error = (model_profiles[model_name] - truth_profile) / (np.abs(truth_profile) + 1e-10) * 100
            
            short_name = model_name.replace('Baseline-', 'B-').replace('Ri-', 'Ri-')
            ax.plot(rel_error, heights, linestyle, color=color, linewidth=2.5,
                   label=short_name, alpha=0.85)
        
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('Relative Error (%)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Vertical Level (k)', fontsize=11, fontweight='bold')
        ax.set_title('Relative Error Profile', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        fig.suptitle(f'{var_label} - Vertical Profiles ({group_name})',
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        output_file = output_dir / f'vertical_profiles_{variable}_{group_name}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"✓ Saved: {output_file.name}")


def plot_distribution_comparison(all_predictions: Dict[str, Dict],
                                 truth: Dict,
                                 output_dir: Path,
                                 variable: str = 'visc_coeff',
                                 log_scale: bool = True,
                                 max_models_per_plot: int = 6):
    """
    Plot probability distributions (KDE) for all models.
    
    Parameters
    ----------
    log_scale : bool
        If True, plot in log10 space
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    var_label = variable.replace('_coeff', '').title()
    model_names = sorted(all_predictions.keys())
    
    # Prepare truth
    true_flat = truth[variable].flatten()
    valid = ~(np.isnan(true_flat) | np.isinf(true_flat))
    true_valid = true_flat[valid]
    
    if log_scale:
        epsilon = 1e-6
        true_valid = np.log10(np.maximum(true_valid, epsilon))
        x_label = f'Log10({var_label})'
    else:
        x_label = var_label
    
    # Split into multiple plots if too many models
    n_figures = int(np.ceil(len(model_names) / max_models_per_plot))
    
    for fig_idx in range(n_figures):
        start_idx = fig_idx * max_models_per_plot
        end_idx = min(start_idx + max_models_per_plot, len(model_names))
        models_in_fig = model_names[start_idx:end_idx]
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Plot truth
        sns.kdeplot(true_valid, ax=ax, color='black', linestyle='--',
                   linewidth=3, label='MONC Truth', alpha=0.9)
        
        # Plot predictions
        for model_name in models_in_fig:
            pred_flat = all_predictions[model_name][variable].flatten()
            pred_flat = pred_flat[valid]
            
            if log_scale:
                pred_flat = np.log10(np.maximum(pred_flat, epsilon))
            
            color, linestyle = get_model_style(model_name)
            short_name = model_name.replace('Baseline-', 'B-').replace('Ri-', 'Ri-')
            
            sns.kdeplot(pred_flat, ax=ax, color=color, linestyle=linestyle,
                       linewidth=2.5, label=short_name, alpha=0.85)
        
        ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
        ax.set_ylabel('Density', fontsize=12, fontweight='bold')
        ax.set_title(f'{var_label} Distribution Comparison (Part {fig_idx+1}/{n_figures})',
                    fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=10, framealpha=0.95, ncol=2)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        scale_str = 'log' if log_scale else 'linear'
        output_file = output_dir / f'distribution_{variable}_{scale_str}_part{fig_idx+1}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"✓ Saved: {output_file.name}")


def create_summary_figure(all_metrics: Dict[str, Dict],
                         output_dir: Path,
                         variables: List[str] = ['visc_coeff', 'diff_coeff']):
    """
    Create a comprehensive summary figure with key metrics.
    """
    output_dir = Path(output_dir)
    
    model_names = sorted(all_metrics.keys())
    n_models = len(model_names)
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # Top row: R² comparison
    for i, var in enumerate(variables):
        ax = fig.add_subplot(gs[0, i*2:(i+1)*2])
        
        r2_values = [all_metrics[m][var]['r2'] if var in all_metrics[m] else np.nan 
                    for m in model_names]
        colors = [get_model_style(m)[0] for m in model_names]
        
        bars = ax.barh(range(n_models), r2_values, color=colors, alpha=0.8)
        
        ax.axvline(x=0.7, color='green', linestyle='--', alpha=0.5, label='Target')
        ax.axvline(x=0.85, color='gold', linestyle='--', alpha=0.5, label='Excellent')
        
        ax.set_yticks(range(n_models))
        ax.set_yticklabels([m.replace('Baseline-', 'B-').replace('Ri-', 'Ri-') 
                           for m in model_names], fontsize=8)
        ax.set_xlabel('R²', fontsize=11, fontweight='bold')
        ax.set_title(f'{var.replace("_coeff", "").upper()} - R²',
                    fontsize=12, fontweight='bold')
        ax.set_xlim([0, 1])
        ax.grid(True, alpha=0.3, axis='x')
        ax.legend(fontsize=8)
    
    # Bottom row: RMSE and Variance Ratio
    for i, var in enumerate(variables):
        # RMSE
        ax = fig.add_subplot(gs[1, i*2])
        
        rmse_values = [all_metrics[m][var]['rmse'] if var in all_metrics[m] else np.nan 
                      for m in model_names]
        colors = [get_model_style(m)[0] for m in model_names]
        
        ax.barh(range(n_models), rmse_values, color=colors, alpha=0.8)
        ax.set_yticks(range(n_models))
        ax.set_yticklabels([m.replace('Baseline-', 'B-').replace('Ri-', 'Ri-') 
                           for m in model_names], fontsize=8)
        ax.set_xlabel('RMSE', fontsize=10, fontweight='bold')
        ax.set_title(f'{var.replace("_coeff", "").upper()} - RMSE',
                    fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Variance Ratio
        ax = fig.add_subplot(gs[1, i*2+1])
        
        var_ratio = [all_metrics[m][var].get('variance_ratio', np.nan) 
                    if var in all_metrics[m] else np.nan 
                    for m in model_names]
        
        ax.barh(range(n_models), var_ratio, color=colors, alpha=0.8)
        ax.axvline(x=1.0, color='black', linestyle='--', alpha=0.5, label='Perfect')
        ax.set_yticks(range(n_models))
        ax.set_yticklabels([m.replace('Baseline-', 'B-').replace('Ri-', 'Ri-') 
                           for m in model_names], fontsize=8)
        ax.set_xlabel('Variance Ratio', fontsize=10, fontweight='bold')
        ax.set_title(f'{var.replace("_coeff", "").upper()} - Var Ratio',
                    fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        ax.legend(fontsize=8)
    
    plt.suptitle('Model Performance Summary', fontsize=16, fontweight='bold', y=0.98)
    
    output_file = output_dir / 'summary_figure.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"✓ Saved: {output_file.name}")
