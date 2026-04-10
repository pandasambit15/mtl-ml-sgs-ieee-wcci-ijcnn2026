#!/usr/bin/env python3
"""
Advanced Metrics and Statistical Analysis
==========================================

Additional analysis tools:
- Statistical significance tests (t-test, Wilcoxon, bootstrap)
- Taylor diagrams
- Skill scores (Nash-Sutcliffe, Kling-Gupta, Index of Agreement)
- Confidence intervals
- Model ranking with uncertainty
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.stats import t as t_dist
import pandas as pd
import logging

logger = logging.getLogger(__name__)


# ==================== STATISTICAL SIGNIFICANCE TESTS ====================

def paired_ttest(predictions1: np.ndarray, predictions2: np.ndarray, 
                truth: np.ndarray, alpha: float = 0.05) -> Dict:
    """
    Paired t-test to determine if two models have significantly different errors.
    
    H0: Mean squared errors are equal
    H1: Mean squared errors are different
    
    Parameters
    ----------
    predictions1, predictions2 : np.ndarray
        Model predictions
    truth : np.ndarray
        Ground truth
    alpha : float
        Significance level (default: 0.05)
    
    Returns
    -------
    result : Dict
        {
            'statistic': t-statistic,
            'p_value': p-value,
            'significant': bool,
            'mean_diff': mean difference in SE,
            'ci_lower': lower bound of 95% CI,
            'ci_upper': upper bound of 95% CI
        }
    """
    # Compute squared errors
    se1 = (predictions1 - truth) ** 2
    se2 = (predictions2 - truth) ** 2
    
    # Paired differences
    diff = se1 - se2
    
    # Filter valid
    valid = ~(np.isnan(diff) | np.isinf(diff))
    diff_valid = diff[valid]
    
    if len(diff_valid) < 30:
        logger.warning(f"Small sample size: {len(diff_valid)}")
    
    # t-test
    statistic, p_value = stats.ttest_1samp(diff_valid, 0)
    
    # Confidence interval
    mean_diff = np.mean(diff_valid)
    std_diff = np.std(diff_valid, ddof=1)
    se_diff = std_diff / np.sqrt(len(diff_valid))
    
    t_crit = t_dist.ppf(1 - alpha/2, len(diff_valid) - 1)
    ci_lower = mean_diff - t_crit * se_diff
    ci_upper = mean_diff + t_crit * se_diff
    
    return {
        'statistic': float(statistic),
        'p_value': float(p_value),
        'significant': p_value < alpha,
        'mean_diff': float(mean_diff),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'n_samples': int(len(diff_valid)),
        'effect_size': float(mean_diff / std_diff) if std_diff > 0 else np.nan
    }


def wilcoxon_test(predictions1: np.ndarray, predictions2: np.ndarray,
                 truth: np.ndarray, alpha: float = 0.05) -> Dict:
    """
    Wilcoxon signed-rank test (non-parametric alternative to t-test).
    
    More robust when errors are not normally distributed.
    """
    # Compute absolute errors
    ae1 = np.abs(predictions1 - truth)
    ae2 = np.abs(predictions2 - truth)
    
    # Paired differences
    diff = ae1 - ae2
    
    # Filter valid
    valid = ~(np.isnan(diff) | np.isinf(diff))
    diff_valid = diff[valid]
    
    # Wilcoxon test
    statistic, p_value = stats.wilcoxon(diff_valid, alternative='two-sided')
    
    return {
        'statistic': float(statistic),
        'p_value': float(p_value),
        'significant': p_value < alpha,
        'median_diff': float(np.median(diff_valid)),
        'n_samples': int(len(diff_valid))
    }


def bootstrap_confidence_interval(predictions: np.ndarray, truth: np.ndarray,
                                  metric_func, n_bootstrap: int = 1000,
                                  confidence: float = 0.95) -> Dict:
    """
    Bootstrap confidence interval for any metric.
    
    Parameters
    ----------
    metric_func : callable
        Function that computes metric: metric_func(pred, truth) -> float
    n_bootstrap : int
        Number of bootstrap samples
    confidence : float
        Confidence level (e.g., 0.95 for 95% CI)
    
    Returns
    -------
    result : Dict
        {
            'mean': mean of bootstrap distribution,
            'std': standard error,
            'ci_lower': lower confidence bound,
            'ci_upper': upper confidence bound
        }
    """
    # Filter valid
    valid = ~(np.isnan(predictions) | np.isnan(truth) | 
             np.isinf(predictions) | np.isinf(truth))
    pred_valid = predictions[valid]
    truth_valid = truth[valid]
    
    n_samples = len(pred_valid)
    
    # Bootstrap
    bootstrap_metrics = []
    
    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n_samples, n_samples, replace=True)
        
        pred_boot = pred_valid[indices]
        truth_boot = truth_valid[indices]
        
        # Compute metric
        metric = metric_func(pred_boot, truth_boot)
        bootstrap_metrics.append(metric)
    
    bootstrap_metrics = np.array(bootstrap_metrics)
    
    # Confidence interval
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_metrics, alpha/2 * 100)
    ci_upper = np.percentile(bootstrap_metrics, (1 - alpha/2) * 100)
    
    return {
        'mean': float(np.mean(bootstrap_metrics)),
        'std': float(np.std(bootstrap_metrics)),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'confidence': confidence
    }


def pairwise_significance_matrix(all_predictions: Dict[str, Dict],
                                 truth: Dict,
                                 variable: str = 'visc_coeff',
                                 test: str = 'ttest',
                                 alpha: float = 0.05) -> pd.DataFrame:
    """
    Create pairwise significance matrix for all models.
    
    Parameters
    ----------
    test : str
        'ttest' or 'wilcoxon'
    
    Returns
    -------
    matrix : pd.DataFrame
        N×N matrix where entry (i,j) is p-value for models i vs j
    """
    model_names = sorted(all_predictions.keys())
    n_models = len(model_names)
    
    # Initialize matrix
    p_matrix = np.ones((n_models, n_models))
    
    truth_flat = truth[variable].flatten()
    
    for i, model1 in enumerate(model_names):
        pred1 = all_predictions[model1][variable].flatten()
        
        for j, model2 in enumerate(model_names):
            if i >= j:
                continue
            
            pred2 = all_predictions[model2][variable].flatten()
            
            if test == 'ttest':
                result = paired_ttest(pred1, pred2, truth_flat, alpha)
            else:
                result = wilcoxon_test(pred1, pred2, truth_flat, alpha)
            
            p_matrix[i, j] = result['p_value']
            p_matrix[j, i] = result['p_value']
    
    # Create DataFrame
    df = pd.DataFrame(
        p_matrix,
        index=[m.replace('Baseline-', 'B-').replace('Ri-', 'Ri-') for m in model_names],
        columns=[m.replace('Baseline-', 'B-').replace('Ri-', 'Ri-') for m in model_names]
    )
    
    return df


# ==================== SKILL SCORES ====================

def nash_sutcliffe_efficiency(predictions: np.ndarray, truth: np.ndarray) -> float:
    """
    Nash-Sutcliffe Efficiency (NSE).
    
    NSE = 1 - Σ(obs - pred)² / Σ(obs - mean_obs)²
    
    Range: (-∞, 1]
    - NSE = 1: Perfect prediction
    - NSE = 0: As good as mean
    - NSE < 0: Worse than mean
    """
    valid = ~(np.isnan(predictions) | np.isnan(truth) | 
             np.isinf(predictions) | np.isinf(truth))
    
    pred = predictions[valid]
    obs = truth[valid]
    
    numerator = np.sum((obs - pred) ** 2)
    denominator = np.sum((obs - np.mean(obs)) ** 2)
    
    if denominator == 0:
        return np.nan
    
    nse = 1 - (numerator / denominator)
    
    return float(nse)


def kling_gupta_efficiency(predictions: np.ndarray, truth: np.ndarray) -> Dict:
    """
    Kling-Gupta Efficiency (KGE).
    
    KGE = 1 - √[(r-1)² + (α-1)² + (β-1)²]
    
    where:
    - r = correlation coefficient
    - α = ratio of std devs (σ_pred / σ_obs)
    - β = ratio of means (μ_pred / μ_obs)
    
    Range: (-∞, 1]
    - KGE = 1: Perfect prediction
    - KGE > -0.41: Better than mean
    """
    valid = ~(np.isnan(predictions) | np.isnan(truth) | 
             np.isinf(predictions) | np.isinf(truth))
    
    pred = predictions[valid]
    obs = truth[valid]
    
    # Correlation
    r = np.corrcoef(obs, pred)[0, 1]
    
    # Variability ratio
    alpha = np.std(pred) / np.std(obs) if np.std(obs) > 0 else np.nan
    
    # Bias ratio
    beta = np.mean(pred) / np.mean(obs) if np.mean(obs) != 0 else np.nan
    
    # KGE
    kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    
    return {
        'kge': float(kge),
        'r': float(r),
        'alpha': float(alpha),
        'beta': float(beta)
    }


def index_of_agreement(predictions: np.ndarray, truth: np.ndarray) -> float:
    """
    Index of Agreement (Willmott's d).
    
    d = 1 - Σ(obs - pred)² / Σ(|pred - mean_obs| + |obs - mean_obs|)²
    
    Range: [0, 1]
    - d = 1: Perfect agreement
    - d = 0: No agreement
    """
    valid = ~(np.isnan(predictions) | np.isnan(truth) | 
             np.isinf(predictions) | np.isinf(truth))
    
    pred = predictions[valid]
    obs = truth[valid]
    
    mean_obs = np.mean(obs)
    
    numerator = np.sum((obs - pred) ** 2)
    denominator = np.sum((np.abs(pred - mean_obs) + np.abs(obs - mean_obs)) ** 2)
    
    if denominator == 0:
        return np.nan
    
    d = 1 - (numerator / denominator)
    
    return float(d)


def percent_bias(predictions: np.ndarray, truth: np.ndarray) -> float:
    """
    Percent Bias (PBIAS).
    
    PBIAS = 100 × Σ(obs - pred) / Σ(obs)
    
    - PBIAS = 0: Perfect
    - PBIAS > 0: Underestimation
    - PBIAS < 0: Overestimation
    """
    valid = ~(np.isnan(predictions) | np.isnan(truth) | 
             np.isinf(predictions) | np.isinf(truth))
    
    pred = predictions[valid]
    obs = truth[valid]
    
    numerator = np.sum(obs - pred)
    denominator = np.sum(obs)
    
    if denominator == 0:
        return np.nan
    
    pbias = 100 * (numerator / denominator)
    
    return float(pbias)


def compute_all_skill_scores(predictions: np.ndarray, truth: np.ndarray) -> Dict:
    """Compute all skill scores for a model."""
    
    scores = {
        'nse': nash_sutcliffe_efficiency(predictions, truth),
        'kge_dict': kling_gupta_efficiency(predictions, truth),
        'index_of_agreement': index_of_agreement(predictions, truth),
        'pbias': percent_bias(predictions, truth)
    }
    
    # Flatten KGE dict
    scores['kge'] = scores['kge_dict']['kge']
    scores['kge_r'] = scores['kge_dict']['r']
    scores['kge_alpha'] = scores['kge_dict']['alpha']
    scores['kge_beta'] = scores['kge_dict']['beta']
    del scores['kge_dict']
    
    return scores


# ==================== TAYLOR DIAGRAM ====================

class TaylorDiagram:
    """
    Taylor diagram for visualizing model performance.
    
    Shows correlation, standard deviation, and RMSE in a single plot.
    """
    
    def __init__(self, std_truth: float, fig=None, rect=111, label='_'):
        """
        Initialize Taylor diagram.
        
        Parameters
        ----------
        std_truth : float
            Standard deviation of truth/observations
        fig : matplotlib.figure.Figure
            Figure to add diagram to (creates new if None)
        rect : int
            Subplot position (e.g., 111)
        """
        import matplotlib.pyplot as plt
        from matplotlib.projections import PolarAxes
        import mpl_toolkits.axisartist.floating_axes as FA
        import mpl_toolkits.axisartist.grid_finder as GF
        
        self.std_truth = std_truth
        
        # Correlation range
        tr = PolarAxes.PolarTransform()
        
        # Correlation labels
        rlocs = np.array([0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0])
        tlocs = np.arccos(rlocs)
        gl1 = GF.FixedLocator(tlocs)
        tf1 = GF.DictFormatter(dict(zip(tlocs, map(str, rlocs))))
        
        # Standard deviation range
        max_std = 2.0 * std_truth
        rlocs_std = np.linspace(0, max_std, 5)
        gl2 = GF.FixedLocator(rlocs_std)
        tf2 = GF.DictFormatter(dict(zip(rlocs_std, map(str, rlocs_std))))
        
        # Grid helper
        ghelper = FA.GridHelperCurveLinear(
            tr,
            extremes=(0, np.pi/2, 0, max_std),
            grid_locator1=gl1,
            grid_locator2=gl2,
            tick_formatter1=tf1,
            tick_formatter2=tf2
        )
        
        if fig is None:
            fig = plt.figure(figsize=(8, 8))
        
        ax = FA.FloatingSubplot(fig, rect, grid_helper=ghelper)
        fig.add_subplot(ax)
        
        # Adjust axes
        ax.axis["top"].set_axis_direction("bottom")
        ax.axis["top"].toggle(ticklabels=True, label=True)
        ax.axis["top"].major_ticklabels.set_axis_direction("top")
        ax.axis["top"].label.set_axis_direction("top")
        ax.axis["top"].label.set_text("Correlation")
        
        ax.axis["left"].set_axis_direction("bottom")
        ax.axis["left"].label.set_text("Standard Deviation")
        
        ax.axis["right"].set_axis_direction("top")
        ax.axis["right"].toggle(ticklabels=True)
        ax.axis["right"].major_ticklabels.set_axis_direction("left")
        
        ax.axis["bottom"].set_visible(False)
        
        self._ax = ax
        self.ax = ax.get_aux_axes(tr)
        
        # Add reference point (truth)
        self.ax.plot([0], [std_truth], 'k*', markersize=15, label='Reference', zorder=100)
        
        # Add std dev arcs
        for std in [0.5, 1.0, 1.5]:
            self.ax.plot(np.linspace(0, np.pi/2, 100), 
                        [std * std_truth] * 100,
                        'k:', alpha=0.3, linewidth=0.5)
        
        # Add RMSE contours
        rs, ts = np.meshgrid(np.linspace(0, max_std, 100),
                            np.linspace(0, np.pi/2, 100))
        
        # RMSE = sqrt(std_pred² + std_obs² - 2*std_pred*std_obs*corr)
        rms = np.sqrt(std_truth**2 + rs**2 - 2*std_truth*rs*np.cos(ts))
        
        contours = self.ax.contour(ts, rs, rms, levels=5, colors='gray', 
                                   alpha=0.4, linewidths=0.5)
        self.ax.clabel(contours, inline=True, fontsize=8, fmt='%.2f')
        
        self.sample_points = []
    
    def add_sample(self, std_pred: float, correlation: float, 
                  label: str = '', color: str = 'blue', marker: str = 'o'):
        """
        Add a model to the diagram.
        
        Parameters
        ----------
        std_pred : float
            Standard deviation of predictions
        correlation : float
            Correlation between predictions and truth
        label : str
            Model label
        color : str
            Marker color
        marker : str
            Marker style
        """
        theta = np.arccos(correlation)
        
        point = self.ax.plot(theta, std_pred, marker, color=color, 
                            markersize=10, label=label, alpha=0.8, 
                            markeredgecolor='black', markeredgewidth=0.5)
        
        self.sample_points.append(point[0])
        
        return point
    
    def add_legend(self, **kwargs):
        """Add legend to diagram."""
        self.ax.legend(**kwargs)
    
    def set_title(self, title: str, **kwargs):
        """Set diagram title."""
        self._ax.set_title(title, **kwargs)


def plot_taylor_diagram(all_predictions: Dict[str, Dict],
                       truth: Dict,
                       output_dir: Path,
                       variable: str = 'visc_coeff'):
    """
    Create Taylor diagram for all models.
    
    Parameters
    ----------
    all_predictions : Dict[str, Dict]
        Nested dict: {'Model1': {'visc_coeff': array, ...}, ...}
    truth : Dict
        Ground truth
    output_dir : Path
        Output directory
    variable : str
        Variable to plot
    """
    from plotting_utils import get_model_style
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    var_label = variable.replace('_coeff', '').title()
    
    # Prepare truth
    truth_flat = truth[variable].flatten()
    valid = ~(np.isnan(truth_flat) | np.isinf(truth_flat))
    truth_valid = truth_flat[valid]
    
    std_truth = np.std(truth_valid)
    
    # Create diagram
    fig = plt.figure(figsize=(10, 10))
    taylor = TaylorDiagram(std_truth, fig=fig, rect=111)
    
    # Add each model
    for model_name, preds in all_predictions.items():
        pred_flat = preds[variable].flatten()
        pred_valid = pred_flat[valid]
        
        # Compute stats
        std_pred = np.std(pred_valid)
        correlation = np.corrcoef(truth_valid, pred_valid)[0, 1]
        
        # Get style
        color, _ = get_model_style(model_name)
        
        # Determine marker based on architecture
        if 'MLP' in model_name and 'ResMLP' not in model_name:
            marker = 'o'
        elif 'ResMLP' in model_name:
            marker = 's'
        else:
            marker = '^'
        
        short_name = model_name.replace('Baseline-', 'B-').replace('Ri-', 'Ri-')
        
        taylor.add_sample(std_pred, correlation, 
                         label=short_name, color=color, marker=marker)
    
    taylor.add_legend(loc='upper right', fontsize=9, framealpha=0.9, ncol=2)
    taylor.set_title(f'Taylor Diagram - {var_label}', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    output_file = output_dir / f'taylor_diagram_{variable}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"✓ Saved: {output_file.name}")


# ==================== MODEL RANKING ====================

def rank_models_with_uncertainty(all_metrics: Dict[str, Dict],
                                 all_predictions: Dict[str, Dict],
                                 truth: Dict,
                                 variable: str = 'visc_coeff',
                                 criterion: str = 'r2',
                                 n_bootstrap: int = 1000) -> pd.DataFrame:
    """
    Rank models with bootstrap confidence intervals.
    
    Returns DataFrame with:
    - Model name
    - Mean metric
    - CI lower/upper
    - Rank
    """
    from sklearn.metrics import r2_score, mean_squared_error
    
    results = []
    
    truth_flat = truth[variable].flatten()
    
    for model_name in all_metrics.keys():
        pred_flat = all_predictions[model_name][variable].flatten()
        
        # Define metric function
        if criterion == 'r2':
            metric_func = lambda p, t: r2_score(t, p)
        elif criterion == 'rmse':
            metric_func = lambda p, t: np.sqrt(mean_squared_error(t, p))
        else:
            metric_func = lambda p, t: np.mean(np.abs(p - t))
        
        # Bootstrap CI
        ci_result = bootstrap_confidence_interval(
            pred_flat, truth_flat, metric_func, 
            n_bootstrap=n_bootstrap
        )
        
        results.append({
            'Model': model_name,
            'Mean': ci_result['mean'],
            'CI_Lower': ci_result['ci_lower'],
            'CI_Upper': ci_result['ci_upper'],
            'Std': ci_result['std']
        })
    
    df = pd.DataFrame(results)
    
    # Rank (higher is better for R², lower is better for RMSE/MAE)
    if criterion == 'r2':
        df = df.sort_values('Mean', ascending=False)
    else:
        df = df.sort_values('Mean', ascending=True)
    
    df['Rank'] = range(1, len(df) + 1)
    
    return df


def plot_ranked_models_with_ci(ranking_df: pd.DataFrame,
                               output_dir: Path,
                               variable: str = 'visc_coeff',
                               criterion: str = 'r2'):
    """
    Plot ranked models with confidence intervals.
    """
    from plotting_utils import get_model_style
    
    output_dir = Path(output_dir)
    
    fig, ax = plt.subplots(figsize=(12, max(6, len(ranking_df) * 0.4)))
    
    y_pos = np.arange(len(ranking_df))
    
    colors = [get_model_style(name)[0] for name in ranking_df['Model']]
    
    # Plot means
    ax.barh(y_pos, ranking_df['Mean'], color=colors, alpha=0.7, edgecolor='black')
    
    # Plot error bars (CI)
    errors_lower = ranking_df['Mean'] - ranking_df['CI_Lower']
    errors_upper = ranking_df['CI_Upper'] - ranking_df['Mean']
    
    ax.errorbar(ranking_df['Mean'], y_pos, 
               xerr=[errors_lower, errors_upper],
               fmt='none', ecolor='black', capsize=5, capthick=2, alpha=0.8)
    
    # Reference line
    if criterion == 'r2':
        ax.axvline(x=0.7, color='green', linestyle='--', alpha=0.5, label='Target (0.70)')
        ax.axvline(x=0.85, color='gold', linestyle='--', alpha=0.5, label='Excellent (0.85)')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{rank}. {name.replace('Baseline-', 'B-').replace('Ri-', 'Ri-')}" 
                        for rank, name in zip(ranking_df['Rank'], ranking_df['Model'])],
                       fontsize=9)
    
    ax.set_xlabel(criterion.upper() + ' (95% CI)', fontsize=12, fontweight='bold')
    ax.set_title(f'Model Ranking - {variable.replace("_coeff", "").upper()} - {criterion.upper()}',
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend(fontsize=9)
    
    plt.tight_layout()
    
    output_file = output_dir / f'ranked_models_{variable}_{criterion}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"✓ Saved: {output_file.name}")


# ==================== COMPREHENSIVE STATISTICAL REPORT ====================

def generate_statistical_report(all_predictions: Dict[str, Dict],
                               truth: Dict,
                               output_dir: Path,
                               variable: str = 'visc_coeff'):
    """
    Generate comprehensive statistical analysis report.
    
    Includes:
    - Pairwise significance tests
    - Skill scores for all models
    - Model ranking with CI
    - Taylor diagram
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"STATISTICAL ANALYSIS: {variable}")
    logger.info(f"{'='*80}\n")
    
    # 1. Skill scores
    logger.info("Computing skill scores...")
    
    skill_scores = {}
    for model_name, preds in all_predictions.items():
        pred_flat = preds[variable].flatten()
        truth_flat = truth[variable].flatten()
        
        scores = compute_all_skill_scores(pred_flat, truth_flat)
        skill_scores[model_name] = scores
    
    # Save to CSV
    skill_df = pd.DataFrame(skill_scores).T
    skill_df.index.name = 'Model'
    skill_df.to_csv(output_dir / f'skill_scores_{variable}.csv', float_format='%.6f')
    logger.info(f"✓ Saved: skill_scores_{variable}.csv")
    
    # 2. Pairwise significance
    logger.info("Computing pairwise significance tests...")
    
    p_matrix_ttest = pairwise_significance_matrix(
        all_predictions, truth, variable, test='ttest'
    )
    p_matrix_ttest.to_csv(output_dir / f'significance_ttest_{variable}.csv', 
                         float_format='%.4f')
    logger.info(f"✓ Saved: significance_ttest_{variable}.csv")
    
    p_matrix_wilcoxon = pairwise_significance_matrix(
        all_predictions, truth, variable, test='wilcoxon'
    )
    p_matrix_wilcoxon.to_csv(output_dir / f'significance_wilcoxon_{variable}.csv',
                            float_format='%.4f')
    logger.info(f"✓ Saved: significance_wilcoxon_{variable}.csv")
    
    # 3. Model ranking with CI
    logger.info("Ranking models with bootstrap CI...")
    
    ranking_r2 = rank_models_with_uncertainty(
        {}, all_predictions, truth, variable, 'r2', n_bootstrap=500
    )
    ranking_r2.to_csv(output_dir / f'ranking_r2_{variable}.csv', 
                     index=False, float_format='%.6f')
    logger.info(f"✓ Saved: ranking_r2_{variable}.csv")
    
    plot_ranked_models_with_ci(ranking_r2, output_dir, variable, 'r2')
    
    # 4. Taylor diagram
    logger.info("Creating Taylor diagram...")
    plot_taylor_diagram(all_predictions, truth, output_dir, variable)
    
    logger.info(f"\n{'='*80}\n")
    
    return {
        'skill_scores': skill_scores,
        'p_matrix_ttest': p_matrix_ttest,
        'p_matrix_wilcoxon': p_matrix_wilcoxon,
        'ranking': ranking_r2
    }


# ==================== TESTING ====================

if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)
    
    truth = np.random.randn(10000)
    pred1 = truth + np.random.randn(10000) * 0.1  # Better model
    pred2 = truth + np.random.randn(10000) * 0.3  # Worse model
    
    print("\n" + "="*60)
    print("TESTING STATISTICAL FUNCTIONS")
    print("="*60)
    
    # t-test
    result = paired_ttest(pred1, pred2, truth)
    print("\nPaired t-test:")
    print(f"  p-value: {result['p_value']:.6f}")
    print(f"  Significant: {result['significant']}")
    
    # Skill scores
    scores1 = compute_all_skill_scores(pred1, truth)
    print("\nSkill scores (Model 1):")
    for key, val in scores1.items():
        print(f"  {key}: {val:.4f}")
    
    print("\n" + "="*60)
