#!/usr/bin/env python3
"""
Master Model Comparison Script (Enhanced)
==========================================

Comprehensive comparison of all ML-SGS models with advanced statistics:
- Baseline models
- Richardson-conditioned models
- Q1-Q4 configurations

Features:
- Unified inference across all models
- Save predictions for future analysis
- Generate comprehensive metrics and plots
- Statistical significance tests
- Taylor diagrams
- Skill scores
- Model ranking with confidence intervals
- Publication-ready outputs

Usage:
    # Full analysis with statistics
    python master_comparison.py \
        --data-dir /path/to/netcdf_files/ \
        --output results/ \
        --baseline-mlp baseline_mlp.pt \
        --ri-mlp ri_mlp.pt \
        --q1-mlp q1_mlp.pt \
        --q2-mlp q2_mlp.pt \
        --scaler-dir scalers/ \
        --config config.yaml \
        --save-predictions \
        --enable-statistics \
        --enable-taylor-diagram
"""

import numpy as np
from pathlib import Path
import argparse
import logging
from typing import Dict, List
import yaml
import re
from tqdm import tqdm
import json

# Import unified framework
from unified_inference_engine import UnifiedInferenceEngine
from analysis_utils import (
    calculate_domain_metrics,
    calculate_nonzero_metrics,
    save_metrics_csv,
    create_comparison_table,
    identify_best_models,
    print_metrics_summary,
    NumpyEncoder
)
from plotting_utils import (
    plot_metrics_comparison_bar,
    plot_scatter_comparison_grid,
    plot_vertical_profiles_comparison,
    plot_distribution_comparison,
    create_summary_figure
)
from prediction_storage import PredictionStorage
from run_best_models_analysis import extract_truth_from_netcdf

# Import advanced metrics
from advanced_metrics import (
    compute_all_skill_scores,
    pairwise_significance_matrix,
    rank_models_with_uncertainty,
    plot_ranked_models_with_ci,
    plot_taylor_diagram,
    generate_statistical_report,
    paired_ttest,
    bootstrap_confidence_interval
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def reshape_for_profiles(flat_predictions: Dict[str, np.ndarray],
                        flat_truth: Dict[str, np.ndarray],
                        shape_3d: tuple) -> tuple:
    """
    Reshape flattened data back to 3D for vertical profiles.
    
    Parameters
    ----------
    flat_predictions : Dict[str, np.ndarray]
        Flattened predictions
    flat_truth : Dict[str, np.ndarray]
        Flattened truth
    shape_3d : tuple
        Target 3D shape (nx, ny, nz)
    
    Returns
    -------
    predictions_3d : Dict[str, np.ndarray]
        Reshaped predictions
    truth_3d : Dict[str, np.ndarray]
        Reshaped truth
    """
    nx, ny, nz = shape_3d
    total_points = nx * ny * nz
    
    predictions_3d = {}
    truth_3d = {}
    
    for var in ['visc_coeff', 'diff_coeff', 'richardson']:
        if var in flat_truth and len(flat_truth[var]) > 0:
            # Take only the amount needed for one file
            truth_3d[var] = flat_truth[var][:total_points].reshape(nx, ny, nz)
        
        if var in flat_predictions:
            predictions_3d[var] = flat_predictions[var][:total_points].reshape(nx, ny, nz)
    
    return predictions_3d, truth_3d


def main():
    parser = argparse.ArgumentParser(
        description='Master Model Comparison Script with Advanced Statistics',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Data paths
    parser.add_argument('--mode', type=str, default='timeseries',
                       choices=['single', 'timeseries'])
    parser.add_argument('--nc-file', type=Path, help='Single NetCDF file')
    parser.add_argument('--data-dir', type=Path, help='Directory with NetCDF files')
    parser.add_argument('--output', type=Path, required=True)
    
    # Baseline models
    parser.add_argument('--baseline-mlp', type=Path)
    parser.add_argument('--baseline-resmlp', type=Path)
    parser.add_argument('--baseline-tabtransformer', type=Path)
    
    # Ri-conditioned models
    parser.add_argument('--ri-mlp', type=Path)
    parser.add_argument('--ri-resmlp', type=Path)
    parser.add_argument('--ri-tabtransformer', type=Path)
    
    # Q1 models
    parser.add_argument('--q1-mlp', type=Path)
    parser.add_argument('--q1-resmlp', type=Path)
    parser.add_argument('--q1-tabtransformer', type=Path)
    
    # Q2 models
    parser.add_argument('--q2-mlp', type=Path)
    parser.add_argument('--q2-resmlp', type=Path)
    parser.add_argument('--q2-tabtransformer', type=Path)
    
    # Q3 models
    parser.add_argument('--q3-mlp', type=Path)
    parser.add_argument('--q3-resmlp', type=Path)
    parser.add_argument('--q3-tabtransformer', type=Path)
    
    # Q4 models
    parser.add_argument('--q4-mlp', type=Path)
    parser.add_argument('--q4-resmlp', type=Path)
    parser.add_argument('--q4-tabtransformer', type=Path)
    
    # Configuration
    parser.add_argument('--scaler-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, help='Config YAML for Q1-Q4')
    parser.add_argument('--time-idx', type=int, default=0)
    parser.add_argument('--k-min', type=int, default=0)
    parser.add_argument('--k-max', type=int, default=98)
    parser.add_argument('--n-workers', type=int, default=None)
    
    # Analysis options
    parser.add_argument('--save-predictions', action='store_true',
                       help='Save predictions to HDF5 for future analysis')
    parser.add_argument('--load-predictions', action='store_true',
                       help='Load existing predictions instead of running inference')
    parser.add_argument('--prediction-dir', type=Path, default=None,
                       help='Directory for prediction storage')
    parser.add_argument('--sample-rate', type=float, default=1.0,
                       help='Fraction of files to process (for testing)')
    
    # Advanced statistics options
    parser.add_argument('--enable-statistics', action='store_true',
                       help='Enable statistical significance tests')
    parser.add_argument('--enable-taylor-diagram', action='store_true',
                       help='Generate Taylor diagrams')
    parser.add_argument('--enable-skill-scores', action='store_true',
                       help='Compute skill scores (NSE, KGE, etc.)')
    parser.add_argument('--enable-bootstrap', action='store_true',
                       help='Compute bootstrap confidence intervals')
    parser.add_argument('--n-bootstrap', type=int, default=1000,
                       help='Number of bootstrap samples')
    
    # Plotting options
    parser.add_argument('--skip-plots', action='store_true',
                       help='Skip plot generation (metrics only)')
    parser.add_argument('--plot-vertical-profiles', action='store_true',
                       help='Generate vertical profile plots (requires 3D data)')
    
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.prediction_dir is None:
        args.prediction_dir = output_dir / 'predictions'
    
    logger.info("\n" + "="*80)
    logger.info("MASTER MODEL COMPARISON (ENHANCED)")
    logger.info("="*80 + "\n")
    
    # Build model paths
    baseline_paths = {}
    if args.baseline_mlp: baseline_paths['MLP'] = args.baseline_mlp
    if args.baseline_resmlp: baseline_paths['ResMLP'] = args.baseline_resmlp
    if args.baseline_tabtransformer: baseline_paths['TabTransformer'] = args.baseline_tabtransformer
    
    ri_paths = {}
    if args.ri_mlp: ri_paths['MLP'] = args.ri_mlp
    if args.ri_resmlp: ri_paths['ResMLP'] = args.ri_resmlp
    if args.ri_tabtransformer: ri_paths['TabTransformer'] = args.ri_tabtransformer
    
    q1q4_paths = {}
    
    # Q1
    q1_paths = {}
    if args.q1_mlp: q1_paths['MLP'] = args.q1_mlp
    if args.q1_resmlp: q1_paths['ResMLP'] = args.q1_resmlp
    if args.q1_tabtransformer: q1_paths['TabTransformer'] = args.q1_tabtransformer
    if q1_paths: q1q4_paths['Q1'] = q1_paths
    
    # Q2
    q2_paths = {}
    if args.q2_mlp: q2_paths['MLP'] = args.q2_mlp
    if args.q2_resmlp: q2_paths['ResMLP'] = args.q2_resmlp
    if args.q2_tabtransformer: q2_paths['TabTransformer'] = args.q2_tabtransformer
    if q2_paths: q1q4_paths['Q2'] = q2_paths
    
    # Q3
    q3_paths = {}
    if args.q3_mlp: q3_paths['MLP'] = args.q3_mlp
    if args.q3_resmlp: q3_paths['ResMLP'] = args.q3_resmlp
    if args.q3_tabtransformer: q3_paths['TabTransformer'] = args.q3_tabtransformer
    if q3_paths: q1q4_paths['Q3'] = q3_paths
    
    # Q4
    q4_paths = {}
    if args.q4_mlp: q4_paths['MLP'] = args.q4_mlp
    if args.q4_resmlp: q4_paths['ResMLP'] = args.q4_resmlp
    if args.q4_tabtransformer: q4_paths['TabTransformer'] = args.q4_tabtransformer
    if q4_paths: q1q4_paths['Q4'] = q4_paths
    
    if not baseline_paths and not ri_paths and not q1q4_paths:
        parser.error("Must provide at least one model")
    
    # Initialize prediction storage
    storage = PredictionStorage(args.prediction_dir)
    
    # Initialize unified engine (unless loading predictions)
    if not args.load_predictions:
        logger.info("Initializing unified inference engine...")
        
        engine = UnifiedInferenceEngine()
        
        if baseline_paths:
            engine.add_baseline_models(baseline_paths, args.scaler_dir, args.n_workers)
        
        if ri_paths:
            engine.add_ri_models(ri_paths, args.scaler_dir, args.n_workers)
        
        if q1q4_paths:
            if not args.config:
                parser.error("--config required for Q1-Q4 models")
            with open(args.config) as f:
                config = yaml.safe_load(f)
            engine.add_q1q4_models(q1q4_paths, config, args.scaler_dir, args.n_workers)
        
        engine.summary()
        model_names = engine.get_all_model_names()
    else:
        logger.info("Loading predictions from storage...")
        storage.summary()
        model_names = storage.get_model_list()
        engine = None
    
    # Get file list
    if args.mode == 'single':
        nc_files = [args.nc_file]
    else:
        nc_files = sorted(list(args.data_dir.glob('*.nc')))
        nc_files.sort(key=lambda f: int(re.search(r'(\d+)', f.name).group()) 
                     if re.search(r'(\d+)', f.name) else f.name)
        
        if args.sample_rate < 1.0:
            n_sample = int(len(nc_files) * args.sample_rate)
            nc_files = nc_files[:n_sample]
            logger.info(f"Sampling {n_sample} of {len(nc_files)} files")
    
    logger.info(f"Processing {len(nc_files)} files")
    
    # Storage for aggregated data
    agg_predictions = {name: {'visc_coeff': [], 'diff_coeff': [], 'richardson': []} 
                      for name in model_names}
    agg_truth = {'visc_coeff': [], 'diff_coeff': [], 'richardson': []}
    
    # Store one file's 3D data for profiles
    last_3d_predictions = None
    last_3d_truth = None
    shape_3d = None
    
    # Process files
    logger.info("\n" + "="*80)
    logger.info("RUNNING INFERENCE" if not args.load_predictions else "LOADING PREDICTIONS")
    logger.info("="*80 + "\n")
    
    for file_idx, nc_file in enumerate(tqdm(nc_files, desc="Processing files")):
        try:
            # Load or compute predictions
            if args.load_predictions:
                # Load from storage
                file_predictions = {}
                for model_name in model_names:
                    try:
                        file_predictions[model_name] = storage.load_predictions(
                            model_name, nc_file
                        )
                    except KeyError:
                        logger.warning(f"No predictions for {model_name} - {nc_file.name}")
            else:
                # Run inference
                file_predictions = engine.predict_all(
                    nc_file, args.time_idx, args.k_min, args.k_max
                )
                
                # Save if requested
                if args.save_predictions:
                    for model_name, preds in file_predictions.items():
                        model_info = engine.get_model_info(model_name)
                        storage.save_predictions(
                            model_name, nc_file, preds, 
                            model_info=model_info, overwrite=True
                        )
            
            # Load truth
            truth = extract_truth_from_netcdf(
                nc_file, args.time_idx, args.k_min, args.k_max
            )
            
            # Store last file's 3D data for profiles
            if file_idx == len(nc_files) - 1:
                last_3d_predictions = file_predictions
                last_3d_truth = truth
                shape_3d = truth['visc_coeff'].shape
            
            # Aggregate
            for var in ['visc_coeff', 'diff_coeff', 'richardson']:
                if var in truth:
                    agg_truth[var].append(truth[var].flatten())
                
                for model_name in model_names:
                    if model_name in file_predictions and var in file_predictions[model_name]:
                        agg_predictions[model_name][var].append(
                            file_predictions[model_name][var].flatten()
                        )
        
        except Exception as e:
            logger.error(f"Error processing {nc_file.name}: {e}")
            continue
    
    # Concatenate
    logger.info("\nAggregating results...")
    
    final_truth = {
        var: np.concatenate(arrays) for var, arrays in agg_truth.items() if arrays
    }
    
    final_predictions = {}
    for model_name in model_names:
        final_predictions[model_name] = {
            var: np.concatenate(arrays) 
            for var, arrays in agg_predictions[model_name].items() 
            if arrays
        }
    
    logger.info(f"Total points: {len(final_truth['visc_coeff']):,}")
    
    # Calculate metrics
    logger.info("\n" + "="*80)
    logger.info("CALCULATING METRICS")
    logger.info("="*80 + "\n")
    
    all_metrics = {}
    all_nonzero_metrics = {}
    
    for model_name in model_names:
        logger.info(f"Processing {model_name}...")
        
        # Domain metrics
        metrics = calculate_domain_metrics(
            final_predictions[model_name], 
            final_truth
        )
        all_metrics[model_name] = metrics
        
        # Non-zero metrics
        nonzero_metrics = calculate_nonzero_metrics(
            final_predictions[model_name],
            final_truth
        )
        all_nonzero_metrics[model_name] = nonzero_metrics
    
    # Print summary
    print_metrics_summary(all_metrics)
    
    # Identify best models
    logger.info("\n" + "="*80)
    logger.info("BEST MODELS")
    logger.info("="*80 + "\n")
    
    best_r2 = identify_best_models(all_metrics, criterion='r2')
    best_rmse = identify_best_models(all_metrics, criterion='rmse')
    
    # Advanced statistics
    statistical_results = {}
    
    if args.enable_statistics or args.enable_skill_scores or args.enable_taylor_diagram or args.enable_bootstrap:
        logger.info("\n" + "="*80)
        logger.info("ADVANCED STATISTICAL ANALYSIS")
        logger.info("="*80 + "\n")
        
        stats_dir = output_dir / 'statistics'
        stats_dir.mkdir(exist_ok=True)
        
        for var in ['visc_coeff', 'diff_coeff']:
            logger.info(f"\nAnalyzing {var}...")
            
            # Skill scores
            if args.enable_skill_scores:
                logger.info("  Computing skill scores...")
                skill_scores = {}
                for model_name in model_names:
                    scores = compute_all_skill_scores(
                        final_predictions[model_name][var],
                        final_truth[var]
                    )
                    skill_scores[model_name] = scores
                
                import pandas as pd
                skill_df = pd.DataFrame(skill_scores).T
                skill_df.index.name = 'Model'
                skill_df.to_csv(stats_dir / f'skill_scores_{var}.csv', float_format='%.6f')
                logger.info(f"    ✓ Saved: skill_scores_{var}.csv")
                
                statistical_results[f'{var}_skill_scores'] = skill_scores
            
            # Statistical significance
            if args.enable_statistics:
                logger.info("  Computing pairwise significance tests...")
                
                p_matrix_ttest = pairwise_significance_matrix(
                    final_predictions, final_truth, var, test='ttest'
                )
                p_matrix_ttest.to_csv(stats_dir / f'significance_ttest_{var}.csv',
                                     float_format='%.4f')
                logger.info(f"    ✓ Saved: significance_ttest_{var}.csv")
                
                p_matrix_wilcoxon = pairwise_significance_matrix(
                    final_predictions, final_truth, var, test='wilcoxon'
                )
                p_matrix_wilcoxon.to_csv(stats_dir / f'significance_wilcoxon_{var}.csv',
                                        float_format='%.4f')
                logger.info(f"    ✓ Saved: significance_wilcoxon_{var}.csv")
                
                statistical_results[f'{var}_significance_ttest'] = p_matrix_ttest
                statistical_results[f'{var}_significance_wilcoxon'] = p_matrix_wilcoxon
            
            # Bootstrap CI and ranking
            if args.enable_bootstrap:
                logger.info(f"  Computing bootstrap confidence intervals (n={args.n_bootstrap})...")
                
                ranking_r2 = rank_models_with_uncertainty(
                    all_metrics, final_predictions, final_truth,
                    var, 'r2', n_bootstrap=args.n_bootstrap
                )
                ranking_r2.to_csv(stats_dir / f'ranking_r2_{var}.csv',
                                 index=False, float_format='%.6f')
                logger.info(f"    ✓ Saved: ranking_r2_{var}.csv")
                
                ranking_rmse = rank_models_with_uncertainty(
                    all_metrics, final_predictions, final_truth,
                    var, 'rmse', n_bootstrap=args.n_bootstrap
                )
                ranking_rmse.to_csv(stats_dir / f'ranking_rmse_{var}.csv',
                                   index=False, float_format='%.6f')
                logger.info(f"    ✓ Saved: ranking_rmse_{var}.csv")
                
                # Plot rankings
                if not args.skip_plots:
                    plot_ranked_models_with_ci(ranking_r2, stats_dir, var, 'r2')
                    plot_ranked_models_with_ci(ranking_rmse, stats_dir, var, 'rmse')
                
                statistical_results[f'{var}_ranking_r2'] = ranking_r2
                statistical_results[f'{var}_ranking_rmse'] = ranking_rmse
            
            # Taylor diagram
            if args.enable_taylor_diagram:
                logger.info("  Creating Taylor diagram...")
                
                # Need 3D data for Taylor diagram
                taylor_preds = {}
                for model_name in model_names:
                    taylor_preds[model_name] = {var: final_predictions[model_name][var]}
                
                taylor_truth = {var: final_truth[var]}
                
                plot_taylor_diagram(taylor_preds, taylor_truth, stats_dir, var)
    
    # Save metrics
    logger.info("\n" + "="*80)
    logger.info("SAVING RESULTS")
    logger.info("="*80 + "\n")
    
    # CSV files
    save_metrics_csv(all_metrics, output_dir / 'metrics_domain.csv', 'domain')
    save_metrics_csv(all_nonzero_metrics, output_dir / 'metrics_nonzero.csv', 'nonzero')
    
    # Comparison table
    comparison_df = create_comparison_table(all_metrics)
    comparison_df.to_csv(output_dir / 'comparison_table.csv', index=False, float_format='%.6f')
    logger.info("✓ Saved: comparison_table.csv")
    
    # JSON files
    with open(output_dir / 'metrics_all.json', 'w') as f:
        json.dump({
            'domain': all_metrics,
            'nonzero': all_nonzero_metrics,
            'best_models': {
                'r2': best_r2,
                'rmse': best_rmse
            },
            'n_files': len(nc_files),
            'n_points': len(final_truth['visc_coeff']),
            'models': model_names
        }, f, indent=2, cls=NumpyEncoder)
    logger.info("✓ Saved: metrics_all.json")
    
    # Save statistical results if computed
    if statistical_results:
        # Convert pandas DataFrames to dict for JSON
        stats_json = {}
        for key, value in statistical_results.items():
            if hasattr(value, 'to_dict'):
                stats_json[key] = value.to_dict()
            else:
                stats_json[key] = value
        
        with open(output_dir / 'statistics' / 'statistical_analysis.json', 'w') as f:
            json.dump(stats_json, f, indent=2, cls=NumpyEncoder)
        logger.info("✓ Saved: statistical_analysis.json")
    
    # Generate plots
    if not args.skip_plots:
        logger.info("\n" + "="*80)
        logger.info("GENERATING PLOTS")
        logger.info("="*80 + "\n")
        
        plot_dir = output_dir / 'plots'
        plot_dir.mkdir(exist_ok=True)
        
        # Metrics comparison bars
        for metric in ['r2', 'rmse', 'mae', 'correlation']:
            plot_metrics_comparison_bar(all_metrics, plot_dir, metric=metric)
        
        # Scatter plots
        for var in ['visc_coeff', 'diff_coeff']:
            plot_scatter_comparison_grid(
                final_predictions, final_truth, plot_dir, 
                variable=var, max_models_per_plot=9
            )
        
        # Vertical profiles (if requested and 3D data available)
        if args.plot_vertical_profiles and last_3d_predictions is not None:
            logger.info("\nGenerating vertical profiles...")
            for var in ['visc_coeff', 'diff_coeff']:
                plot_vertical_profiles_comparison(
                    last_3d_predictions, last_3d_truth, plot_dir,
                    variable=var, group_by='architecture'
                )
                plot_vertical_profiles_comparison(
                    last_3d_predictions, last_3d_truth, plot_dir,
                    variable=var, group_by='configuration'
                )
        
        # Distributions
        for var in ['visc_coeff', 'diff_coeff']:
            plot_distribution_comparison(
                final_predictions, final_truth, plot_dir,
                variable=var, log_scale=True, max_models_per_plot=6
            )
            plot_distribution_comparison(
                final_predictions, final_truth, plot_dir,
                variable=var, log_scale=False, max_models_per_plot=6
            )
        
        # Summary figure
        create_summary_figure(all_metrics, plot_dir)
    
    # Save storage info if predictions were saved
    if args.save_predictions:
        storage.summary()
        with open(output_dir / 'prediction_storage_info.json', 'w') as f:
            json.dump(storage.get_storage_info(), f, indent=2)
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("✅ ANALYSIS COMPLETE")
    logger.info("="*80)
    logger.info(f"\nResults saved to: {output_dir}")
    logger.info(f"\n📊 Standard Outputs:")
    logger.info(f"  - Metrics: metrics_domain.csv, metrics_nonzero.csv")
    logger.info(f"  - Comparison: comparison_table.csv")
    logger.info(f"  - Complete metrics: metrics_all.json")
    
    if not args.skip_plots:
        logger.info(f"  - Plots: {plot_dir}/")
    
    if args.save_predictions:
        logger.info(f"  - Predictions: {args.prediction_dir}/")
    
    if statistical_results:
        logger.info(f"\n📈 Advanced Statistics:")
        if args.enable_skill_scores:
            logger.info(f"  - Skill scores (NSE, KGE, etc.)")
        if args.enable_statistics:
            logger.info(f"  - Pairwise significance tests (t-test, Wilcoxon)")
        if args.enable_bootstrap:
            logger.info(f"  - Bootstrap confidence intervals")
            logger.info(f"  - Model rankings with uncertainty")
        if args.enable_taylor_diagram:
            logger.info(f"  - Taylor diagrams")
        logger.info(f"  - All saved in: {output_dir}/statistics/")
    
    logger.info(f"\n📈 Summary:")
    logger.info(f"  - Files processed: {len(nc_files)}")
    logger.info(f"  - Data points: {len(final_truth['visc_coeff']):,}")
    logger.info(f"  - Models compared: {len(model_names)}")
    
    # Best models summary
    logger.info(f"\n🏆 Best Models (R²):")
    for var, model in best_r2.items():
        r2_val = all_metrics[model][var]['r2']
        logger.info(f"  - {var}: {model} (R² = {r2_val:.4f})")
    
    logger.info("\n" + "="*80 + "\n")


if __name__ == '__main__':
    main()
