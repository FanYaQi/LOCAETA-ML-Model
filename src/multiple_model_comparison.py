import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from utils.path_util import DATA_PATH, MODELS_PATH
from trainModel import ConfigurableMLTrainer
import warnings
warnings.filterwarnings('ignore')

class ExperimentRunner:
    """
    Systematic experiment runner to test model robustness across validation strategies
    """
    
    def __init__(self, base_config, data_paths):
        """
        Initialize experiment runner
        
        Args:
            base_config: Dictionary with common configuration settings
            data_paths: Dictionary with 'X_features' and 'y_targets_template' paths
        """
        self.base_config = base_config
        self.data_paths = data_paths
        self.results = []
        self.experiment_configs = []
        
        # Create main results directory
        self.results_dir = Path(base_config['output_dir'])
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def experiment_1_model_comparison(self):
        """
        Experiment 1: Compare all models with fixed settings
        Focus: Which models are most robust across validation strategies?
        """
        print(f"\n{'='*80}")
        print("EXPERIMENT 1: MODEL COMPARISON")
        print(f"{'='*80}")
        print("Testing model robustness across sample-based vs site-based validation")
        
        models = ['RF', 'LGBM', 'XGB', 'MLR', 'SVR', 'GBM']
        validation_types = ['sample_based', 'site_based']
        
        exp_results = []
        
        for model_type in models:
            for validation_type in validation_types:
                config = self.base_config.copy()
                config.update({
                    'model_type': model_type,
                    'validation_type': validation_type,
                    'method': 'method1',  # Fixed
                    'target_log_transform': True,  # Fixed based on your experience
                    'figure_suffix': f'exp1_models_{model_type.lower()}',
                    'output_dir': str(self.results_dir / 'exp1_model_comparison'),
                    'generate_heatmaps': validation_type == 'site_based'  # Generate heatmaps for site validation
                })
                
                result = self._run_single_experiment(config, f"Exp1: {model_type} - {validation_type}")
                if result:
                    result['experiment'] = 'model_comparison'
                    result['variable_name'] = 'model_type'
                    result['variable_value'] = model_type
                    exp_results.append(result)
        
        # Analyze robustness
        self._analyze_robustness(exp_results, 'model_type', 'Model Type')
        return exp_results
    
    def experiment_2_log_transform(self):
        """
        Experiment 2: Compare log transform impact with RF
        Focus: How much does log transform help, and is it consistent across validation strategies?
        """
        print(f"\n{'='*80}")
        print("EXPERIMENT 2: LOG TRANSFORM COMPARISON")
        print(f"{'='*80}")
        print("Testing log transform impact with RF model")
        
        log_options = [True, False]
        validation_types = ['sample_based', 'site_based']
        
        exp_results = []
        
        for log_transform in log_options:
            for validation_type in validation_types:
                config = self.base_config.copy()
                config.update({
                    'model_type': 'RF',  # Fixed based on your experience
                    'validation_type': validation_type,
                    'method': 'method1',  # Fixed
                    'target_log_transform': log_transform,
                    'figure_suffix': f'exp2_log_{str(log_transform).lower()}',
                    'output_dir': str(self.results_dir / 'exp2_log_transform'),
                    'generate_heatmaps': validation_type == 'site_based'  # Generate heatmaps for site validation
                })
                
                result = self._run_single_experiment(config, f"Exp2: Log={log_transform} - {validation_type}")
                if result:
                    result['experiment'] = 'log_transform'
                    result['variable_name'] = 'target_log_transform'
                    result['variable_value'] = log_transform
                    exp_results.append(result)
        
        # Analyze robustness
        self._analyze_robustness(exp_results, 'target_log_transform', 'Log Transform')
        return exp_results
    
    def experiment_3_method_comparison(self):
        """
        Experiment 3: Compare methods with optimal RF+log settings
        Focus: Which method is most robust and gives best performance?
        """
        print(f"\n{'='*80}")
        print("EXPERIMENT 3: METHOD COMPARISON")
        print(f"{'='*80}")
        print("Testing method robustness with RF + log transform")
        
        methods = ['method1', 'method2', 'method3']
        validation_types = ['sample_based', 'site_based']
        
        exp_results = []
        
        for method in methods:
            for validation_type in validation_types:
                config = self.base_config.copy()
                config.update({
                    'model_type': 'RF',  # Fixed - best from exp 1
                    'validation_type': validation_type,
                    'method': method,
                    'target_log_transform': True,  # Fixed - best from exp 2
                    'figure_suffix': f'exp3_methods_{method}',
                    'output_dir': str(self.results_dir / 'exp3_method_comparison'),
                    'generate_heatmaps': validation_type == 'site_based'  # Generate for site-based only
                })
                
                result = self._run_single_experiment(config, f"Exp3: {method} - {validation_type}")
                if result:
                    result['experiment'] = 'method_comparison'
                    result['variable_name'] = 'method'
                    result['variable_value'] = method
                    exp_results.append(result)
        
        # Analyze robustness
        self._analyze_robustness(exp_results, 'method', 'Method')
        return exp_results
    
    def _run_single_experiment(self, config, description):
        """Run a single experiment configuration"""
        print(f"\nRunning: {description}")
        
        try:
            # Construct target file path
            method = config['method']
            y_targets_path = self.data_paths['y_targets_template'].format(method=method)
            
            if not Path(y_targets_path).exists():
                print(f"  ✗ Target file not found: {y_targets_path}")
                return None
            
            # Initialize trainer
            trainer = ConfigurableMLTrainer(config)
            
            # Load and prepare data
            X, y = trainer.load_and_prepare_data(self.data_paths['X_features'], y_targets_path)
            
            # Split data
            trainer.split_data(X, y)
            
            # Train model
            metrics = trainer.train_model()
            
            # Create plots
            plot_paths = trainer.create_plots()
            
            # Save results
            pred_path, config_path = trainer.save_results()
            
            # Store result
            result = {
                'model_type': config['model_type'],
                'validation_type': config['validation_type'],
                'method': config['method'],
                'target_log_transform': config['target_log_transform'],
                'r2_score': metrics['r2'],
                'rmse': metrics['rmse'],
                'mae': metrics['mae'],
                'n_train': metrics['n_train'],
                'n_test': metrics['n_test'],
                'plot_paths': [str(p) for p in plot_paths],
                'pred_path': str(pred_path),
                'config_path': str(config_path),
                'status': 'success'
            }
            
            self.results.append(result)
            print(f"  ✓ Success: R² = {metrics['r2']:.3f}, RMSE = {metrics['rmse']:.3f}")
            
            return result
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            
            # Store failed result
            failed_result = {
                'model_type': config.get('model_type', 'unknown'),
                'validation_type': config.get('validation_type', 'unknown'),
                'method': config.get('method', 'unknown'),
                'target_log_transform': config.get('target_log_transform', False),
                'r2_score': np.nan,
                'rmse': np.nan,
                'mae': np.nan,
                'n_train': np.nan,
                'n_test': np.nan,
                'plot_paths': [],
                'pred_path': 'FAILED',
                'config_path': 'FAILED',
                'status': 'failed',
                'error': str(e)
            }
            
            self.results.append(failed_result)
            return None
    
    def _analyze_robustness(self, exp_results, variable_col, variable_name):
        """Analyze robustness across validation strategies for a specific experiment"""
        if not exp_results:
            print(f"No results to analyze for {variable_name}")
            return
        
        df = pd.DataFrame(exp_results)
        successful_df = df[df['status'] == 'success'].copy()
        
        if len(successful_df) == 0:
            print(f"No successful experiments for {variable_name}")
            return
        
        print(f"\n{'-'*60}")
        print(f"{variable_name.upper()} ROBUSTNESS ANALYSIS")
        print(f"{'-'*60}")
        
        # Calculate robustness metrics
        robustness_metrics = []
        
        for variable_value in successful_df[variable_col].unique():
            subset = successful_df[successful_df[variable_col] == variable_value]
            
            if len(subset) >= 2:  # Need both validation types
                sample_based = subset[subset['validation_type'] == 'sample_based']
                site_based = subset[subset['validation_type'] == 'site_based']
                
                if len(sample_based) > 0 and len(site_based) > 0:
                    sb_r2 = sample_based['r2_score'].iloc[0]
                    stb_r2 = site_based['r2_score'].iloc[0]
                    
                    # Robustness = how similar performance is across validation strategies
                    r2_difference = abs(sb_r2 - stb_r2)
                    mean_r2 = (sb_r2 + stb_r2) / 2
                    relative_difference = r2_difference / mean_r2 if mean_r2 > 0 else np.inf
                    
                    robustness_metrics.append({
                        variable_col: variable_value,
                        'sample_based_r2': sb_r2,
                        'site_based_r2': stb_r2,
                        'mean_r2': mean_r2,
                        'r2_difference': r2_difference,
                        'relative_difference_pct': relative_difference * 100,
                        'robustness_score': 1 / (1 + relative_difference)  # Higher = more robust
                    })
        
        if robustness_metrics:
            robustness_df = pd.DataFrame(robustness_metrics)
            robustness_df = robustness_df.sort_values('robustness_score', ascending=False)
            
            print("\nRobustness Ranking (Higher score = more robust across validation strategies):")
            print(robustness_df.to_string(index=False, float_format='%.3f'))
            
            # Identify most robust option
            most_robust = robustness_df.iloc[0]
            print(f"\nMOST ROBUST {variable_name}: {most_robust[variable_col]}")
            print(f"  Sample-based R²: {most_robust['sample_based_r2']:.3f}")
            print(f"  Site-based R²: {most_robust['site_based_r2']:.3f}")
            print(f"  R² difference: {most_robust['r2_difference']:.3f} ({most_robust['relative_difference_pct']:.1f}%)")
            print(f"  Robustness score: {most_robust['robustness_score']:.3f}")
            
            # Save robustness analysis (commented out to reduce file clutter)
            # analysis_path = self.results_dir / f"{exp_results[0]['experiment']}_robustness_analysis.csv"
            # robustness_df.to_csv(analysis_path, index=False)
            # print(f"\nRobustness analysis saved to: {analysis_path}")
    
    def run_all_experiments(self):
        """Run all three experiments in sequence"""
        print("Starting systematic robustness experiments...")
        
        # Run experiments
        exp1_results = self.experiment_1_model_comparison()
        exp2_results = self.experiment_2_log_transform()
        exp3_results = self.experiment_3_method_comparison()
        
        # Combine all results
        all_results = exp1_results + exp2_results + exp3_results
        
        # Save comprehensive summary
        self._save_comprehensive_summary(all_results)
        
        # Create comparison visualizations
        self._create_comparison_plots(all_results)
        
        return all_results
    
    def _save_comprehensive_summary(self, all_results):
        """Save comprehensive summary of all experiments"""
        if not all_results:
            print("No results to summarize")
            return
        
        results_df = pd.DataFrame(all_results)
        
        # Save detailed results
        summary_path = self.results_dir / 'comprehensive_experiment_summary.csv'
        results_df.to_csv(summary_path, index=False)
        
        # Create summary statistics
        successful_df = results_df[results_df['status'] == 'success'].copy()
        
        if len(successful_df) > 0:
            print(f"\n{'='*80}")
            print("COMPREHENSIVE EXPERIMENT SUMMARY")
            print(f"{'='*80}")
            print(f"Total experiments: {len(results_df)}")
            print(f"Successful: {len(successful_df)}")
            print(f"Failed: {len(results_df) - len(successful_df)}")
            
            # Best overall configuration
            best_config = successful_df.loc[successful_df['r2_score'].idxmax()]
            print(f"\nBEST OVERALL CONFIGURATION:")
            print(f"  Model: {best_config['model_type']}")
            print(f"  Method: {best_config['method']}")
            print(f"  Log Transform: {best_config['target_log_transform']}")
            print(f"  Validation: {best_config['validation_type']}")
            print(f"  R² Score: {best_config['r2_score']:.3f}")
            print(f"  RMSE: {best_config['rmse']:.3f}")
            
            print(f"\nDetailed results saved to: {summary_path}")
        
    def _create_comparison_plots(self, all_results):
        """Create comprehensive comparison plots"""
        if not all_results:
            return
        
        results_df = pd.DataFrame(all_results)
        successful_df = results_df[results_df['status'] == 'success'].copy()
        
        if len(successful_df) == 0:
            print("No successful results to plot")
            return
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: R² by experiment and validation type
        experiment_order = ['model_comparison', 'log_transform', 'method_comparison']
        
        for i, experiment in enumerate(experiment_order):
            exp_data = successful_df[successful_df['experiment'] == experiment]
            if len(exp_data) > 0:
                sample_data = exp_data[exp_data['validation_type'] == 'sample_based']
                site_data = exp_data[exp_data['validation_type'] == 'site_based']
                
                sample_r2 = sample_data['r2_score'].values
                site_r2 = site_data['r2_score'].values
                
                x_pos = np.arange(len(sample_r2))
                width = 0.35
                
                if i == 0:  # Model comparison
                    bars1 = axes[0,0].bar(x_pos - width/2, sample_r2, width, label='Sample-based', alpha=0.8, color='skyblue')
                    bars2 = axes[0,0].bar(x_pos + width/2, site_r2, width, label='Site-based', alpha=0.8, color='lightcoral')
                    axes[0,0].set_title('Model Comparison')
                    axes[0,0].set_xticks(x_pos)
                    axes[0,0].set_xticklabels(sample_data['model_type'].values, rotation=45)
                    
                    # Add R² values on bars
                    for bar, value in zip(bars1, sample_r2):
                        axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                                     f'{value:.3f}', ha='center', va='bottom', fontsize=9)
                    for bar, value in zip(bars2, site_r2):
                        axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                                     f'{value:.3f}', ha='center', va='bottom', fontsize=9)
                    
                elif i == 1:  # Log transform
                    bars1 = axes[0,1].bar(x_pos - width/2, sample_r2, width, label='Sample-based', alpha=0.8, color='skyblue')
                    bars2 = axes[0,1].bar(x_pos + width/2, site_r2, width, label='Site-based', alpha=0.8, color='lightcoral')
                    axes[0,1].set_title('Log Transform Impact')
                    axes[0,1].set_xticks(x_pos)
                    axes[0,1].set_xticklabels([f'Log={val}' for val in sample_data['target_log_transform'].values])
                    
                    # Add R² values on bars
                    for bar, value in zip(bars1, sample_r2):
                        axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                                     f'{value:.3f}', ha='center', va='bottom', fontsize=9)
                    for bar, value in zip(bars2, site_r2):
                        axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                                     f'{value:.3f}', ha='center', va='bottom', fontsize=9)
                    
                else:  # Method comparison
                    bars1 = axes[1,0].bar(x_pos - width/2, sample_r2, width, label='Sample-based', alpha=0.8, color='skyblue')
                    bars2 = axes[1,0].bar(x_pos + width/2, site_r2, width, label='Site-based', alpha=0.8, color='lightcoral')
                    axes[1,0].set_title('Method Comparison')
                    axes[1,0].set_xticks(x_pos)
                    axes[1,0].set_xticklabels(sample_data['method'].values)
                    
                    # Add R² values on bars
                    for bar, value in zip(bars1, sample_r2):
                        axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                                     f'{value:.3f}', ha='center', va='bottom', fontsize=9)
                    for bar, value in zip(bars2, site_r2):
                        axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                                     f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Add legend and labels
        for ax in [axes[0,0], axes[0,1], axes[1,0]]:
            ax.set_ylabel('R² Score')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
        
        # Plot 4: Robustness heatmap
        robustness_data = []
        for experiment in experiment_order:
            exp_data = successful_df[successful_df['experiment'] == experiment]
            for variable_value in exp_data['variable_value'].unique():
                subset = exp_data[exp_data['variable_value'] == variable_value]
                if len(subset) >= 2:
                    sample_r2 = subset[subset['validation_type'] == 'sample_based']['r2_score'].iloc[0]
                    site_r2 = subset[subset['validation_type'] == 'site_based']['r2_score'].iloc[0]
                    difference = abs(sample_r2 - site_r2)
                    
                    robustness_data.append({
                        'experiment': experiment,
                        'variable': str(variable_value),
                        'r2_difference': difference,
                        'mean_r2': (sample_r2 + site_r2) / 2
                    })
        
        if robustness_data:
            rob_df = pd.DataFrame(robustness_data)
            pivot_rob = rob_df.pivot(index='variable', columns='experiment', values='r2_difference')
            
            sns.heatmap(pivot_rob, annot=True, fmt='.3f', ax=axes[1,1], 
                       cmap='YlOrRd_r', cbar_kws={'label': 'R² Difference'})
            axes[1,1].set_title('Validation Strategy Robustness\n(Lower = More Robust)')
            axes[1,1].set_xlabel('Experiment')
            axes[1,1].set_ylabel('Configuration')
        
        plt.tight_layout()
        
        # Save comparison plot
        comparison_path = self.results_dir / 'comprehensive_comparison_plots.png'
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comparison plots saved to: {comparison_path}")


# Example usage
if __name__ == "__main__":
    # Define base configuration
    BASE_CONFIG = {
        'features': [
            # Satellite features
            'sat_at_target', 'sat_decay_1_3km', 'sat_directional_asymmetry',
            
            # Meteorology features  
            'dewpoint_temp_2m', 'temp_2m', 'u_wind_10m', 'v_wind_10m',
            
            # Facility features
            'facility_height', 'distance_to_facility', 
            'bearing_from_facility', 'NEI_annual_emission_t', 'monthly_emission_rate_t_per_hr',
            
            # Topographical features
            'elevation', 'elevation_diff_from_facility', 'terrain_slope', 'terrain_roughness',
            
            # Road features
            'road_density_interstate'
        ],
        'test_size': 0.2,
        'random_state': 42,
        'output_dir': f"{MODELS_PATH}/robustness_experiments"
    }
    
    # Define data paths
    DATA_PATHS = {
        'X_features': f"{DATA_PATH}/processed_data/climate_trace_9/X_features_all_facilities_mar2023.csv",
        'y_targets_template': f"{DATA_PATH}/processed_data/climate_trace_9/y_{{method}}_all_facilities_mar2023.csv"
    }
    
    # Run experiments
    experiment_runner = ExperimentRunner(BASE_CONFIG, DATA_PATHS)
    all_results = experiment_runner.run_all_experiments()
    
    print(f"\n{'='*80}")
    print("ALL EXPERIMENTS COMPLETED!")
    print(f"Check results in: {BASE_CONFIG['output_dir']}")
    print(f"{'='*80}")