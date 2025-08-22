import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
import warnings
import time
from utils.path_util import DATA_PATH,MODELS_PATH
warnings.filterwarnings('ignore')

class ComprehensiveSiteBasedCV:
    """
    Comprehensive site-based cross-validation testing all 9 facilities
    Each facility is held out as test set once, with heatmap generation
    """
    
    def __init__(self, config):
        """
        Initialize comprehensive site-based CV
        
        config = {
            'method': 'method1',  # method to test
            'features': [...],    # feature list
            'target_log_transform': True,
            'model_type': 'RF',
            'hyperparameter_tuning': True,
            'tuning_scope': 'thorough',
            'random_state': 42,
            'output_dir': './site_cv_results',
            'heatmap_style': 'side_by_side'  # 'side_by_side', 'difference', or 'both'
        }
        """
        self.config = config
        self.results = []
        self.facility_models = {}
        self.facility_metrics = {}
        self.feature_names = None
        
        # Create output directory
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / 'heatmaps').mkdir(exist_ok=True)
        (self.output_dir / 'facility_plots').mkdir(exist_ok=True)
        
    def load_and_prepare_data(self, X_features_path, y_targets_path):
        """Load and prepare data for site-based CV"""
        print("Loading and preparing data...")
        
        # Load data
        X_features = pd.read_csv(X_features_path)
        y_targets = pd.read_csv(y_targets_path)
        
        print(f"Loaded data: {len(X_features)} samples")
        
        # Get all facilities
        all_facilities = sorted(X_features['facility_id'].unique())
        print(f"All facilities: {all_facilities}")
        print(f"Total facilities: {len(all_facilities)}")
        
        # Prepare features
        feature_cols = self.config['features'].copy()
        essential_cols = ['month', 'grid_i', 'grid_j', 'facility_id']
        for col in essential_cols:
            if col not in feature_cols and col in X_features.columns:
                feature_cols.append(col)
        
        available_features = [col for col in feature_cols if col in X_features.columns]
        missing_features = [col for col in self.config['features'] if col not in X_features.columns]
        
        if missing_features:
            print(f"Warning: Missing features: {missing_features}")
        
        # Prepare X and y
        X = X_features[available_features].copy()
        y = y_targets['pm25_concentration'].copy()
        
        # Remove rows with NaN values
        valid_mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[valid_mask].reset_index(drop=True)
        y = y[valid_mask].reset_index(drop=True)
        
        print(f"After cleaning: {len(X)} samples")
        
        # Log transform target if specified
        if self.config['target_log_transform']:
            y_min = y[y > 0].min() if (y > 0).any() else 1e-6
            y = np.log(y + y_min * 0.01)
            print("Applied log transform to target")
        
        # Store feature names (exclude non-predictive columns)
        exclude_cols = ['month', 'grid_i', 'grid_j', 'facility_id']
        self.feature_names = [col for col in X.columns if col not in exclude_cols]
        
        print(f"Using {len(self.feature_names)} features for modeling")
        
        return X, y, all_facilities
    
    def run_comprehensive_site_cv(self, X, y, all_facilities):
        """Run site-based CV with each facility as test set"""
        print(f"\n{'='*80}")
        print("RUNNING COMPREHENSIVE SITE-BASED CROSS-VALIDATION")
        print(f"Testing all {len(all_facilities)} facilities")
        print(f"{'='*80}")
        
        cv_results = []
        
        for i, test_facility in enumerate(all_facilities):
            print(f"\nFold {i+1}/{len(all_facilities)}: Testing facility '{test_facility}'")
            print("-" * 60)
            
            # Split data: test_facility vs all others
            test_mask = X['facility_id'] == test_facility
            train_mask = ~test_mask
            
            X_train_full = X[train_mask].reset_index(drop=True)
            X_test = X[test_mask].reset_index(drop=True)
            y_train_full = y[train_mask].reset_index(drop=True)
            y_test = y[test_mask].reset_index(drop=True)
            
            if len(X_test) == 0:
                print(f"  ✗ No test data for facility {test_facility}")
                continue
            
            # Further split training data into train/validation (80/20 of remaining data)
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_full, y_train_full, 
                test_size=0.2, 
                random_state=self.config['random_state']
            )
            
            print(f"  Train: {len(X_train)} samples ({len(X_train['facility_id'].unique())} facilities)")
            print(f"  Validation: {len(X_val)} samples ({len(X_val['facility_id'].unique())} facilities)")
            print(f"  Test: {len(X_test)} samples (facility: {test_facility})")
            
            # Train model for this fold
            model, best_params, tuning_improvement, train_time = self._train_model_for_fold(
                X_train[self.feature_names], y_train,
                X_val[self.feature_names], y_val,
                test_facility
            )
            
            # Add this before y_pred = model.predict():
            predict_start_time = time.time()
            y_pred = model.predict(X_test[self.feature_names])
            predict_end_time = time.time()
            predict_time = predict_end_time - predict_start_time

            print(f"  Training time: {train_time:.3f}s, Prediction time: {predict_time:.3f}s")
            
            # Calculate metrics
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            print(f"  Results: R² = {r2:.3f}, RMSE = {rmse:.3f}, MAE = {mae:.3f}")
            
            # Store results
            fold_result = {
                'fold': i + 1,
                'test_facility': test_facility,
                'train_facilities': sorted(X_train['facility_id'].unique()),
                'n_train': len(X_train),
                'n_val': len(X_val),
                'n_test': len(X_test),
                'r2_score': r2,
                'rmse': rmse,
                'mae': mae,
                'best_params': best_params,
                'tuning_improvement': tuning_improvement,
                'model': model,
                'test_data': {
                    'X_test': X_test,
                    'y_test': y_test,
                    'y_pred': y_pred
                },
                'train_time': train_time,
                'predict_time': predict_time,
                'predict_time_per_sample': predict_time / len(X_test)
            }
            
            cv_results.append(fold_result)
            self.facility_models[test_facility] = model
            self.facility_metrics[test_facility] = {'r2': r2, 'rmse': rmse, 'mae': mae}
            
            # Create facility-specific plots
            self._create_facility_plots(fold_result, i + 1)
            
            # Create heatmaps for this facility
            self._create_facility_heatmaps(fold_result)
        
        # Store all results
        self.results = cv_results
        
        # Create comprehensive analysis
        self._create_comprehensive_analysis()
        
        return cv_results
    
    def _train_model_for_fold(self, X_train, y_train, X_val, y_val, test_facility):
        """Train model with hyperparameter tuning for specific fold"""
        print(f"    Training model for test facility: {test_facility}")
        
        # Initialize model
        base_model = RandomForestRegressor(
            random_state=self.config['random_state'],
            n_jobs=-1
        )
        
        if not self.config.get('hyperparameter_tuning', False):
            # Use default parameters
            base_model.fit(X_train, y_train)
            return base_model, "default", 0.0
        
        # Hyperparameter tuning
        param_grid = self._get_param_grid()
        
        print(f"    Hyperparameter tuning with {len(param_grid)} parameter combinations...")
        
        # Manual hyperparameter search using validation set
        best_score = -np.inf
        best_params = None
        best_model = None
        default_score = None
        
        for params in param_grid:
            # Add this before model.fit():
            print(f"    Training model for test facility: {test_facility}")
            train_start_time = time.time()
            model = RandomForestRegressor(**params, 
                                        random_state=self.config['random_state'], 
                                        n_jobs=-1)
            model.fit(X_train, y_train)
            # After model.fit():
            train_end_time = time.time()
            train_time = train_end_time - train_start_time
            # Evaluate on validation set
            val_pred = model.predict(X_val)
            val_score = r2_score(y_val, val_pred)
            
            # Track default performance (first parameter combination)
            if default_score is None:
                default_score = val_score
            
            if val_score > best_score:
                best_score = val_score
                best_params = params
                best_model = model
        
        tuning_improvement = best_score - default_score if default_score else 0.0
        
        print(f"    Best validation R²: {best_score:.3f}")
        print(f"    Improvement from tuning: +{tuning_improvement:.3f}")
        print(f"    Best parameters: {best_params}")
        
        return best_model, best_params, tuning_improvement, train_time
    
    def _get_param_grid(self):
        """Get parameter grid based on tuning scope"""
        scope = self.config.get('tuning_scope', 'quick')
        
        if scope == 'quick':
            param_combinations = [
                {'n_estimators': 100, 'max_depth': None, 'min_samples_split': 2},
                {'n_estimators': 200, 'max_depth': None, 'min_samples_split': 2},
                {'n_estimators': 100, 'max_depth': 20, 'min_samples_split': 2},
                {'n_estimators': 200, 'max_depth': 20, 'min_samples_split': 5}
            ]
        else:  # thorough
            param_combinations = []
            for n_est in [100, 200, 500]:
                for max_depth in [None, 10, 20, 30]:
                    for min_split in [2, 5, 10]:
                        for max_feat in ['sqrt', 0.3]:
                            param_combinations.append({
                                'n_estimators': n_est,
                                'max_depth': max_depth,
                                'min_samples_split': min_split,
                                'max_features': max_feat
                            })
        
        return param_combinations
    
    def _create_facility_plots(self, fold_result, fold_num):
        """Create individual plots for each facility fold"""
        test_facility = fold_result['test_facility']
        y_test = fold_result['test_data']['y_test']
        y_pred = fold_result['test_data']['y_pred']
        r2 = fold_result['r2_score']
        rmse = fold_result['rmse']
        
        # Create scatter plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Scatter plot with density
        try:
            from scipy.stats import gaussian_kde
            xy = np.vstack([y_test, y_pred])
            density = gaussian_kde(xy)(xy)
        except:
            density = np.ones(len(y_test))
        
        scatter = ax1.scatter(y_test, y_pred, c=density, cmap='viridis', alpha=0.7, s=30)
        
        # Add 1:1 line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
        
        ax1.set_xlabel('Measured PM₂.₅ (log)')
        ax1.set_ylabel('Predicted PM₂.₅ (log)')
        ax1.set_title(f'Fold {fold_num}: {test_facility}\nR² = {r2:.3f}, RMSE = {rmse:.3f}')
        ax1.grid(alpha=0.3)
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax1, label='Point Density')
        
        # Plot 2: Feature importance
        if hasattr(fold_result['model'], 'feature_importances_'):
            importances = fold_result['model'].feature_importances_
            
            # Get top 10 features
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False).head(10)
            
            feature_importance = feature_importance.sort_values('importance', ascending=True)
            
            bars = ax2.barh(feature_importance['feature'], feature_importance['importance'])
            bars[-1].set_color('#d62728')  # Highlight most important
            bars[-2].set_color('#ff7f0e')  # Second most important
            
            ax2.set_xlabel('Feature Importance')
            ax2.set_title(f'Top 10 Feature Importance\n{test_facility}')
            ax2.grid(axis='x', alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Feature importance\nnot available', 
                   ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title(f'Feature Importance\n{test_facility}')
        
        plt.tight_layout()
        
        # Save facility plot
        facility_plot_path = self.output_dir / 'facility_plots' / f'fold_{fold_num:02d}_{test_facility}_analysis.png'
        plt.savefig(facility_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    Saved facility plot: {facility_plot_path.name}")
    
    def _create_facility_heatmaps(self, fold_result):
        """Create spatial heatmaps for the test facility"""
        test_facility = fold_result['test_facility']
        X_test = fold_result['test_data']['X_test']
        y_test = fold_result['test_data']['y_test']
        y_pred = fold_result['test_data']['y_pred']
        
        if len(X_test) < 3:
            print(f"    Skipping heatmap for {test_facility}: insufficient data points")
            return
        
        # Check if we have spatial coordinates
        if 'grid_i' not in X_test.columns or 'grid_j' not in X_test.columns:
            print(f"    Skipping heatmap for {test_facility}: no spatial coordinates")
            return
        
        print(f"    Creating heatmap for {test_facility}")
        
        grid_i = X_test['grid_i'].values
        grid_j = X_test['grid_j'].values
        
        heatmap_style = self.config.get('heatmap_style', 'side_by_side')
        
        # Convert log values back to original scale for display
        if self.config['target_log_transform']:
            y_test_display = np.exp(y_test)
            y_pred_display = np.exp(y_pred)
        else:
            y_test_display = y_test
            y_pred_display = y_pred

        # Calculate shared color scale using unlog values
        combined_values = np.concatenate([y_test_display, y_pred_display])
        vmin = np.percentile(combined_values, 5)   # 5th percentile as minimum
        vmax = np.percentile(combined_values, 95)  # 95th percentile as maximum

        if heatmap_style == 'side_by_side':
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Plot 1: Measured values
            scatter1 = ax1.scatter(grid_i, grid_j, c=y_test_display, cmap='jet', s=60, alpha=0.8, vmin=vmin, vmax=vmax)
            ax1.set_xlabel('Grid I')
            ax1.set_ylabel('Grid J')
            ax1.set_title(f'Measured PM₂.₅\n{test_facility}')
            ax1.grid(True, alpha=0.3)
            ax1.set_aspect('equal')
            cbar1 = plt.colorbar(scatter1, ax=ax1)
            cbar1.set_label('PM₂.₅ (μg/m³)')
            
            # Plot 2: Predicted values
            scatter2 = ax2.scatter(grid_i, grid_j, c=y_pred_display, cmap='jet', s=60, alpha=0.8, vmin=vmin, vmax=vmax)
            ax2.set_xlabel('Grid I')
            ax2.set_ylabel('Grid J')
            ax2.set_title(f'Predicted PM₂.₅\n{test_facility}')
            ax2.grid(True, alpha=0.3)
            ax2.set_aspect('equal')
            cbar2 = plt.colorbar(scatter2, ax=ax2)
            cbar2.set_label('PM₂.₅ (μg/m³)')
            
        elif heatmap_style == 'difference':
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            
            # Plot prediction error
            error = y_pred_display - y_test_display
            vmax = max(abs(error.min()), abs(error.max()))
            
            scatter = ax.scatter(grid_i, grid_j, c=error, cmap='RdBu_r', 
                               vmin=-vmax, vmax=vmax, s=60, alpha=0.8)
            ax.set_xlabel('Grid I')
            ax.set_ylabel('Grid J')
            ax.set_title(f'Prediction Error (Predicted - Measured)\n{test_facility}')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Error (μg/m³)')
            
        else:  # both
            fig, ((ax1, ax2), (ax3, ax_empty)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Plot 1: Measured
            scatter1 = ax1.scatter(grid_i, grid_j, c=y_test_display, cmap='jet', s=60, alpha=0.8)
            ax1.set_xlabel('Grid I')
            ax1.set_ylabel('Grid J')
            ax1.set_title(f'Measured PM₂.₅')
            ax1.grid(True, alpha=0.3)
            ax1.set_aspect('equal')
            plt.colorbar(scatter1, ax=ax1, label='PM₂.₅ (μg/m³)')
            
            # Plot 2: Predicted
            scatter2 = ax2.scatter(grid_i, grid_j, c=y_pred_display, cmap='jet', s=60, alpha=0.8)
            ax2.set_xlabel('Grid I')
            ax2.set_ylabel('Grid J')
            ax2.set_title(f'Predicted PM₂.₅')
            ax2.grid(True, alpha=0.3)
            ax2.set_aspect('equal')
            plt.colorbar(scatter2, ax=ax2, label='PM₂.₅ (μg/m³)')
            
            # Plot 3: Error
            error = y_pred - y_test
            vmax = max(abs(error.min()), abs(error.max()))
            scatter3 = ax3.scatter(grid_i, grid_j, c=error, cmap='RdBu_r', 
                                 vmin=-vmax, vmax=vmax, s=60, alpha=0.8)
            ax3.set_xlabel('Grid I')
            ax3.set_ylabel('Grid J')
            ax3.set_title(f'Prediction Error')
            ax3.grid(True, alpha=0.3)
            ax3.set_aspect('equal')
            plt.colorbar(scatter3, ax=ax3, label='Error (μg/m³)')
            
            # Hide fourth subplot
            ax_empty.axis('off')
        
        # Add overall title with metrics
        r2 = fold_result['r2_score']
        rmse = fold_result['rmse']
        n_points = len(X_test)
        
        fig.suptitle(f'Site-Based CV Heatmap: {test_facility}\n'
                    f'R² = {r2:.3f}, RMSE = {rmse:.3f}, N = {n_points}',
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save heatmap
        heatmap_filename = f'{test_facility}_heatmap_{heatmap_style}.png'
        heatmap_path = self.output_dir / 'heatmaps' / heatmap_filename
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    Saved heatmap: {heatmap_filename}")
    
    def _create_comprehensive_analysis(self):
        """Create comprehensive analysis across all folds"""
        print(f"\n{'='*80}")
        print("CREATING COMPREHENSIVE ANALYSIS")
        print(f"{'='*80}")
        
        if not self.results:
            print("No results to analyze")
            return
        
        # Extract metrics
        facilities = [r['test_facility'] for r in self.results]
        r2_scores = [r['r2_score'] for r in self.results]
        rmse_scores = [r['rmse'] for r in self.results]
        mae_scores = [r['mae'] for r in self.results]
        n_test_samples = [r['n_test'] for r in self.results]
        
        # Create comprehensive analysis plots
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # Plot 1: R² scores by facility
        bars1 = axes[0, 0].bar(range(len(facilities)), r2_scores, color='skyblue', alpha=0.8)
        axes[0, 0].set_xticks(range(len(facilities)))
        axes[0, 0].set_xticklabels(facilities, rotation=45, ha='right')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].set_title('Site-Based CV Performance\nR² Score by Test Facility')
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Add R² values on bars
        for bar, r2 in zip(bars1, r2_scores):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                          f'{r2:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Add mean line
        mean_r2 = np.mean(r2_scores)
        axes[0, 0].axhline(y=mean_r2, color='red', linestyle='--', alpha=0.8, linewidth=2)
        axes[0, 0].text(0.02, 0.98, f'Mean R²: {mean_r2:.3f}', transform=axes[0, 0].transAxes, 
                       fontsize=12, fontweight='bold', va='top',
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        # Plot 2: RMSE scores by facility
        bars2 = axes[0, 1].bar(range(len(facilities)), rmse_scores, color='lightcoral', alpha=0.8)
        axes[0, 1].set_xticks(range(len(facilities)))
        axes[0, 1].set_xticklabels(facilities, rotation=45, ha='right')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].set_title('Site-Based CV Performance\nRMSE by Test Facility')
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # Add RMSE values on bars
        for bar, rmse in zip(bars2, rmse_scores):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(rmse_scores)*0.01, 
                          f'{rmse:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Add mean line
        mean_rmse = np.mean(rmse_scores)
        axes[0, 1].axhline(y=mean_rmse, color='red', linestyle='--', alpha=0.8, linewidth=2)
        axes[0, 1].text(0.02, 0.98, f'Mean RMSE: {mean_rmse:.3f}', transform=axes[0, 1].transAxes, 
                       fontsize=12, fontweight='bold', va='top',
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        # Plot 3: Sample sizes by facility
        bars3 = axes[0, 2].bar(range(len(facilities)), n_test_samples, color='lightgreen', alpha=0.8)
        axes[0, 2].set_xticks(range(len(facilities)))
        axes[0, 2].set_xticklabels(facilities, rotation=45, ha='right')
        axes[0, 2].set_ylabel('Number of Test Samples')
        axes[0, 2].set_title('Test Sample Distribution\nby Facility')
        axes[0, 2].grid(axis='y', alpha=0.3)
        
        # Add sample counts on bars
        for bar, n_samples in zip(bars3, n_test_samples):
            axes[0, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(n_test_samples)*0.01, 
                          str(n_samples), ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Plot 4: Performance distribution
        axes[1, 0].hist(r2_scores, bins=min(len(r2_scores), 10), alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 0].axvline(mean_r2, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_r2:.3f}')
        axes[1, 0].set_xlabel('R² Score')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('R² Score Distribution\nAcross All Facilities')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        # Plot 5: Performance vs sample size
        scatter = axes[1, 1].scatter(n_test_samples, r2_scores, s=100, alpha=0.7, c=r2_scores, cmap='viridis')
        
        # Add facility labels
        for i, facility in enumerate(facilities):
            axes[1, 1].annotate(facility, (n_test_samples[i], r2_scores[i]), 
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        axes[1, 1].set_xlabel('Number of Test Samples')
        axes[1, 1].set_ylabel('R² Score')
        axes[1, 1].set_title('Performance vs Sample Size')
        axes[1, 1].grid(alpha=0.3)
        plt.colorbar(scatter, ax=axes[1, 1], label='R² Score')
        
        # Plot 6: Summary statistics table
        axes[1, 2].axis('off')
        
        # Create summary statistics
        summary_stats = {
            'Metric': ['R² Score', 'RMSE', 'MAE', 'Test Samples'],
            'Mean': [np.mean(r2_scores), np.mean(rmse_scores), np.mean(mae_scores), np.mean(n_test_samples)],
            'Std': [np.std(r2_scores), np.std(rmse_scores), np.std(mae_scores), np.std(n_test_samples)],
            'Min': [np.min(r2_scores), np.min(rmse_scores), np.min(mae_scores), np.min(n_test_samples)],
            'Max': [np.max(r2_scores), np.max(rmse_scores), np.max(mae_scores), np.max(n_test_samples)]
        }
        
        summary_df = pd.DataFrame(summary_stats)
        
        # Create table
        table_data = []
        for _, row in summary_df.iterrows():
            if row['Metric'] == 'Test Samples':
                table_data.append([row['Metric'], f"{row['Mean']:.0f}", f"{row['Std']:.0f}", 
                                 f"{row['Min']:.0f}", f"{row['Max']:.0f}"])
            else:
                table_data.append([row['Metric'], f"{row['Mean']:.3f}", f"{row['Std']:.3f}", 
                                 f"{row['Min']:.3f}", f"{row['Max']:.3f}"])
        
        table = axes[1, 2].table(cellText=table_data,
                               colLabels=['Metric', 'Mean', 'Std', 'Min', 'Max'],
                               cellLoc='center',
                               loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        axes[1, 2].set_title('Summary Statistics\nSite-Based Cross-Validation', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        # Save comprehensive analysis
        analysis_path = self.output_dir / 'comprehensive_site_cv_analysis.png'
        plt.savefig(analysis_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save detailed results CSV
        results_data = []
        for result in self.results:
            results_data.append({
                'fold': result['fold'],
                'test_facility': result['test_facility'],
                'train_facilities': '; '.join(result['train_facilities']),
                'n_train': result['n_train'],
                'n_val': result['n_val'],
                'n_test': result['n_test'],
                'r2_score': result['r2_score'],
                'rmse': result['rmse'],
                'mae': result['mae'],
                'best_params': str(result['best_params']),
                'tuning_improvement': result['tuning_improvement']
            })
        
        results_df = pd.DataFrame(results_data)
        results_csv_path = self.output_dir / 'site_cv_detailed_results.csv'
        results_df.to_csv(results_csv_path, index=False)
        
        # Print summary to console
        print(f"Analysis saved to: {analysis_path}")
        print(f"Detailed results saved to: {results_csv_path}")
        print(f"\nSITE-BASED CROSS-VALIDATION SUMMARY:")
        print(f"  Number of facilities tested: {len(facilities)}")
        print(f"  Mean R² Score: {mean_r2:.3f} ± {np.std(r2_scores):.3f}")
        print(f"  Mean RMSE: {mean_rmse:.3f} ± {np.std(rmse_scores):.3f}")
        print(f"  Best performing facility: {facilities[np.argmax(r2_scores)]} (R² = {max(r2_scores):.3f})")
        print(f"  Worst performing facility: {facilities[np.argmin(r2_scores)]} (R² = {min(r2_scores):.3f})")
        
        return analysis_path, results_csv_path
    
    def create_facility_comparison_heatmap_grid(self):
        """Create a grid of all facility heatmaps for easy comparison"""
        if not self.results:
            print("No results available for facility comparison grid")
            return
        
        print("Creating facility comparison heatmap grid...")
        
        # Calculate grid dimensions
        n_facilities = len(self.results)
        n_cols = 3  # 3 columns for better layout
        n_rows = (n_facilities + n_cols - 1) // n_cols
        
        # Create large figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5))
        
        # Flatten axes for easier indexing
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        # Create mini-heatmap for each facility
        for i, result in enumerate(self.results):
            test_facility = result['test_facility']
            X_test = result['test_data']['X_test']
            y_test = result['test_data']['y_test']
            y_pred = result['test_data']['y_pred']
            r2 = result['r2_score']
            
            ax = axes[i]
            
            if len(X_test) >= 3 and 'grid_i' in X_test.columns and 'grid_j' in X_test.columns:
                # Create prediction error heatmap
                grid_i = X_test['grid_i'].values
                grid_j = X_test['grid_j'].values
                error = y_pred - y_test
                
                # Use consistent color scale across all facilities
                global_vmax = 2.0  # Adjust based on your data range
                
                scatter = ax.scatter(grid_i, grid_j, c=error, cmap='RdBu_r', 
                                   vmin=-global_vmax, vmax=global_vmax, s=30, alpha=0.8)
                
                ax.set_xlabel('Grid I', fontsize=8)
                ax.set_ylabel('Grid J', fontsize=8)
                ax.set_title(f'{test_facility}\nR² = {r2:.3f}', fontsize=10, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.set_aspect('equal')
                
                # Smaller tick labels
                ax.tick_params(axis='both', which='major', labelsize=8)
                
            else:
                # No spatial data available
                ax.text(0.5, 0.5, f'{test_facility}\nNo spatial data\nR² = {r2:.3f}', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
        
        # Hide unused subplots
        for i in range(len(self.results), len(axes)):
            axes[i].axis('off')
        
        # Add overall title and colorbar
        fig.suptitle('Site-Based Cross-Validation: All Facility Heatmaps\n'
                    'Color shows Prediction Error (Predicted - Measured)', 
                    fontsize=16, fontweight='bold')
        
        # Add shared colorbar
        if len(self.results) > 0 and any('grid_i' in r['test_data']['X_test'].columns for r in self.results):
            # Create colorbar
            sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=plt.Normalize(vmin=-2.0, vmax=2.0))
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', pad=0.02, shrink=0.8)
            cbar.set_label('Prediction Error (log scale)', fontsize=12)
        
        plt.tight_layout()
        
        # Save facility comparison grid
        grid_path = self.output_dir / 'all_facilities_heatmap_grid.png'
        plt.savefig(grid_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Facility comparison grid saved to: {grid_path}")
        return grid_path
    
    def print_detailed_summary(self):
        """Print detailed summary of all results"""
        if not self.results:
            print("No results to summarize")
            return
        
        print(f"\n{'='*100}")
        print("DETAILED SITE-BASED CROSS-VALIDATION SUMMARY")
        print(f"{'='*100}")
        
        # Overall statistics
        r2_scores = [r['r2_score'] for r in self.results]
        rmse_scores = [r['rmse'] for r in self.results]
        
        print(f"\nOVERALL PERFORMANCE:")
        print(f"  Cross-Validation R² Score: {np.mean(r2_scores):.3f} ± {np.std(r2_scores):.3f}")
        print(f"  Cross-Validation RMSE: {np.mean(rmse_scores):.3f} ± {np.std(rmse_scores):.3f}")
        print(f"  Range: R² = [{np.min(r2_scores):.3f}, {np.max(r2_scores):.3f}]")
        
        # Individual facility results
        print(f"\nINDIVIDUAL FACILITY RESULTS:")
        print(f"{'Facility':<15} {'R²':<8} {'RMSE':<8} {'MAE':<8} {'N_test':<8} {'Improvement':<12}")
        print("-" * 80)
        
        for result in sorted(self.results, key=lambda x: x['r2_score'], reverse=True):
            facility = result['test_facility']
            r2 = result['r2_score']
            rmse = result['rmse']
            mae = result['mae']
            n_test = result['n_test']
            improvement = result['tuning_improvement']
            
            print(f"{facility:<15} {r2:<8.3f} {rmse:<8.3f} {mae:<8.3f} {n_test:<8} {improvement:<12.3f}")
        
        # Best parameters summary
        print(f"\nHYPERPARAMETER TUNING SUMMARY:")
        improvements = [r['tuning_improvement'] for r in self.results]
        print(f"  Mean tuning improvement: {np.mean(improvements):.3f} ± {np.std(improvements):.3f}")
        print(f"  Best improvement: {np.max(improvements):.3f}")
        
        # Feature importance analysis (if available)
        if hasattr(self.results[0]['model'], 'feature_importances_'):
            print(f"\nFEATURE IMPORTANCE ANALYSIS:")
            
            # Average feature importance across all models
            all_importances = np.array([r['model'].feature_importances_ for r in self.results])
            mean_importance = np.mean(all_importances, axis=0)
            std_importance = np.std(all_importances, axis=0)
            
            feature_importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'mean_importance': mean_importance,
                'std_importance': std_importance
            }).sort_values('mean_importance', ascending=False)
            
            print("  Top 10 Most Important Features (averaged across all folds):")
            for i, row in feature_importance_df.head(10).iterrows():
                print(f"    {row['feature']:<30} {row['mean_importance']:.3f} ± {row['std_importance']:.3f}")
        # Add after the existing summary sections:
        print(f"\nTIMING ANALYSIS:")
        train_times = [r['train_time'] for r in self.results]
        predict_times = [r['predict_time'] for r in self.results]
        predict_per_sample = [r['predict_time_per_sample'] for r in self.results]

        print(f"  Mean training time: {np.mean(train_times):.3f}s ± {np.std(train_times):.3f}s")
        print(f"  Mean prediction time: {np.mean(predict_times):.3f}s ± {np.std(predict_times):.3f}s")
        print(f"  Mean prediction time per sample: {np.mean(predict_per_sample)*1000:.3f}ms ± {np.std(predict_per_sample)*1000:.3f}ms")
        print(f"  Total time for all folds: {sum(train_times) + sum(predict_times):.3f}s")
        print(f"\n{'='*100}")



# Example usage and main execution
if __name__ == "__main__":
    # Configuration for comprehensive site-based CV
    SITE_CV_CONFIG = {
        'method': 'method1',  # Change to test different methods
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
        'target_log_transform': True,
        'model_type': 'RF',
        'hyperparameter_tuning': True,
        'tuning_scope': 'quick',  # 'quick' or 'thorough'
        'random_state': 42,
        # 'output_dir': f"{MODELS_PATH}/comprehensive_site_cv_march_test",
        'output_dir': f"{MODELS_PATH}/comprehensive_site_cv",
        'heatmap_style': 'side_by_side'  # 'side_by_side', 'difference', or 'both'
    }
    
    # Data paths (using combined dataset)
    # DATA_PATHS = {
    #     'X_features': f"{DATA_PATH}/processed_data/climate_trace_9/X_features_all_facilities_mar2023.csv",
    #     'y_targets': f"{DATA_PATH}/processed_data/climate_trace_9/y_{SITE_CV_CONFIG['method']}_all_facilities_mar2023.csv"
    # }
    DATA_PATHS = {
        'X_features': f"{DATA_PATH}/processed_data/combined_large/X_features_combined_all_facilities.csv",
        'y_targets': f"{DATA_PATH}/processed_data/combined_large/y_{SITE_CV_CONFIG['method']}_combined_all_facilities.csv"
    }
    
    # Run comprehensive site-based cross-validation
    print("Starting Comprehensive Site-Based Cross-Validation...")
    print(f"Testing method: {SITE_CV_CONFIG['method']}")
    print(f"Output directory: {SITE_CV_CONFIG['output_dir']}")
    
    # Initialize and run
    site_cv = ComprehensiveSiteBasedCV(SITE_CV_CONFIG)
    
    # Load data
    X, y, all_facilities = site_cv.load_and_prepare_data(
        DATA_PATHS['X_features'], 
        DATA_PATHS['y_targets']
    )
    
    # Run comprehensive CV
    cv_results = site_cv.run_comprehensive_site_cv(X, y, all_facilities)
    
    # Create facility comparison grid
    grid_path = site_cv.create_facility_comparison_heatmap_grid()
    
    # Print detailed summary
    site_cv.print_detailed_summary()
    
    print(f"\n{'='*100}")
    print("COMPREHENSIVE SITE-BASED CROSS-VALIDATION COMPLETED!")
    print(f"{'='*100}")
    print(f"Results saved in: {SITE_CV_CONFIG['output_dir']}")
    print(f"  - Individual facility plots: ./facility_plots/")
    print(f"  - Individual heatmaps: ./heatmaps/")
    print(f"  - Comprehensive analysis: comprehensive_site_cv_analysis.png")
    print(f"  - Facility comparison grid: all_facilities_heatmap_grid.png")
    print(f"  - Detailed results: site_cv_detailed_results.csv")
    print(f"{'='*100}")