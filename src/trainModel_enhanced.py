import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GroupShuffleSplit, learning_curve
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
from pathlib import Path
import warnings
from scipy.interpolate import griddata
from utils.path_util import DATA_PATH, MODELS_PATH
warnings.filterwarnings('ignore')

class ConfigurableMLTrainer:
    """
    Configurable ML trainer for air quality prediction with multiple validation strategies
    and enhanced visualization capabilities
    """
    
    def __init__(self, config):
        """
        Initialize with configuration dictionary
        
        config = {
            'method': 'method1',  # method1, method2, method3
            'features': ['sat_at_target', 'sat_decay_1_3km', 'sat_directional_asymmetry'],
            'target_log_transform': True,
            'validation_type': 'sample_based',  # 'sample_based' or 'site_based'
            'model_type': 'RF',  # 'MLR', 'RF', 'LGBM', 'XGB', 'SVR', 'GBM'
            'test_size': 0.2,
            'random_state': 42,
            'output_dir': './results',
            
            # New configuration options
            'figure_suffix': None,  # optional suffix for figure names
            'test_facilities': None,  # optional: user-specified test facilities
            'generate_heatmaps': False,  # whether to generate spatial heatmaps
            'heatmap_facilities': 'auto',  # 'auto' uses test facilities, or specify list
            'heatmap_months': [3, 6, 9],  # which months to visualize
            'heatmap_extent_km': 50,  # spatial extent around facility
            'heatmap_style': 'side_by_side',  # 'side_by_side', 'difference', or 'both'
        }
        """
        self.config = config
        self.model = None
        self.scaler = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.feature_names = None
        
        # Create output directory
        Path(self.config['output_dir']).mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize model based on configuration"""
        model_type = self.config['model_type']
        random_state = self.config.get('random_state', 42)
        
        if model_type == 'MLR':
            self.model = LinearRegression()
            self.scaler = StandardScaler()
        elif model_type == 'RF':
            self.model = RandomForestRegressor(
                n_estimators=100,
                random_state=random_state,
                n_jobs=-1
            )
        elif model_type == 'LGBM':
            self.model = lgb.LGBMRegressor(
                n_estimators=100,
                random_state=random_state,
                verbose=-1
            )
        elif model_type == 'XGB':
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                random_state=random_state,
                verbosity=0
            )
        elif model_type == 'SVR':
            self.model = SVR(kernel='rbf')
            self.scaler = StandardScaler()
        elif model_type == 'GBM':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                random_state=random_state
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _get_base_name(self):
        """Generate base name for files with optional suffix"""
        base_name = f"{self.config['model_type']}_{self.config['method']}_{self.config['validation_type']}"
        
        if self.config.get('figure_suffix'):
            base_name += f"_{self.config['figure_suffix']}"
            
        return base_name
    
    def load_and_prepare_data(self, X_features_path, y_targets_path):
        """Load and prepare data based on configuration"""
        # Load data
        X_features = pd.read_csv(X_features_path)
        y_targets = pd.read_csv(y_targets_path)
        
        print(f"Loaded data: {len(X_features)} samples")
        print(f"Available facilities: {sorted(X_features['facility_id'].unique())}")
        
        # Select target method
        method = self.config['method']
        if f'y_{method}' in locals():
            # If separate target files
            y_data = pd.read_csv(y_targets_path.replace('y_target', f'y_target_{method}'))
        else:
            # If targets are in the same file with method columns
            y_data = y_targets
        
        # Select features
        feature_cols = self.config['features'].copy()
        
        # Add essential columns if not present
        essential_cols = ['month', 'grid_i', 'grid_j', 'facility_id']
        for col in essential_cols:
            if col not in feature_cols and col in X_features.columns:
                feature_cols.append(col)
        
        # Filter features
        available_features = [col for col in feature_cols if col in X_features.columns]
        missing_features = [col for col in self.config['features'] if col not in X_features.columns]
        
        if missing_features:
            print(f"Warning: Missing features: {missing_features}")
        
        print(f"Using features: {available_features}")
        
        # Prepare X and y
        X = X_features[available_features].copy()
        y = y_data['pm25_concentration'].copy()
        
        # Remove rows with NaN values
        valid_mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[valid_mask].reset_index(drop=True)
        y = y[valid_mask].reset_index(drop=True)
        
        print(f"After cleaning: {len(X)} samples")
        
        # Log transform target if specified
        if self.config['target_log_transform']:
            # Add small constant to avoid log(0)
            y_min = y[y > 0].min() if (y > 0).any() else 1e-6
            y = np.log(y + y_min * 0.01)
            print("Applied log transform to target")
        
        # Store feature names (exclude non-predictive columns)
        exclude_cols = ['month', 'grid_i', 'grid_j', 'facility_id']
        self.feature_names = [col for col in X.columns if col not in exclude_cols]
        
        return X, y
    
    def split_data(self, X, y):
        """Split data based on validation strategy with enhanced control"""
        validation_type = self.config['validation_type']
        test_size = self.config['test_size']
        random_state = self.config['random_state']
        
        if validation_type == 'sample_based':
            # Random sample-based split
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            print(f"Sample-based split: {len(self.X_train)} train, {len(self.X_test)} test")
            
        elif validation_type == 'site_based':
            # Site-based split with user control
            if 'facility_id' not in X.columns:
                raise ValueError("facility_id column required for site-based validation")
            
            facilities = sorted(X['facility_id'].unique())
            print(f"Available facilities: {facilities}")
            
            if self.config.get('test_facilities'):
                # User-specified test facilities
                test_facilities = self.config['test_facilities']
                
                # Validate that specified facilities exist
                missing_facilities = [f for f in test_facilities if f not in facilities]
                if missing_facilities:
                    raise ValueError(f"Specified test facilities not found: {missing_facilities}")
                
                test_mask = X['facility_id'].isin(test_facilities)
                train_mask = ~test_mask
                
                self.X_train = X[train_mask].reset_index(drop=True)
                self.X_test = X[test_mask].reset_index(drop=True)
                self.y_train = y[train_mask].reset_index(drop=True)
                self.y_test = y[test_mask].reset_index(drop=True)
                
                print(f"User-specified test facilities: {test_facilities}")
                
            else:
                # Auto-selection using GroupShuffleSplit
                gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
                train_idx, test_idx = next(gss.split(X, y, groups=X['facility_id']))
                
                self.X_train = X.iloc[train_idx].reset_index(drop=True)
                self.X_test = X.iloc[test_idx].reset_index(drop=True)
                self.y_train = y.iloc[train_idx].reset_index(drop=True)
                self.y_test = y.iloc[test_idx].reset_index(drop=True)
                
                test_facilities = sorted(self.X_test['facility_id'].unique())
                print(f"Auto-selected test facilities: {test_facilities}")
            
            train_facilities = sorted(self.X_train['facility_id'].unique())
            test_facilities = sorted(self.X_test['facility_id'].unique())
            
            print(f"Site-based split:")
            print(f"  Train facilities: {train_facilities}")
            print(f"  Test facilities: {test_facilities}")
            print(f"  Train samples: {len(self.X_train)}, Test samples: {len(self.X_test)}")
            
            # Set default heatmap facilities if not specified
            if (self.config.get('generate_heatmaps', False) and 
                self.config.get('heatmap_facilities') == 'auto'):
                self.config['heatmap_facilities'] = test_facilities
                print(f"Heatmap facilities set to: {test_facilities}")
        
        else:
            raise ValueError(f"Unknown validation type: {validation_type}")
    
    def train_model(self):
        """Train the model"""
        # Prepare training data (exclude non-predictive columns)
        X_train_features = self.X_train[self.feature_names]
        X_test_features = self.X_test[self.feature_names]
        
        # Scale features if needed
        if self.scaler is not None:
            X_train_features = self.scaler.fit_transform(X_train_features)
            X_test_features = self.scaler.transform(X_test_features)
        
        # Train model
        print(f"Training {self.config['model_type']} model...")
        self.model.fit(X_train_features, self.y_train)
        
        # Make predictions
        self.y_pred = self.model.predict(X_test_features)
        
        # Calculate metrics
        r2 = r2_score(self.y_test, self.y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, self.y_pred))
        mae = mean_absolute_error(self.y_test, self.y_pred)
        
        self.metrics = {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'n_train': len(self.y_train),
            'n_test': len(self.y_test)
        }
        
        print(f"Model performance:")
        print(f"  R² = {r2:.3f}")
        print(f"  RMSE = {rmse:.3f}")
        print(f"  MAE = {mae:.3f}")
        
        return self.metrics
    
    def create_plots(self):
        """Create separate plots and save individually"""
        output_dir = Path(self.config['output_dir'])
        base_name = self._get_base_name()
        
        plot_paths = []
        
        # Plot 1: Scatter plot
        fig1, ax1 = plt.subplots(1, 1, figsize=(8, 6))
        self._plot_scatter_with_intensity(ax1)
        scatter_path = output_dir / f"{base_name}_scatter.png"
        plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths.append(scatter_path)
        
        # Plot 2: Feature importance  
        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))
        self._plot_feature_importance(ax2)
        importance_path = output_dir / f"{base_name}_importance.png"
        plt.savefig(importance_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths.append(importance_path)
        
        # Plot 3: Learning curve
        fig3, ax3 = plt.subplots(1, 1, figsize=(8, 6))
        self._plot_learning_curve(ax3)
        learning_path = output_dir / f"{base_name}_learning.png"
        plt.savefig(learning_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths.append(learning_path)
        
        # Plot 4: Heatmaps (if enabled and site-based validation)
        if (self.config.get('generate_heatmaps', False) and 
            self.config['validation_type'] == 'site_based'):
            heatmap_paths = self._create_heatmaps()
            plot_paths.extend(heatmap_paths)
        
        print(f"Plots saved:")
        for path in plot_paths:
            print(f"  {path.name}")
        
        return plot_paths
    
    def _create_heatmaps(self):
        """Create spatial heatmaps for specified facilities and months"""
        output_dir = Path(self.config['output_dir'])
        base_name = self._get_base_name()
        heatmap_paths = []
        
        # Get facilities to visualize
        heatmap_facilities = self.config.get('heatmap_facilities', 'auto')
        if heatmap_facilities == 'auto':
            facilities_to_plot = sorted(self.X_test['facility_id'].unique())
        else:
            facilities_to_plot = heatmap_facilities
        
        months_to_plot = self.config.get('heatmap_months', [3, 6, 9])
        extent_km = self.config.get('heatmap_extent_km', 50)
        style = self.config.get('heatmap_style', 'side_by_side')
        
        print(f"Generating heatmaps for facilities: {facilities_to_plot}")
        print(f"Months: {months_to_plot}, Style: {style}")
        
        for facility_id in facilities_to_plot:
            for month in months_to_plot:
                try:
                    heatmap_path = self._create_single_heatmap(
                        facility_id, month, extent_km, style, base_name, output_dir
                    )
                    if heatmap_path:
                        heatmap_paths.append(heatmap_path)
                except Exception as e:
                    print(f"Warning: Failed to create heatmap for {facility_id}, month {month}: {e}")
        
        return heatmap_paths
    
    def _create_single_heatmap(self, facility_id, month, extent_km, style, base_name, output_dir):
        """Create a single heatmap for specified facility and month"""
        # Filter data for this facility and month
        facility_mask = (self.X_test['facility_id'] == facility_id) & (self.X_test['month'] == month)
        
        if not facility_mask.any():
            print(f"No data found for facility {facility_id}, month {month}")
            return None
        
        # Get facility data
        facility_data = self.X_test[facility_mask].copy()
        facility_measured = self.y_test[facility_mask].copy()
        facility_predicted = self.y_pred[facility_mask]
        
        if len(facility_data) < 3:
            print(f"Insufficient data points ({len(facility_data)}) for facility {facility_id}, month {month}")
            return None
        
        # Get grid coordinates
        grid_i = facility_data['grid_i'].values
        grid_j = facility_data['grid_j'].values
        
        # Create figure based on style
        if style == 'side_by_side':
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            axes = [ax1, ax2]
            titles = ['Measured PM₂.₅', 'Predicted PM₂.₅']
            data_arrays = [facility_measured.values, facility_predicted]
        elif style == 'difference':
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            axes = [ax]
            titles = ['Prediction Error (Predicted - Measured)']
            data_arrays = [facility_predicted - facility_measured.values]
        elif style == 'both':
            fig, ((ax1, ax2), (ax3, ax_empty)) = plt.subplots(2, 2, figsize=(15, 12))
            axes = [ax1, ax2, ax3]
            titles = ['Measured PM₂.₅', 'Predicted PM₂.₅', 'Prediction Error']
            data_arrays = [facility_measured.values, facility_predicted, facility_predicted - facility_measured.values]
            ax_empty.axis('off')  # Hide the fourth subplot
        
        # Create heatmaps
        for ax, title, data in zip(axes, titles, data_arrays):
            # Create scatter plot with color mapping
            if 'Error' in title:
                # Use diverging colormap for error
                vmax = max(abs(data.min()), abs(data.max()))
                scatter = ax.scatter(grid_i, grid_j, c=data, cmap='RdBu_r', 
                                   vmin=-vmax, vmax=vmax, s=60, alpha=0.8)
            else:
                # Use sequential colormap for concentrations
                scatter = ax.scatter(grid_i, grid_j, c=data, cmap='turbo', 
                                   s=60, alpha=0.8)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            unit = ' (log)' if self.config['target_log_transform'] else ' (μg/m³)'
            cbar.set_label(f'PM₂.₅{unit}' if 'Error' not in title else f'Error{unit}')
            
            # Set labels and title
            ax.set_xlabel('Grid I')
            ax.set_ylabel('Grid J')
            ax.set_title(f'{title}\n{facility_id}, Month {month}')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
        
        # Add overall title with metrics for this facility/month subset
        if len(facility_measured) > 1:
            r2_subset = r2_score(facility_measured, facility_predicted)
            rmse_subset = np.sqrt(mean_squared_error(facility_measured, facility_predicted))
            fig.suptitle(f'Spatial Distribution - {facility_id} (Month {month})\n'
                        f'R² = {r2_subset:.3f}, RMSE = {rmse_subset:.2f}, N = {len(facility_data)}',
                        fontsize=14)
        
        # Save figure
        heatmap_filename = f"{base_name}_heatmap_{facility_id}_month{month}_{style}.png"
        heatmap_path = output_dir / heatmap_filename
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return heatmap_path
    
    def _plot_scatter_with_intensity(self, ax):
        """Create scatter plot with intensity coloring like your reference"""
        # Calculate point density for coloring
        from scipy.stats import gaussian_kde
        
        # Create 2D density
        try:
            xy = np.vstack([self.y_test, self.y_pred])
            density = gaussian_kde(xy)(xy)
        except:
            # Fallback if density calculation fails
            density = np.ones(len(self.y_test))
        
        # Create scatter plot with density colors
        scatter = ax.scatter(self.y_test, self.y_pred, c=density, 
                           cmap='viridis', alpha=0.7, s=20)
        
        # Add 1:1 line
        min_val = min(self.y_test.min(), self.y_pred.min())
        max_val = max(self.y_test.max(), self.y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
        
        # Set labels and title
        target_label = "Measured PM₂.₅"
        pred_label = "Estimated PM₂.₅"
        
        if self.config['target_log_transform']:
            target_label += " (log)"
            pred_label += " (log)"
        
        ax.set_xlabel(target_label, fontsize=12)
        ax.set_ylabel(pred_label, fontsize=12)
        
        # Add metrics text
        r2 = self.metrics['r2']
        rmse = self.metrics['rmse']
        n = self.metrics['n_test']
        
        ax.text(0.05, 0.95, f'{self.config["model_type"]}\n'
                           f'R² = {r2:.3f}\n'
                           f'RMSE = {rmse:.2f}\n'
                           f'N = {n}', 
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Point Density', fontsize=10)
        
        # Set title
        ax.set_title(f'{self.config["validation_type"].replace("_", "-").title()} CV', fontsize=14)
    
    def _plot_feature_importance(self, ax):
        """Plot top 10 feature importance"""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_)
        else:
            ax.text(0.5, 0.5, 'Feature importance\nnot available\nfor this model', 
                ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Feature Importance')
            return
        
        # Sort features by importance and take top 10
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(10)
        
        # Sort for plotting (ascending for horizontal bars)
        feature_importance = feature_importance.sort_values('importance', ascending=True)
        
        # Plot horizontal bar chart
        bars = ax.barh(feature_importance['feature'], feature_importance['importance'])
        
        # Color bars (gradient or highlight top)
        bars[-1].set_color('#d62728')  # Highlight most important in red
        bars[-2].set_color('#ff7f0e')  # Second most important in orange
        
        ax.set_xlabel('Feature Importance')
        ax.set_title('Top 10 Feature Importance')
        ax.grid(axis='x', alpha=0.3)
        
        # Improve readability
        ax.tick_params(axis='y', labelsize=10)
        plt.tight_layout()
    
    def _plot_learning_curve(self, ax):
        """Plot learning curve"""
        # Prepare data for learning curve
        X_features = pd.concat([self.X_train, self.X_test])[self.feature_names]
        y_all = pd.concat([self.y_train, self.y_test])
        
        if self.scaler is not None:
            X_features = self.scaler.fit_transform(X_features)
        
        # Calculate learning curve
        train_sizes = np.linspace(0.1, 1.0, 10)
        
        try:
            train_sizes_abs, train_scores, val_scores = learning_curve(
                self.model, X_features, y_all, 
                train_sizes=train_sizes, cv=5, 
                scoring='r2', n_jobs=-1
            )
            
            # Calculate means and stds
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)
            
            # Plot learning curves
            ax.plot(train_sizes_abs, train_mean, 'o-', label='Training R²', color='blue')
            ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
            
            ax.plot(train_sizes_abs, val_mean, 'o-', label='Validation R²', color='red')
            ax.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
            
            ax.set_xlabel('Training Set Size')
            ax.set_ylabel('R² Score')
            ax.set_title('Learning Curve')
            ax.legend()
            ax.grid(alpha=0.3)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Learning curve\ncalculation failed:\n{str(e)[:50]}...', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Learning Curve')
    
    def save_results(self):
        """Save results to CSV"""
        base_name = self._get_base_name()
        
        # Prepare results dataframe
        results_df = pd.DataFrame({
            'measured': self.y_test,
            'predicted': self.y_pred,
            'facility_id': self.X_test['facility_id'] if 'facility_id' in self.X_test.columns else None,
            'month': self.X_test['month'] if 'month' in self.X_test.columns else None,
            'grid_i': self.X_test['grid_i'] if 'grid_i' in self.X_test.columns else None,
            'grid_j': self.X_test['grid_j'] if 'grid_j' in self.X_test.columns else None,
        })
        
        # Add configuration info
        config_info = {
            'model_type': self.config['model_type'],
            'method': self.config['method'],
            'validation_type': self.config['validation_type'],
            'target_log_transform': self.config['target_log_transform'],
            'features_used': ','.join(self.feature_names),
            'figure_suffix': self.config.get('figure_suffix', ''),
            'test_facilities': ','.join(map(str, self.config.get('test_facilities', []))),
            'generate_heatmaps': self.config.get('generate_heatmaps', False),
            'r2_score': self.metrics['r2'],
            'rmse': self.metrics['rmse'],
            'mae': self.metrics['mae'],
            'n_train': self.metrics['n_train'],
            'n_test': self.metrics['n_test']
        }
        
        # Save predictions
        pred_path = Path(self.config['output_dir']) / f"predictions_{base_name}.csv"
        results_df.to_csv(pred_path, index=False)
        
        # Save configuration and metrics
        config_path = Path(self.config['output_dir']) / f"config_{base_name}.csv"
        pd.DataFrame([config_info]).to_csv(config_path, index=False)
        
        print(f"Results saved to {pred_path}")
        print(f"Configuration saved to {config_path}")
        
        return pred_path, config_path


# Example usage with enhanced configuration
if __name__ == "__main__":
    # Features configuration
    FEATURES = [
        # Satellite features (your new ones)
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
    ]
    
    # Enhanced configuration examples
    CONFIGS = [
        {
            'method': 'method2',
            'features': FEATURES,
            'target_log_transform': True,
            'validation_type': 'site_based',
            'model_type': 'RF',
            'test_size': 0.2,
            'random_state': 42,
            'output_dir': f"{MODELS_PATH}/results",
            
            # New enhanced options
            'figure_suffix': 'march_test',
            'test_facilities': None,  # Let it auto-select, or specify like ['facility_A', 'facility_C']
            'generate_heatmaps': True,
            'heatmap_facilities': 'auto',  # Will use test facilities
            'heatmap_months': [3],
            'heatmap_extent_km': 50,
            'heatmap_style': 'side_by_side',  # or 'difference' or 'both'
        },
        
        # Example with user-specified test facilities and different heatmap style
        {
            'method': 'method3',
            'features': FEATURES,
            'target_log_transform': True,
            'validation_type': 'site_based',
            'model_type': 'RF',
            'test_size': 0.2,
            'random_state': 42,
            'output_dir': f"{MODELS_PATH}/results",
            
            'figure_suffix': 'march_test',
            'test_facilities': None,  # Specify exact facilities
            'generate_heatmaps': True,
            'heatmap_facilities': "auto",  # Only generate heatmaps for facility_A
            'heatmap_months': [3],  # Only June
            'heatmap_extent_km': 50,
            'heatmap_style': 'side_by_side',  # Show measured, predicted, and difference
        }
    ]
    
    # Data paths
    X_FEATURES_PATH = f"{DATA_PATH}/processed_data/climate_trace_9/X_features_all_facilities_mar2023.csv"
    
    # Run batch processing
    for i, config in enumerate(CONFIGS):
        print(f"\n{'='*80}")
        print(f"Running Configuration {i+1}/{len(CONFIGS)}")
        print(f"Model: {config['model_type']} - Method: {config['method']} - Validation: {config['validation_type']}")
        print(f"Suffix: {config.get('figure_suffix', 'None')}")
        print(f"Heatmaps: {config.get('generate_heatmaps', False)}")
        print(f"{'='*80}")
        
        Y_TARGETS_PATH = f"{DATA_PATH}/processed_data/climate_trace_9/y_{config['method']}_all_facilities_mar2023.csv"
        
        try:
            # Initialize trainer
            trainer = ConfigurableMLTrainer(config)
            
            # Load and prepare data
            X, y = trainer.load_and_prepare_data(X_FEATURES_PATH, Y_TARGETS_PATH)
            
            # Split data (will show which facilities are selected)
            trainer.split_data(X, y)
            
            # Train model
            metrics = trainer.train_model()
            
            # Create plots (including heatmaps if enabled)
            plot_paths = trainer.create_plots()
            
            # Save results
            pred_path, config_path = trainer.save_results()
            
            print(f"✓ Configuration {i+1} completed successfully")
            print(f"  Generated {len(plot_paths)} plot files")
            
        except Exception as e:
            print(f"✗ Configuration {i+1} failed: {e}")
            import traceback
            traceback.print_exc()
            continue