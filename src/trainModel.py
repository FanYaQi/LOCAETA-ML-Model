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
from utils.path_util import DATA_PATH, MODELS_PATH
warnings.filterwarnings('ignore')

class ConfigurableMLTrainer:
    """
    Configurable ML trainer for air quality prediction with multiple validation strategies
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
            'output_dir': './results'
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
    
    def load_and_prepare_data(self, X_features_path, y_targets_path):
        """Load and prepare data based on configuration"""
        # Load data
        X_features = pd.read_csv(X_features_path)
        y_targets = pd.read_csv(y_targets_path)
        
        print(f"Loaded data: {len(X_features)} samples")
        print(f"Available facilities: {X_features['facility_id'].unique()}")
        
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
        """Split data based on validation strategy"""
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
            # Site-based split (group by facility_id)
            if 'facility_id' not in X.columns:
                raise ValueError("facility_id column required for site-based validation")
            
            facilities = X['facility_id'].unique()
            n_test_facilities = max(1, int(len(facilities) * test_size))
            
            # Use GroupShuffleSplit to ensure facilities don't overlap
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            train_idx, test_idx = next(gss.split(X, y, groups=X['facility_id']))
            
            self.X_train = X.iloc[train_idx].reset_index(drop=True)
            self.X_test = X.iloc[test_idx].reset_index(drop=True)
            self.y_train = y.iloc[train_idx].reset_index(drop=True)
            self.y_test = y.iloc[test_idx].reset_index(drop=True)
            
            train_facilities = self.X_train['facility_id'].unique()
            test_facilities = self.X_test['facility_id'].unique()
            
            print(f"Site-based split:")
            print(f"  Train facilities: {train_facilities}")
            print(f"  Test facilities: {test_facilities}")
            print(f"  Train samples: {len(self.X_train)}, Test samples: {len(self.X_test)}")
        
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
        base_name = f"{self.config['model_type']}_{self.config['method']}_{self.config['validation_type']}"
        
        # Plot 1: Scatter plot
        fig1, ax1 = plt.subplots(1, 1, figsize=(8, 6))
        self._plot_scatter_with_intensity(ax1)
        scatter_path = output_dir / f"{base_name}_scatter.png"
        plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Feature importance  
        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))
        self._plot_feature_importance(ax2)
        importance_path = output_dir / f"{base_name}_importance.png"
        plt.savefig(importance_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 3: Learning curve
        fig3, ax3 = plt.subplots(1, 1, figsize=(8, 6))
        self._plot_learning_curve(ax3)
        learning_path = output_dir / f"{base_name}_learning.png"
        plt.savefig(learning_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plots saved:")
        print(f"  Scatter: {scatter_path}")
        print(f"  Importance: {importance_path}")
        print(f"  Learning: {learning_path}")
        
        return [scatter_path, importance_path, learning_path]
    
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
        # Prepare results dataframe
        results_df = pd.DataFrame({
            'measured': self.y_test,
            'predicted': self.y_pred,
            'facility_id': self.X_test['facility_id'] if 'facility_id' in self.X_test.columns else None,
            'month': self.X_test['month'] if 'month' in self.X_test.columns else None
        })
        
        # Add configuration info
        config_info = {
            'model_type': self.config['model_type'],
            'method': self.config['method'],
            'validation_type': self.config['validation_type'],
            'target_log_transform': self.config['target_log_transform'],
            'features_used': ','.join(self.feature_names),
            'r2_score': self.metrics['r2'],
            'rmse': self.metrics['rmse'],
            'mae': self.metrics['mae'],
            'n_train': self.metrics['n_train'],
            'n_test': self.metrics['n_test']
        }
        
        # Save predictions
        pred_path = Path(self.config['output_dir']) / f"predictions_{self.config['model_type']}_{self.config['method']}_{self.config['validation_type']}.csv"
        results_df.to_csv(pred_path, index=False)
        
        # Save configuration and metrics
        config_path = Path(self.config['output_dir']) / f"config_{self.config['model_type']}_{self.config['method']}_{self.config['validation_type']}.csv"
        pd.DataFrame([config_info]).to_csv(config_path, index=False)
        
        print(f"Results saved to {pred_path}")
        print(f"Configuration saved to {config_path}")
        
        return pred_path, config_path


# Example usage and batch processing
if __name__ == "__main__":
    # Configuration options
    # ALL_FEATURES = [
    #     # Satellite features (your new ones)
    #     'sat_at_target', 'sat_decay_1_3km', 'sat_directional_asymmetry',
        
    #     # Meteorology features
    #     'dewpoint_temp_2m', 'temp_2m', 'u_wind_10m', 'v_wind_10m', 
    #     'surface_pressure', 'total_precipitation', 'wind_speed', 'wind_direction',
        
    #     # Facility features
    #     'facility_lat', 'facility_lon', 'facility_height', 'distance_to_facility', 
    #     'bearing_from_facility', 'NEI_annual_emission_t', 'monthly_emission_rate_t_per_hr',
        
    #     # Topographical features
    #     'elevation', 'elevation_diff_from_facility', 'terrain_slope', 'terrain_roughness',
        
    #     # Land cover features
    #     'landcover_dominant_class', 'landcover_diversity', 'landcover_urban_percent',
    #     'landcover_forest_percent', 'landcover_agriculture_percent',
        
    #     # Road features
    #     'road_density_total', 'road_density_interstate', 'road_density_us_highway',
    #     'distance_to_nearest_road'
    # ]
    FEATURES = [
        # Satellite features (your new ones)
        'sat_at_target', 'sat_decay_1_3km', 'sat_directional_asymmetry',
        
        # Meteorology features
        'dewpoint_temp_2m', 'temp_2m', 'u_wind_10m', 'v_wind_10m', 
        # 'surface_pressure', 'total_precipitation', 'wind_speed', 'wind_direction',
        
        # Facility features
        'facility_height', 'distance_to_facility', 
        'bearing_from_facility', 'NEI_annual_emission_t', 'monthly_emission_rate_t_per_hr',
        
        # Topographical features
        'elevation', 'elevation_diff_from_facility', 'terrain_slope', 'terrain_roughness',
        
        # Road features
        'road_density_interstate'

    ]
    CONFIGS = [
        {
            'method': 'method1',
            'features': FEATURES,
            'target_log_transform': True,
            'validation_type': 'site_based',
            'model_type': 'RF',
            'test_size': 0.2,
            'random_state': 42,
            'output_dir': f"{MODELS_PATH}/results"
        }
    ]
    
    # Data paths
    # X_FEATURES_PATH = f"{DATA_PATH}/processed_data/combined/X_features_all_facilities_2023_grid24.csv"
    X_FEATURES_PATH = f"{DATA_PATH}/processed_data/climate_trace_9/X_features_all_facilities_mar2023.csv"

    
    # Run batch processing
    for config in CONFIGS:
        print(f"\n{'='*60}")
        print(f"Running: {config['model_type']} - {config['method']} - {config['validation_type']}")
        print(f"{'='*60}")
        # Y_TARGETS_PATH = f"{DATA_PATH}/processed_data/combined/y_target_{config['method']}_all_facilities_2023_grid24.csv"
        Y_TARGETS_PATH = f"{DATA_PATH}/processed_data/climate_trace_9/y_{config['method']}_all_facilities_mar2023.csv"
        try:
            # Initialize trainer
            trainer = ConfigurableMLTrainer(config)
            
            # Load and prepare data
            X, y = trainer.load_and_prepare_data(X_FEATURES_PATH, Y_TARGETS_PATH)
            
            # Split data
            trainer.split_data(X, y)
            
            # Train model
            metrics = trainer.train_model()
            
            # Create plots
            plot_path = trainer.create_plots()
            
            # Save results
            pred_path, config_path = trainer.save_results()
            
            print(f"✓ Completed successfully")
            
        except Exception as e:
            print(f"✗ Failed: {e}")
            import traceback
            traceback.print_exc()