import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, GroupShuffleSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
from scipy.interpolate import griddata
from scipy.stats import gaussian_kde
import warnings
import sys
## add sys path for local functions
path_to_add = '/Users/yaqifan/Documents/Github/LOCAETA-ML/src'
if path_to_add not in sys.path:
    sys.path.insert(0, path_to_add)
from utils.path_util import DATA_PATH,MODELS_PATH
warnings.filterwarnings('ignore')

class TripleValidationTrainer:
    """
    Enhanced trainer with sample-based, site-based, AND temporal-based validation
    """
    
    def __init__(self, config):
        """
        Enhanced configuration with temporal validation options
        
        config = {
            'method': 'method1',
            'features': [...],
            'target_log_transform': True,
            'validation_type': 'sample_based',  # 'sample_based', 'site_based', or 'temporal_based'
            'model_type': 'RF',
            'hyperparameter_tuning': True,  # Enable 3-set approach with tuning
            'tuning_strategy': 'grid',  # 'grid' or 'random'
            'tuning_scope': 'quick',  # 'quick' or 'thorough'
            'temporal_split_strategy': 'month_based',  # 'month_based', 'season_based', 'progressive'
            'temporal_facilities': ['suncor', 'rmbc', 'bluespruce'],
            'train_months': [1,2,3,4,5,6,7,8,9],  # Jan-Sep
            'val_months': [10],  # Oct
            'test_months': [11,12],  # Nov-Dec
            'test_size': 0.2,
            'val_size': 0.2,  # For 3-set approach
            'random_state': 42,
            'output_dir': './results',
            'figure_suffix': None,
            'generate_heatmaps': False,
            'heatmap_facilities': 'auto',
            'heatmap_months': [3, 6, 9, 12],
            'heatmap_style': 'side_by_side'
        }
        """
        self.config = config
        self.model = None
        self.best_model = None  # After hyperparameter tuning
        self.scaler = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.y_pred = None
        self.feature_names = None
        self.best_params = None
        self.tuning_results = None
        
        # Create output directory
        Path(self.config['output_dir']).mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize model with hyperparameter grids"""
        model_type = self.config['model_type']
        random_state = self.config.get('random_state', 42)
        
        # Define hyperparameter grids
        self.param_grids = {
            'RF': {
                'quick': {
                    'n_estimators': [100, 200],
                    'max_depth': [None, 20],
                    'min_samples_split': [2, 5]
                },
                'thorough': {
                    'n_estimators': [100, 200, 500],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', 0.3]
                }
            },
            'LGBM': {
                'quick': {
                    'n_estimators': [100, 200],
                    'max_depth': [-1, 20],
                    'learning_rate': [0.1, 0.05]
                },
                'thorough': {
                    'n_estimators': [100, 200, 500],
                    'max_depth': [-1, 10, 20, 30],
                    'learning_rate': [0.1, 0.05, 0.01],
                    'num_leaves': [31, 50, 100],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            'XGB': {
                'quick': {
                    'n_estimators': [100, 200],
                    'max_depth': [6, 10],
                    'learning_rate': [0.1, 0.05]
                },
                'thorough': {
                    'n_estimators': [100, 200, 500],
                    'max_depth': [3, 6, 10, 15],
                    'learning_rate': [0.1, 0.05, 0.01],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                }
            }
        }
        
        # Initialize base model
        if model_type == 'MLR':
            self.model = LinearRegression()
            self.scaler = StandardScaler()
        elif model_type == 'RF':
            self.model = RandomForestRegressor(random_state=random_state, n_jobs=-1)
        elif model_type == 'LGBM':
            self.model = lgb.LGBMRegressor(random_state=random_state, verbose=-1)
        elif model_type == 'XGB':
            self.model = xgb.XGBRegressor(random_state=random_state, verbosity=0)
        elif model_type == 'SVR':
            self.model = SVR(kernel='rbf')
            self.scaler = StandardScaler()
        elif model_type == 'GBM':
            self.model = GradientBoostingRegressor(random_state=random_state)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _get_base_name(self):
        """Generate base name for files"""
        base_name = f"{self.config['model_type']}_{self.config['method']}_{self.config['validation_type']}"
        
        if self.config.get('figure_suffix'):
            base_name += f"_{self.config['figure_suffix']}"
            
        return base_name
    
    def load_and_prepare_data(self, X_features_path, y_targets_path):
        """Load and prepare data"""
        # Load data
        X_features = pd.read_csv(X_features_path)
        y_targets = pd.read_csv(y_targets_path)
        
        print(f"Loaded data: {len(X_features)} samples")
        print(f"Available facilities: {sorted(X_features['facility_id'].unique())}")
        print(f"Available months: {sorted(X_features['month'].unique())}")
        
        # Select features
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
        
        # Store feature names
        exclude_cols = ['month', 'grid_i', 'grid_j', 'facility_id']
        self.feature_names = [col for col in X.columns if col not in exclude_cols]
        
        return X, y
    
    def split_data(self, X, y):
        """Enhanced split with three-set approach for all validation types"""
        validation_type = self.config['validation_type']
        test_size = self.config['test_size']
        val_size = self.config.get('val_size', 0.2)
        random_state = self.config['random_state']
        
        print(f"\nSplitting data using {validation_type} validation with 3-set approach...")
        
        if validation_type == 'sample_based':
            self._split_sample_based(X, y, test_size, val_size, random_state)
            
        elif validation_type == 'site_based':
            self._split_site_based(X, y, test_size, val_size, random_state)
            
        elif validation_type == 'temporal_based':
            self._split_temporal_based(X, y)
            
        else:
            raise ValueError(f"Unknown validation type: {validation_type}")
        
        print(f"Final split sizes:")
        print(f"  Train: {len(self.X_train)} samples ({len(self.X_train)/len(X)*100:.1f}%)")
        print(f"  Validation: {len(self.X_val)} samples ({len(self.X_val)/len(X)*100:.1f}%)")
        print(f"  Test: {len(self.X_test)} samples ({len(self.X_test)/len(X)*100:.1f}%)")
    
    def _split_sample_based(self, X, y, test_size, val_size, random_state):
        """Sample-based 3-set split"""
        # First split: train+val vs test
        X_trainval, self.X_test, y_trainval, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)  # Adjust for reduced size
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_trainval, y_trainval, test_size=val_size_adjusted, random_state=random_state
        )
        
        print(f"Sample-based 3-set split completed")
    
    def _split_site_based(self, X, y, test_size, val_size, random_state):
        """Site-based 3-set split"""
        if 'facility_id' not in X.columns:
            raise ValueError("facility_id column required for site-based validation")
        
        facilities = sorted(X['facility_id'].unique())
        n_facilities = len(facilities)
        
        print(f"Available facilities: {facilities}")
        
        # Calculate facility splits
        n_test = max(1, int(n_facilities * test_size))
        n_val = max(1, int(n_facilities * val_size))
        n_train = n_facilities - n_test - n_val
        
        if n_train < 1:
            raise ValueError(f"Not enough facilities for 3-set split. Need at least 3 facilities, have {n_facilities}")
        
        # Randomly assign facilities to sets
        np.random.seed(random_state)
        shuffled_facilities = np.random.permutation(facilities)
        
        train_facilities = shuffled_facilities[:n_train]
        val_facilities = shuffled_facilities[n_train:n_train+n_val]
        test_facilities = shuffled_facilities[n_train+n_val:]
        
        # Create masks
        train_mask = X['facility_id'].isin(train_facilities)
        val_mask = X['facility_id'].isin(val_facilities)
        test_mask = X['facility_id'].isin(test_facilities)
        
        # Split data
        self.X_train = X[train_mask].reset_index(drop=True)
        self.X_val = X[val_mask].reset_index(drop=True)
        self.X_test = X[test_mask].reset_index(drop=True)
        self.y_train = y[train_mask].reset_index(drop=True)
        self.y_val = y[val_mask].reset_index(drop=True)
        self.y_test = y[test_mask].reset_index(drop=True)
        
        print(f"Site-based 3-set split:")
        print(f"  Train facilities: {sorted(train_facilities)}")
        print(f"  Validation facilities: {sorted(val_facilities)}")
        print(f"  Test facilities: {sorted(test_facilities)}")
    
    def _split_temporal_based(self, X, y):
        """Temporal-based 3-set split"""
        temporal_facilities = self.config.get('temporal_facilities', ['suncor', 'rmbc', 'bluespruce'])
        train_months = self.config.get('train_months', [1,2,3,4,5,6,7,8,9])
        val_months = self.config.get('val_months', [10])
        test_months = self.config.get('test_months', [11,12])
        
        if 'month' not in X.columns:
            raise ValueError("month column required for temporal validation")
        
        # Filter to only temporal facilities
        temporal_mask = X['facility_id'].isin(temporal_facilities)
        if not temporal_mask.any():
            raise ValueError(f"No data found for temporal facilities: {temporal_facilities}")
        
        X_temporal = X[temporal_mask].copy()
        y_temporal = y[temporal_mask].copy()
        
        print(f"Temporal facilities: {temporal_facilities}")
        print(f"Train months: {train_months}")
        print(f"Validation months: {val_months}")
        print(f"Test months: {test_months}")
        
        # Create temporal masks
        train_mask = X_temporal['month'].isin(train_months)
        val_mask = X_temporal['month'].isin(val_months)
        test_mask = X_temporal['month'].isin(test_months)
        
        # Check if we have data for all splits
        if not train_mask.any():
            raise ValueError(f"No training data found for months: {train_months}")
        if not val_mask.any():
            raise ValueError(f"No validation data found for months: {val_months}")
        if not test_mask.any():
            raise ValueError(f"No test data found for months: {test_months}")
        
        # Split data
        self.X_train = X_temporal[train_mask].reset_index(drop=True)
        self.X_val = X_temporal[val_mask].reset_index(drop=True)
        self.X_test = X_temporal[test_mask].reset_index(drop=True)
        self.y_train = y_temporal[train_mask].reset_index(drop=True)
        self.y_val = y_temporal[val_mask].reset_index(drop=True)
        self.y_test = y_temporal[test_mask].reset_index(drop=True)
        
        print(f"Temporal 3-set split completed")
        
        # Show month distribution
        for split_name, X_split in [('Train', self.X_train), ('Val', self.X_val), ('Test', self.X_test)]:
            month_counts = X_split['month'].value_counts().sort_index()
            print(f"  {split_name} month distribution: {dict(month_counts)}")
    
    def train_model(self):
        """Train model with optional hyperparameter tuning"""
        # Prepare feature data
        X_train_features = self.X_train[self.feature_names]
        X_val_features = self.X_val[self.feature_names]
        X_test_features = self.X_test[self.feature_names]
        
        # Scale features if needed
        if self.scaler is not None:
            X_train_features = self.scaler.fit_transform(X_train_features)
            X_val_features = self.scaler.transform(X_val_features)
            X_test_features = self.scaler.transform(X_test_features)
        
        # Hyperparameter tuning
        if self.config.get('hyperparameter_tuning', False) and self.config['model_type'] in self.param_grids:
            print(f"Performing hyperparameter tuning for {self.config['model_type']}...")
            self._perform_hyperparameter_tuning(X_train_features, self.y_train, X_val_features, self.y_val)
        else:
            print(f"Training {self.config['model_type']} with default parameters...")
            self.model.fit(X_train_features, self.y_train)
            self.best_model = self.model
        
        # Make predictions on test set
        self.y_pred = self.best_model.predict(X_test_features)
        
        # Calculate metrics
        r2 = r2_score(self.y_test, self.y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, self.y_pred))
        mae = mean_absolute_error(self.y_test, self.y_pred)
        
        self.metrics = {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'n_train': len(self.y_train),
            'n_val': len(self.y_val),
            'n_test': len(self.y_test)
        }
        
        print(f"Model performance:")
        print(f"  R² = {r2:.3f}")
        print(f"  RMSE = {rmse:.3f}")
        print(f"  MAE = {mae:.3f}")
        
        return self.metrics
    
    def _perform_hyperparameter_tuning(self, X_train, y_train, X_val, y_val):
        """Perform hyperparameter tuning using validation set"""
        model_type = self.config['model_type']
        tuning_scope = self.config.get('tuning_scope', 'quick')
        
        param_grid = self.param_grids[model_type][tuning_scope]
        
        print(f"  Tuning scope: {tuning_scope}")
        print(f"  Parameter grid size: ~{np.prod([len(v) for v in param_grid.values()])} combinations")
        
        # Manual grid search using validation set
        best_score = -np.inf
        best_params = None
        tuning_results = []
        
        # Generate all parameter combinations
        from itertools import product
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        for param_combination in product(*param_values):
            params = dict(zip(param_names, param_combination))
            
            try:
                # Create and train model with these parameters
                if model_type == 'RF':
                    model = RandomForestRegressor(**params, random_state=self.config['random_state'], n_jobs=-1)
                elif model_type == 'LGBM':
                    model = lgb.LGBMRegressor(**params, random_state=self.config['random_state'], verbose=-1)
                elif model_type == 'XGB':
                    model = xgb.XGBRegressor(**params, random_state=self.config['random_state'], verbosity=0)
                
                model.fit(X_train, y_train)
                
                # Evaluate on validation set
                y_val_pred = model.predict(X_val)
                val_score = r2_score(y_val, y_val_pred)
                
                # Store results
                tuning_results.append({
                    'params': params.copy(),
                    'val_r2': val_score
                })
                
                # Update best
                if val_score > best_score:
                    best_score = val_score
                    best_params = params.copy()
                    self.best_model = model
                
            except Exception as e:
                print(f"    Failed parameter combination {params}: {e}")
                continue
        
        self.best_params = best_params
        self.tuning_results = tuning_results
        
        print(f"  Best validation R²: {best_score:.3f}")
        print(f"  Best parameters: {best_params}")
    
    def create_plots(self):
        """Create enhanced plots including temporal heatmaps if applicable"""
        output_dir = Path(self.config['output_dir'])
        base_name = self._get_base_name()
        plot_paths = []
        
        # Standard plots
        plot_paths.extend(self._create_standard_plots(output_dir, base_name))
        
        # Temporal-specific plots
        if self.config['validation_type'] == 'temporal_based':
            plot_paths.extend(self._create_temporal_plots(output_dir, base_name))
        
        # Site-specific plots
        if (self.config['validation_type'] == 'site_based' and 
            self.config.get('generate_heatmaps', False)):
            plot_paths.extend(self._create_site_heatmaps(output_dir, base_name))
        
        print(f"Generated {len(plot_paths)} plots")
        return plot_paths
    
    def _create_standard_plots(self, output_dir, base_name):
        """Create standard scatter, importance, and learning plots"""
        plot_paths = []
        
        # Scatter plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        self._plot_scatter_with_intensity(ax)
        scatter_path = output_dir / f"{base_name}_scatter.png"
        plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths.append(scatter_path)
        
        # Feature importance
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        self._plot_feature_importance(ax)
        importance_path = output_dir / f"{base_name}_importance.png"
        plt.savefig(importance_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths.append(importance_path)
        
        # Hyperparameter tuning results (if available)
        if self.tuning_results:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            self._plot_hyperparameter_results(ax)
            tuning_path = output_dir / f"{base_name}_hyperparameter_tuning.png"
            plt.savefig(tuning_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths.append(tuning_path)
        
        return plot_paths
    
    def _create_temporal_plots(self, output_dir, base_name):
        """Create enhanced temporal-specific visualization plots"""
        plot_paths = []
        
        if 'month' not in self.X_test.columns:
            return plot_paths
        
        # Enhanced temporal analysis with facility integration
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Prepare test results
        test_results = pd.DataFrame({
            'month': self.X_test['month'],
            'measured': self.y_test,
            'predicted': self.y_pred,
            'facility_id': self.X_test['facility_id'],
            'grid_i': self.X_test['grid_i'],
            'grid_j': self.X_test['grid_j']
        })
        
        # Plot 1: Performance by Month (Fixed for November/December)
        monthly_r2 = test_results.groupby('month').apply(
            lambda x: r2_score(x['measured'], x['predicted']) if len(x) > 1 else np.nan
        ).dropna()
        
        month_labels = {11: 'November', 12: 'December'}
        x_positions = list(range(len(monthly_r2)))
        month_names = [month_labels.get(month, f'Month {month}') for month in monthly_r2.index]
        
        bars = axes[0,0].bar(x_positions, monthly_r2.values, color=['lightblue', 'lightcoral'])
        axes[0,0].set_xticks(x_positions)
        axes[0,0].set_xticklabels(month_names)
        axes[0,0].set_ylabel('R² Score')
        axes[0,0].set_title('Performance by Test Month')
        axes[0,0].grid(axis='y', alpha=0.3)
        
        # Add R² values on bars
        for i, (month, r2_val) in enumerate(monthly_r2.items()):
            axes[0,0].text(i, r2_val + 0.01, f'{r2_val:.3f}', 
                         ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Plot 2: Monthly predictions vs actual with facility distinction
        colors = ['blue', 'orange', 'green']
        facility_ids = sorted(test_results['facility_id'].unique())
        
        for i, facility in enumerate(facility_ids):
            facility_data = test_results[test_results['facility_id'] == facility]
            
            for month in sorted(facility_data['month'].unique()):
                month_facility_data = facility_data[facility_data['month'] == month]
                marker = 'o' if month == 11 else 's'  # Circle for Nov, square for Dec
                alpha = 0.7 if month == 11 else 0.5
                
                axes[0,1].scatter(month_facility_data['measured'], month_facility_data['predicted'], 
                                c=colors[i % len(colors)], marker=marker, alpha=alpha,
                                label=f'{facility} - {month_labels.get(month, f"Month {month}")}' if i == 0 or facility == facility_ids[0] else "")
        
        # Add 1:1 line
        min_val = min(test_results['measured'].min(), test_results['predicted'].min())
        max_val = max(test_results['measured'].max(), test_results['predicted'].max())
        axes[0,1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
        axes[0,1].set_xlabel('Measured PM₂.₅ (log)')
        axes[0,1].set_ylabel('Predicted PM₂.₅ (log)')
        axes[0,1].set_title('Temporal Predictions by Month & Facility')
        axes[0,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot 3: Integrated Facility-Month Performance Analysis
        facility_month_r2 = test_results.groupby(['facility_id', 'month']).apply(
            lambda x: r2_score(x['measured'], x['predicted']) if len(x) > 1 else np.nan
        ).reset_index()
        facility_month_r2.columns = ['facility_id', 'month', 'r2_score']
        facility_month_r2 = facility_month_r2.dropna()
        
        # Create grouped bar chart
        facilities = sorted(facility_month_r2['facility_id'].unique())
        months = sorted(facility_month_r2['month'].unique())
        
        x = np.arange(len(facilities))
        width = 0.35
        
        for i, month in enumerate(months):
            month_data = facility_month_r2[facility_month_r2['month'] == month]
            r2_values = []
            
            for facility in facilities:
                facility_r2 = month_data[month_data['facility_id'] == facility]['r2_score']
                r2_values.append(facility_r2.iloc[0] if len(facility_r2) > 0 else 0)
            
            bars = axes[1,0].bar(x + i*width, r2_values, width, 
                               label=month_labels.get(month, f'Month {month}'),
                               alpha=0.8)
            
            # Add R² values on bars
            for j, r2_val in enumerate(r2_values):
                if r2_val > 0:
                    axes[1,0].text(x[j] + i*width, r2_val + 0.005, f'{r2_val:.3f}', 
                                 ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        axes[1,0].set_xlabel('Facility')
        axes[1,0].set_ylabel('R² Score')
        axes[1,0].set_title('Performance by Facility and Month')
        axes[1,0].set_xticks(x + width/2)
        axes[1,0].set_xticklabels(facilities, rotation=45)
        axes[1,0].legend()
        axes[1,0].grid(axis='y', alpha=0.3)
        
        # Plot 4: Sample Size Distribution
        sample_counts = test_results.groupby(['facility_id', 'month']).size().reset_index()
        sample_counts.columns = ['facility_id', 'month', 'count']
        
        # Create stacked bar chart for sample sizes
        for i, month in enumerate(months):
            month_data = sample_counts[sample_counts['month'] == month]
            counts = []
            
            for facility in facilities:
                facility_count = month_data[month_data['facility_id'] == facility]['count']
                counts.append(facility_count.iloc[0] if len(facility_count) > 0 else 0)
            
            axes[1,1].bar(facilities, counts, alpha=0.7, 
                        label=month_labels.get(month, f'Month {month}'))
            
            # Add count values on bars
            for j, count in enumerate(counts):
                if count > 0:
                    axes[1,1].text(j, count + 10, str(count), 
                                 ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        axes[1,1].set_xlabel('Facility')
        axes[1,1].set_ylabel('Number of Test Samples')
        axes[1,1].set_title('Test Sample Distribution by Facility and Month')
        axes[1,1].legend()
        axes[1,1].grid(axis='y', alpha=0.3)
        plt.setp(axes[1,1].xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        temporal_path = output_dir / f"{base_name}_temporal_analysis.png"
        plt.savefig(temporal_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths.append(temporal_path)
        
        # Create heatmaps for each test month and facility
        heatmap_paths = self._create_temporal_heatmaps(output_dir, base_name, test_results)
        plot_paths.extend(heatmap_paths)
        
        return plot_paths
    
    def _create_temporal_heatmaps(self, output_dir, base_name, test_results):
        """Create spatial heatmaps for each test month and facility"""
        plot_paths = []
        
        # Create heatmaps for each facility-month combination
        for facility_id in sorted(test_results['facility_id'].unique()):
            for month in sorted(test_results['month'].unique()):
                facility_month_data = test_results[
                    (test_results['facility_id'] == facility_id) & 
                    (test_results['month'] == month)
                ]
                
                if len(facility_month_data) < 3:  # Need minimum points for heatmap
                    continue
                
                # Create side-by-side heatmap
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Get spatial data
                grid_i = facility_month_data['grid_i'].values
                grid_j = facility_month_data['grid_j'].values
                measured = facility_month_data['measured'].values
                predicted = facility_month_data['predicted'].values
                
                # Plot 1: Measured values
                scatter1 = ax1.scatter(grid_i, grid_j, c=measured, cmap='viridis', 
                                     s=60, alpha=0.8)
                ax1.set_xlabel('Grid I')
                ax1.set_ylabel('Grid J')
                ax1.set_title(f'Measured PM₂.₅\n{facility_id} - Month {month}')
                ax1.grid(True, alpha=0.3)
                ax1.set_aspect('equal')
                cbar1 = plt.colorbar(scatter1, ax=ax1)
                cbar1.set_label('PM₂.₅ (log)')
                
                # Plot 2: Predicted values
                scatter2 = ax2.scatter(grid_i, grid_j, c=predicted, cmap='viridis', 
                                     s=60, alpha=0.8)
                ax2.set_xlabel('Grid I')
                ax2.set_ylabel('Grid J')
                ax2.set_title(f'Predicted PM₂.₅\n{facility_id} - Month {month}')
                ax2.grid(True, alpha=0.3)
                ax2.set_aspect('equal')
                cbar2 = plt.colorbar(scatter2, ax=ax2)
                cbar2.set_label('PM₂.₅ (log)')
                
                # Add performance metrics
                if len(measured) > 1:
                    r2_facility_month = r2_score(measured, predicted)
                    rmse_facility_month = np.sqrt(mean_squared_error(measured, predicted))
                    
                    fig.suptitle(f'Temporal Validation Heatmap\n'
                               f'R² = {r2_facility_month:.3f}, RMSE = {rmse_facility_month:.3f}, N = {len(measured)}',
                               fontsize=14)
                
                plt.tight_layout()
                
                # Save heatmap
                month_names = {11: 'November', 12: 'December'}
                month_name = month_names.get(month, f'Month{month}')
                heatmap_filename = f"{base_name}_heatmap_{facility_id}_{month_name}.png"
                heatmap_path = output_dir / heatmap_filename
                plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
                plt.close()
                plot_paths.append(heatmap_path)
        
        return plot_paths
    
    def _create_site_heatmaps(self, output_dir, base_name):
        """Create site-based heatmaps for test facilities"""
        plot_paths = []
        
        if 'facility_id' not in self.X_test.columns:
            return plot_paths
        
        # Get test facilities
        test_facilities = sorted(self.X_test['facility_id'].unique())
        heatmap_months = self.config.get('heatmap_months', [3, 6, 9, 12])
        
        print(f"    Generating heatmaps for test facilities: {test_facilities}")
        
        for facility_id in test_facilities:
            # Get all data for this facility
            facility_data = self.X_test[self.X_test['facility_id'] == facility_id]
            facility_measured = self.y_test[self.X_test['facility_id'] == facility_id]
            facility_predicted = self.y_pred[self.X_test['facility_id'] == facility_id]
            
            if len(facility_data) < 3:
                print(f"      Skipping {facility_id}: insufficient data ({len(facility_data)} points)")
                continue
            
            # Create side-by-side heatmap for this facility
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Get spatial data
            grid_i = facility_data['grid_i'].values
            grid_j = facility_data['grid_j'].values
            
            # Plot 1: Measured values
            scatter1 = ax1.scatter(grid_i, grid_j, c=facility_measured.values, cmap='viridis', 
                                 s=60, alpha=0.8)
            ax1.set_xlabel('Grid I')
            ax1.set_ylabel('Grid J')
            ax1.set_title(f'Measured PM₂.₅\n{facility_id}')
            ax1.grid(True, alpha=0.3)
            ax1.set_aspect('equal')
            cbar1 = plt.colorbar(scatter1, ax=ax1)
            cbar1.set_label('PM₂.₅ (log)')
            
            # Plot 2: Predicted values
            scatter2 = ax2.scatter(grid_i, grid_j, c=facility_predicted, cmap='viridis', 
                                 s=60, alpha=0.8)
            ax2.set_xlabel('Grid I')
            ax2.set_ylabel('Grid J')
            ax2.set_title(f'Predicted PM₂.₅\n{facility_id}')
            ax2.grid(True, alpha=0.3)
            ax2.set_aspect('equal')
            cbar2 = plt.colorbar(scatter2, ax=ax2)
            cbar2.set_label('PM₂.₅ (log)')
            
            # Add performance metrics
            if len(facility_measured) > 1:
                r2_facility = r2_score(facility_measured, facility_predicted)
                rmse_facility = np.sqrt(mean_squared_error(facility_measured, facility_predicted))
                
                fig.suptitle(f'Site-Based Validation Heatmap\n'
                           f'R² = {r2_facility:.3f}, RMSE = {rmse_facility:.3f}, N = {len(facility_data)}',
                           fontsize=14)
            
            plt.tight_layout()
            
            # Save heatmap
            heatmap_filename = f"{base_name}_heatmap_{facility_id}.png"
            heatmap_path = output_dir / heatmap_filename
            plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths.append(heatmap_path)
            
            print(f"      Generated heatmap for {facility_id}")
        
        return plot_paths
    
    def _plot_scatter_with_intensity(self, ax):
        """Enhanced scatter plot with validation type info"""
        # Calculate point density
        try:
            xy = np.vstack([self.y_test, self.y_pred])
            density = gaussian_kde(xy)(xy)
        except:
            density = np.ones(len(self.y_test))
        
        scatter = ax.scatter(self.y_test, self.y_pred, c=density, 
                           cmap='viridis', alpha=0.7, s=20)
        
        # Add 1:1 line
        min_val = min(self.y_test.min(), self.y_pred.min())
        max_val = max(self.y_test.max(), self.y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
        
        # Labels
        target_label = "Measured PM₂.₅"
        pred_label = "Predicted PM₂.₅"
        
        if self.config['target_log_transform']:
            target_label += " (log)"
            pred_label += " (log)"
        
        ax.set_xlabel(target_label, fontsize=12)
        ax.set_ylabel(pred_label, fontsize=12)
        
        # Enhanced metrics text
        r2 = self.metrics['r2']
        rmse = self.metrics['rmse']
        n_test = self.metrics['n_test']
        
        metrics_text = f'{self.config["model_type"]}\n'
        metrics_text += f'R² = {r2:.3f}\n'
        metrics_text += f'RMSE = {rmse:.2f}\n'
        metrics_text += f'N = {n_test}'
        
        if self.best_params:
            metrics_text += '\nTuned'
        
        ax.text(0.05, 0.95, metrics_text, 
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Point Density', fontsize=10)
        
        # Enhanced title with validation type
        validation_type = self.config['validation_type'].replace('_', '-').title()
        ax.set_title(f'{validation_type} Validation', fontsize=14)
    
    def _plot_feature_importance(self, ax):
        """Plot feature importance with enhanced styling"""
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
        elif hasattr(self.best_model, 'coef_'):
            importances = np.abs(self.best_model.coef_)
        else:
            ax.text(0.5, 0.5, 'Feature importance\nnot available\nfor this model', 
                ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Feature Importance')
            return
        
        # Create importance DataFrame
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(10)
        
        # Sort for plotting (ascending for horizontal bars)
        feature_importance = feature_importance.sort_values('importance', ascending=True)
        
        # Plot horizontal bar chart
        bars = ax.barh(feature_importance['feature'], feature_importance['importance'])
        
        # Color bars
        bars[-1].set_color('#d62728')  # Most important in red
        bars[-2].set_color('#ff7f0e')  # Second most important in orange
        
        ax.set_xlabel('Feature Importance')
        title = 'Top 10 Feature Importance'
        if self.best_params:
            title += ' (Tuned Model)'
        ax.set_title(title)
        ax.grid(axis='x', alpha=0.3)
        ax.tick_params(axis='y', labelsize=10)
    
    def _plot_hyperparameter_results(self, ax):
        """Enhanced hyperparameter tuning visualization with before/after comparison"""
        if not self.tuning_results:
            ax.text(0.5, 0.5, 'No hyperparameter\ntuning performed', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Hyperparameter Tuning')
            return
        
        # Get default model performance (first in tuning results with default-like params)
        tuning_df = pd.DataFrame(self.tuning_results)
        tuning_df = tuning_df.sort_values('val_r2', ascending=False)
        
        # Estimate default performance (model with minimal parameters)
        default_params = self._get_default_params()
        default_performance = None
        
        # Find closest to default parameters
        for result in self.tuning_results:
            if self._is_close_to_default(result['params'], default_params):
                default_performance = result['val_r2']
                break
        
        # If no default found, use worst performance as approximation
        if default_performance is None:
            default_performance = tuning_df['val_r2'].min()
        
        best_performance = tuning_df['val_r2'].max()
        
        # Create before/after comparison
        categories = ['Default\nParameters', 'Tuned\nParameters']
        performances = [default_performance, best_performance]
        colors = ['lightcoral', 'lightgreen']
        
        bars = ax.bar(categories, performances, color=colors, alpha=0.8, edgecolor='black')
        
        # Add performance values on bars
        for bar, perf in zip(bars, performances):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                   f'{perf:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Add improvement text
        improvement = best_performance - default_performance
        improvement_pct = (improvement / default_performance) * 100 if default_performance > 0 else 0
        
        ax.text(0.5, 0.85, f'Improvement: +{improvement:.3f}\n({improvement_pct:+.1f}%)', 
               transform=ax.transAxes, ha='center', va='center', fontsize=11,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
        
        ax.set_ylabel('Validation R² Score')
        ax.set_title('Hyperparameter Tuning Impact')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, max(performances) * 1.1)
        
        # Print best parameters to console
        best_params = tuning_df.iloc[0]['params']
        print(f"\n    Best Hyperparameters Found:")
        for param, value in best_params.items():
            print(f"      {param}: {value}")
        print(f"    Performance improvement: {improvement:.3f} ({improvement_pct:+.1f}%)")
    
    def _get_default_params(self):
        """Get default parameters for the model type"""
        model_type = self.config['model_type']
        
        if model_type == 'RF':
            return {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': 'sqrt'
            }
        elif model_type == 'LGBM':
            return {
                'n_estimators': 100,
                'max_depth': -1,
                'learning_rate': 0.1,
                'num_leaves': 31,
                'subsample': 1.0
            }
        elif model_type == 'XGB':
            return {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 1.0,
                'colsample_bytree': 1.0
            }
        return {}
    
    def _is_close_to_default(self, params, default_params):
        """Check if parameters are close to default values"""
        if not default_params:
            return False
        
        # Check if at least 3 parameters match default values
        matches = 0
        for key, default_val in default_params.items():
            if key in params and params[key] == default_val:
                matches += 1
        
        return matches >= 3


class TripleValidationExperimentRunner:
    """
    Run comprehensive experiments across all three validation strategies
    """
    
    def __init__(self, base_config, data_paths):
        """
        Initialize triple validation experiment runner
        
        Args:
            base_config: Base configuration dictionary
            data_paths: Dictionary with 'X_features' and 'y_targets_template' paths
        """
        self.base_config = base_config
        self.data_paths = data_paths
        self.results = []
        
        # Create main results directory
        self.results_dir = Path(base_config['output_dir'])
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def run_triple_validation_experiments(self):
        """
        Run focused experiments with single model (RF) and log transform
        Testing only method comparison across validation strategies
        """
        print("="*80)
        print("STARTING FOCUSED TRIPLE VALIDATION EXPERIMENTS")
        print("RF Model Only + Log Transform + Method Comparison")
        print("="*80)
        
        all_results = []
        
        # Single experiment: Method comparison with RF + log transform
        exp_results = self.experiment_method_comparison_focused()
        all_results.extend(exp_results)
        
        # Create comprehensive analysis
        self._create_triple_validation_analysis(all_results)
        
        return all_results
    
    def experiment_method_comparison_focused(self):
        """Focused experiment: Compare methods across all validation strategies with RF + log"""
        print(f"\n{'='*80}")
        print("FOCUSED EXPERIMENT: METHOD COMPARISON (RF + Log Transform)")
        print(f"{'='*80}")
        
        methods = ['method1', 'method2', 'method3']
        validation_types = ['sample_based', 'site_based', 'temporal_based']
        
        exp_results = []
        
        for method in methods:
            for validation_type in validation_types:
                config = self.base_config.copy()
                config.update({
                    'model_type': 'RF',  # Fixed
                    'validation_type': validation_type,
                    'method': method,
                    'target_log_transform': True,  # Fixed
                    'hyperparameter_tuning': True,
                    'tuning_scope': 'thorough',
                    'figure_suffix': f'{method}_{validation_type}',
                    'output_dir': str(self.results_dir / 'method_comparison'),
                    'generate_heatmaps': True  # Generate for all validation types
                })
                
                result = self._run_single_experiment(
                    config, 
                    f"RF + Log: {method} - {validation_type}"
                )
                
                if result:
                    result['experiment'] = 'method_comparison'
                    result['variable_name'] = 'method_validation_combo'
                    result['variable_value'] = f"{method}_{validation_type}"
                    exp_results.append(result)
        
        # Analyze results
        self._analyze_triple_validation_robustness(exp_results, 'Method Comparison')
        return exp_results
    
    def experiment_1_model_comparison(self):
        """Compare models across all three validation strategies"""
        print(f"\n{'='*80}")
        print("EXPERIMENT 1: MODEL COMPARISON ACROSS ALL VALIDATION TYPES")
        print(f"{'='*80}")
        
        models = ['RF', 'LGBM', 'XGB', 'MLR']
        validation_types = ['sample_based', 'site_based', 'temporal_based']
        
        exp_results = []
        
        for model_type in models:
            for validation_type in validation_types:
                config = self.base_config.copy()
                config.update({
                    'model_type': model_type,
                    'validation_type': validation_type,
                    'method': 'method1',  # Fixed
                    'target_log_transform': True,  # Fixed based on experience
                    'hyperparameter_tuning': True,  # Enable tuning
                    'tuning_scope': 'quick',  # Start with quick tuning
                    'figure_suffix': f'exp1_{model_type.lower()}_{validation_type}',
                    'output_dir': str(self.results_dir / 'exp1_model_comparison'),
                    'generate_heatmaps': validation_type == 'site_based'
                })
                
                result = self._run_single_experiment(
                    config, 
                    f"Exp1: {model_type} - {validation_type}"
                )
                
                if result:
                    result['experiment'] = 'model_comparison'
                    result['variable_name'] = 'model_validation_combo'
                    result['variable_value'] = f"{model_type}_{validation_type}"
                    exp_results.append(result)
        
        # Analyze triple validation robustness
        self._analyze_triple_validation_robustness(exp_results, 'Model Comparison')
        return exp_results
    
    def experiment_2_log_transform(self):
        """Test log transform across all validation strategies"""
        print(f"\n{'='*80}")
        print("EXPERIMENT 2: LOG TRANSFORM ACROSS ALL VALIDATION TYPES")
        print(f"{'='*80}")
        
        log_options = [True, False]
        validation_types = ['sample_based', 'site_based', 'temporal_based']
        
        exp_results = []
        
        for log_transform in log_options:
            for validation_type in validation_types:
                config = self.base_config.copy()
                config.update({
                    'model_type': 'RF',  # Fixed - best from experience
                    'validation_type': validation_type,
                    'method': 'method1',  # Fixed
                    'target_log_transform': log_transform,
                    'hyperparameter_tuning': True,
                    'tuning_scope': 'quick',
                    'figure_suffix': f'exp2_log{str(log_transform).lower()}_{validation_type}',
                    'output_dir': str(self.results_dir / 'exp2_log_transform'),
                    'generate_heatmaps': validation_type in ['site_based', 'temporal_based']
                })
                
                result = self._run_single_experiment(
                    config, 
                    f"Exp2: Log={log_transform} - {validation_type}"
                )
                
                if result:
                    result['experiment'] = 'log_transform'
                    result['variable_name'] = 'log_validation_combo'
                    result['variable_value'] = f"{log_transform}_{validation_type}"
                    exp_results.append(result)
        
        # Analyze results
        self._analyze_triple_validation_robustness(exp_results, 'Log Transform')
        return exp_results
    
    def experiment_3_method_comparison(self):
        """Compare methods across all validation strategies"""
        print(f"\n{'='*80}")
        print("EXPERIMENT 3: METHOD COMPARISON ACROSS ALL VALIDATION TYPES")
        print(f"{'='*80}")
        
        methods = ['method1', 'method2', 'method3']
        validation_types = ['sample_based', 'site_based', 'temporal_based']
        
        exp_results = []
        
        for method in methods:
            for validation_type in validation_types:
                config = self.base_config.copy()
                config.update({
                    'model_type': 'RF',  # Fixed - best from exp 1
                    'validation_type': validation_type,
                    'method': method,
                    'target_log_transform': True,  # Fixed - best from exp 2
                    'hyperparameter_tuning': True,
                    'tuning_scope': 'thorough',  # More thorough for final experiment
                    'figure_suffix': f'exp3_{method}_{validation_type}',
                    'output_dir': str(self.results_dir / 'exp3_method_comparison'),
                    'generate_heatmaps': validation_type in ['site_based', 'temporal_based']
                })
                
                result = self._run_single_experiment(
                    config, 
                    f"Exp3: {method} - {validation_type}"
                )
                
                if result:
                    result['experiment'] = 'method_comparison'
                    result['variable_name'] = 'method_validation_combo'
                    result['variable_value'] = f"{method}_{validation_type}"
                    exp_results.append(result)
        
        # Analyze results
        self._analyze_triple_validation_robustness(exp_results, 'Method Comparison')
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
            trainer = TripleValidationTrainer(config)
            
            # Load and prepare data
            X, y = trainer.load_and_prepare_data(self.data_paths['X_features'], y_targets_path)
            
            # Split data (sample/site/temporal)
            trainer.split_data(X, y)
            
            # Train model with hyperparameter tuning
            metrics = trainer.train_model()
            
            # Create plots
            plot_paths = trainer.create_plots()
            
            # Store comprehensive result
            result = {
                'model_type': config['model_type'],
                'validation_type': config['validation_type'],
                'method': config['method'],
                'target_log_transform': config['target_log_transform'],
                'hyperparameter_tuning': config.get('hyperparameter_tuning', False),
                'tuning_scope': config.get('tuning_scope', 'none'),
                'best_params': str(trainer.best_params) if trainer.best_params else 'default',
                'r2_score': metrics['r2'],
                'rmse': metrics['rmse'],
                'mae': metrics['mae'],
                'n_train': metrics['n_train'],
                'n_val': metrics['n_val'],
                'n_test': metrics['n_test'],
                'plot_paths': [str(p) for p in plot_paths],
                'status': 'success'
            }
            
            self.results.append(result)
            print(f"  ✓ Success: R² = {metrics['r2']:.3f}, RMSE = {metrics['rmse']:.3f}")
            
            return result
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            
            failed_result = {
                'model_type': config.get('model_type', 'unknown'),
                'validation_type': config.get('validation_type', 'unknown'),
                'method': config.get('method', 'unknown'),
                'target_log_transform': config.get('target_log_transform', False),
                'hyperparameter_tuning': False,
                'tuning_scope': 'failed',
                'best_params': 'failed',
                'r2_score': np.nan,
                'rmse': np.nan,
                'mae': np.nan,
                'n_train': np.nan,
                'n_val': np.nan,
                'n_test': np.nan,
                'plot_paths': [],
                'status': 'failed',
                'error': str(e)
            }
            
            self.results.append(failed_result)
            return None
    
    def _analyze_triple_validation_robustness(self, exp_results, experiment_name):
        """Analyze robustness across all three validation strategies"""
        if not exp_results:
            print(f"No results to analyze for {experiment_name}")
            return
        
        df = pd.DataFrame(exp_results)
        successful_df = df[df['status'] == 'success'].copy()
        
        if len(successful_df) == 0:
            print(f"No successful experiments for {experiment_name}")
            return
        
        print(f"\n{'-'*60}")
        print(f"{experiment_name.upper()} TRIPLE VALIDATION ANALYSIS")
        print(f"{'-'*60}")
        
        # Extract model and validation type from variable_value
        successful_df['base_config'] = successful_df['variable_value'].apply(lambda x: '_'.join(x.split('_')[:-1]))
        successful_df['validation_only'] = successful_df['variable_value'].apply(lambda x: x.split('_')[-1])
        
        # Calculate robustness metrics across all three validation types
        robustness_analysis = []
        
        for base_config in successful_df['base_config'].unique():
            subset = successful_df[successful_df['base_config'] == base_config]
            
            if len(subset) >= 2:  # Need at least 2 validation types
                val_types = subset['validation_only'].tolist()
                r2_scores = subset['r2_score'].tolist()
                
                # Calculate statistics
                mean_r2 = np.mean(r2_scores)
                std_r2 = np.std(r2_scores)
                min_r2 = np.min(r2_scores)
                max_r2 = np.max(r2_scores)
                r2_range = max_r2 - min_r2
                cv_r2 = std_r2 / mean_r2 if mean_r2 > 0 else np.inf
                
                # Create detailed breakdown
                val_breakdown = dict(zip(val_types, r2_scores))
                
                robustness_analysis.append({
                    'config': base_config,
                    'mean_r2': mean_r2,
                    'std_r2': std_r2,
                    'min_r2': min_r2,
                    'max_r2': max_r2,
                    'r2_range': r2_range,
                    'cv_r2': cv_r2,  # Coefficient of variation (lower = more robust)
                    'n_validations': len(subset),
                    'sample_based_r2': val_breakdown.get('sample_based', np.nan),
                    'site_based_r2': val_breakdown.get('site_based', np.nan),
                    'temporal_based_r2': val_breakdown.get('temporal_based', np.nan),
                    'robustness_score': 1 / (1 + cv_r2) if cv_r2 != np.inf else 0
                })
        
        if robustness_analysis:
            robustness_df = pd.DataFrame(robustness_analysis)
            robustness_df = robustness_df.sort_values('robustness_score', ascending=False)
            
            print(f"\nTriple Validation Robustness Ranking:")
            print("(Higher robustness score = more consistent across validation strategies)")
            print(robustness_df[['config', 'mean_r2', 'r2_range', 'cv_r2', 'robustness_score', 
                               'sample_based_r2', 'site_based_r2', 'temporal_based_r2']].to_string(index=False, float_format='%.3f'))
            
            # Identify most robust configuration
            most_robust = robustness_df.iloc[0]
            print(f"\nMOST ROBUST CONFIGURATION: {most_robust['config']}")
            print(f"  Mean R²: {most_robust['mean_r2']:.3f}")
            print(f"  R² Range: {most_robust['r2_range']:.3f}")
            print(f"  Coefficient of Variation: {most_robust['cv_r2']:.3f}")
            print(f"  Sample-based R²: {most_robust['sample_based_r2']:.3f}")
            print(f"  Site-based R²: {most_robust['site_based_r2']:.3f}")
            print(f"  Temporal-based R²: {most_robust['temporal_based_r2']:.3f}")
    
    def _create_triple_validation_analysis(self, all_results):
        """Create comprehensive analysis across all experiments and validation types"""
        if not all_results:
            print("No results for comprehensive analysis")
            return
        
        results_df = pd.DataFrame(all_results)
        successful_df = results_df[results_df['status'] == 'success'].copy()
        
        if len(successful_df) == 0:
            print("No successful results for comprehensive analysis")
            return
        
        # Create comprehensive comparison plots
        fig, axes = plt.subplots(3, 2, figsize=(18, 18))
        
        # Plot 1: R² comparison across validation types by experiment
        experiments = ['model_comparison', 'log_transform', 'method_comparison']
        validation_types = ['sample_based', 'site_based', 'temporal_based']
        colors = ['skyblue', 'lightcoral', 'lightgreen']
        
        for i, experiment in enumerate(experiments):
            exp_data = successful_df[successful_df['experiment'] == experiment]
            
            if len(exp_data) > 0:
                # Group by validation type
                validation_means = []
                validation_labels = []
                
                for val_type in validation_types:
                    val_data = exp_data[exp_data['validation_type'] == val_type]
                    if len(val_data) > 0:
                        validation_means.append(val_data['r2_score'].values)
                        validation_labels.append(val_type.replace('_', '-').title())
                
                # Create box plot
                if validation_means:
                    bp = axes[i, 0].boxplot(validation_means, labels=validation_labels, 
                                          patch_artist=True)
                    
                    # Color the boxes
                    for patch, color in zip(bp['boxes'], colors[:len(validation_means)]):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)
                    
                    axes[i, 0].set_ylabel('R² Score')
                    axes[i, 0].set_title(f'{experiment.replace("_", " ").title()}\nValidation Strategy Comparison')
                    axes[i, 0].grid(axis='y', alpha=0.3)
        
        # Plot 2: Overall validation strategy performance
        overall_performance = successful_df.groupby('validation_type')['r2_score'].agg(['mean', 'std', 'count'])
        
        bars = axes[0, 1].bar(overall_performance.index, overall_performance['mean'], 
                            yerr=overall_performance['std'], capsize=5, 
                            color=colors, alpha=0.7)
        
        # Add mean values on bars
        for bar, mean_val in zip(bars, overall_performance['mean']):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                          f'{mean_val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        axes[0, 1].set_ylabel('Mean R² Score')
        axes[0, 1].set_title('Overall Performance by Validation Strategy')
        axes[0, 1].set_xticklabels([x.replace('_', '-').title() for x in overall_performance.index])
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # Plot 3: Model performance consistency
        model_consistency = successful_df.groupby('model_type').apply(
            lambda x: x.groupby('validation_type')['r2_score'].mean()
        ).unstack(fill_value=0)
        
        if not model_consistency.empty:
            im = axes[1, 1].imshow(model_consistency.values, aspect='auto', cmap='viridis')
            axes[1, 1].set_xticks(range(len(model_consistency.columns)))
            axes[1, 1].set_xticklabels([x.replace('_', '-').title() for x in model_consistency.columns])
            axes[1, 1].set_yticks(range(len(model_consistency.index)))
            axes[1, 1].set_yticklabels(model_consistency.index)
            axes[1, 1].set_title('Model Performance Heatmap')
            
            # Add text annotations
            for i in range(len(model_consistency.index)):
                for j in range(len(model_consistency.columns)):
                    text = axes[1, 1].text(j, i, f'{model_consistency.iloc[i, j]:.3f}',
                                         ha="center", va="center", color="white" if model_consistency.iloc[i, j] < 0.5 else "black")
            
            plt.colorbar(im, ax=axes[1, 1], label='R² Score')
        
        # Plot 4: Validation strategy robustness analysis
        robustness_data = []
        for config in successful_df.groupby(['model_type', 'method', 'target_log_transform']):
            config_name = f"{config[0][0]}_{config[0][1]}_log{config[0][2]}"
            config_data = config[1]
            
            if len(config_data) >= 2:  # Need multiple validation types
                r2_scores = config_data['r2_score'].values
                robustness_score = 1 / (1 + np.std(r2_scores) / np.mean(r2_scores)) if np.mean(r2_scores) > 0 else 0
                
                robustness_data.append({
                    'config': config_name,
                    'robustness_score': robustness_score,
                    'mean_r2': np.mean(r2_scores),
                    'std_r2': np.std(r2_scores)
                })
        
        if robustness_data:
            rob_df = pd.DataFrame(robustness_data).sort_values('robustness_score', ascending=False)
            
            bars = axes[2, 1].barh(range(len(rob_df)), rob_df['robustness_score'])
            axes[2, 1].set_yticks(range(len(rob_df)))
            axes[2, 1].set_yticklabels(rob_df['config'], fontsize=8)
            axes[2, 1].set_xlabel('Robustness Score')
            axes[2, 1].set_title('Configuration Robustness Ranking')
            axes[2, 1].grid(axis='x', alpha=0.3)
            
            # Highlight top configuration
            if len(bars) > 0:
                bars[0].set_color('#d62728')
        
        plt.tight_layout()
        
        # Save comprehensive analysis
        analysis_path = self.results_dir / 'triple_validation_comprehensive_analysis.png'
        plt.savefig(analysis_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save detailed results
        summary_path = self.results_dir / 'triple_validation_summary.csv'
        results_df.to_csv(summary_path, index=False)
        
        print(f"\n{'='*80}")
        print("TRIPLE VALIDATION ANALYSIS COMPLETED!")
        print(f"{'='*80}")
        print(f"Comprehensive analysis saved: {analysis_path}")
        print(f"Detailed results saved: {summary_path}")
        print(f"Total experiments: {len(results_df)}")
        print(f"Successful experiments: {len(successful_df)}")
        
        # Print best overall configuration
        if len(successful_df) > 0:
            best_overall = successful_df.loc[successful_df['r2_score'].idxmax()]
            print(f"\nBEST OVERALL CONFIGURATION:")
            print(f"  Model: {best_overall['model_type']}")
            print(f"  Method: {best_overall['method']}")
            print(f"  Log Transform: {best_overall['target_log_transform']}")
            print(f"  Validation: {best_overall['validation_type']}")
            print(f"  R² Score: {best_overall['r2_score']:.3f}")
            print(f"  Tuned: {best_overall['hyperparameter_tuning']}")


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
        'test_size': 0.15,  # 15% for test (with 25,900 samples, this is ~3,885 samples)
        'val_size': 0.15,   # 15% for validation  
        'random_state': 42,
        'output_dir': f"{MODELS_PATH}/triple_validation_experiments",
        
        # Enhanced temporal configuration
        'temporal_facilities': ['suncor', 'rmbc', 'bluespruce'],
        'train_months': [1,2,3,4,5,6,7,8,9],  # Jan-Sep for training
        'val_months': [10],                    # Oct for validation
        'test_months': [11,12]                 # Nov-Dec for testing
    }
    
    # Define data paths (using combined dataset from previous step)
    DATA_PATHS = {
        'X_features': f"{DATA_PATH}/processed_data/combined_large/X_features_combined_all_facilities.csv",
        'y_targets_template': f"{DATA_PATH}/processed_data/combined_large/y_{{method}}_combined_all_facilities.csv"
    }
    
    # Run comprehensive triple validation experiments
    experiment_runner = TripleValidationExperimentRunner(BASE_CONFIG, DATA_PATHS)
    all_results = experiment_runner.run_triple_validation_experiments()
    
    print(f"\n{'='*80}")
    print("ALL TRIPLE VALIDATION EXPERIMENTS COMPLETED!")
    print(f"Check results in: {BASE_CONFIG['output_dir']}")
    print(f"{'='*80}")