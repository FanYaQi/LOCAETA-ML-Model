"""
Base ML trainer with common functionality for air quality prediction
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
from scipy.stats import gaussian_kde
from utils.path_util import DATA_PATH, MODELS_PATH

warnings.filterwarnings('ignore')


class BaseMLTrainer:
    """
    Base ML trainer for air quality prediction with common functionality
    """
    
    def __init__(self, config):
        """Initialize with configuration dictionary"""
        self.config = config
        self.model = None
        self.scaler = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.feature_names = None
        self.metrics = {}
        
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
        print(f"Available facilities: {sorted(X_features['facility_id'].unique()) if 'facility_id' in X_features.columns else 'No facility info'}")
        
        # Select target method
        method = self.config['method']
        if f'y_{method}' in locals():
            y_data = pd.read_csv(y_targets_path.replace('y_target', f'y_target_{method}'))
        else:
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
            y_min = y[y > 0].min() if (y > 0).any() else 1e-6
            y = np.log(y + y_min * 0.01)
            print("Applied log transform to target")
        
        # Store feature names (exclude non-predictive columns)
        exclude_cols = ['month', 'grid_i', 'grid_j', 'facility_id']
        self.feature_names = [col for col in X.columns if col not in exclude_cols]
        
        return X, y
    
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
    
    def _plot_scatter_with_intensity(self, ax):
        """Create scatter plot with intensity coloring"""
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
        
        # Color bars
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
        base_name = f"{self.config['model_type']}_{self.config['method']}_{self.config['validation_type']}"
        
        if self.config.get('figure_suffix'):
            base_name += f"_{self.config['figure_suffix']}"
        
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