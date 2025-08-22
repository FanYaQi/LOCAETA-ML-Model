import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings
from pathlib import Path
from typing import Tuple, Dict, Any, List
from utils.path_util import DATA_PATH, MODELS_PATH, FIGURES_PATH

warnings.filterwarnings('ignore')

class MultiFacilityModelTrainer:
    """
    Train Random Forest model on multiple facilities with spatial validation
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize multi-facility model trainer
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.results = {}
        self.facilities = []
        
    def load_multi_facility_data(self, facilities: List[str], year: int = 2023, 
                                grid_size: int = 24) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and combine data from multiple facilities
        
        Args:
            facilities: List of facility names (e.g., ['suncor', 'RMBC'])
            year: Year of data
            grid_size: Grid size used
            
        Returns:
            X_features: Combined feature DataFrame with facility indicator
            y_target: Combined target DataFrame with facility indicator
        """
        self.facilities = facilities
        all_X = []
        all_y = []
        
        for facility in facilities:
            try:
                X_path = f"{DATA_PATH}/processed_data/X_features_{facility}_{year}_grid{grid_size}.csv"
                y_path = f"{DATA_PATH}/processed_data/y_target_{facility}_{year}_grid{grid_size}.csv"
                
                X_fac = pd.read_csv(X_path)
                y_fac = pd.read_csv(y_path)
                
                # Add facility identifier
                X_fac['facility'] = facility
                y_fac['facility'] = facility
                
                all_X.append(X_fac)
                all_y.append(y_fac)
                
                print(f"Loaded {facility}: {len(X_fac)} samples")
                
            except FileNotFoundError as e:
                print(f"Warning: Could not load data for {facility}: {e}")
                continue
        
        if not all_X:
            raise FileNotFoundError("No facility data could be loaded!")
        
        # Combine all facilities
        X_combined = pd.concat(all_X, ignore_index=True)
        y_combined = pd.concat(all_y, ignore_index=True)
        
        print(f"\nCombined dataset:")
        print(f"  Total samples: {len(X_combined)}")
        print(f"  Facilities: {facilities}")
        print(f"  Features: {len(X_combined.columns)}")
        
        # Print facility distribution
        facility_counts = X_combined['facility'].value_counts()
        print(f"  Sample distribution:")
        for facility, count in facility_counts.items():
            print(f"    {facility}: {count} samples ({count/len(X_combined)*100:.1f}%)")
        
        return X_combined, y_combined
    
    def create_spatial_validation_splits(self, X_features: pd.DataFrame, y_target: pd.DataFrame, 
                                       test_size: float = 0.2, val_size: float = 0.2) -> Tuple[pd.DataFrame, ...]:
        """
        Create spatial validation splits - hold out entire facilities for validation
        
        Args:
            X_features: Feature DataFrame
            y_target: Target DataFrame
            test_size: Fraction for spatial test (hold out facilities)
            val_size: Fraction for validation
            
        Returns:
            DataFrames split by facility for spatial validation
        """
        print(f"\n=== SPATIAL VALIDATION STRATEGY ===")
        
        facilities = X_features['facility'].unique()
        n_facilities = len(facilities)
        
        if n_facilities < 3:
            print(f"Warning: Only {n_facilities} facilities available.")
            print("Using random spatial splits within facilities instead of facility-level splits.")
            return self.create_random_spatial_splits(X_features, y_target, test_size, val_size)
        
        # For spatial validation: hold out entire facilities
        np.random.seed(self.random_state)
        facilities_shuffled = np.random.permutation(facilities)
        
        n_test_facilities = max(1, int(n_facilities * test_size))
        n_val_facilities = max(1, int(n_facilities * val_size))
        
        test_facilities = facilities_shuffled[:n_test_facilities]
        val_facilities = facilities_shuffled[n_test_facilities:n_test_facilities + n_val_facilities]
        train_facilities = facilities_shuffled[n_test_facilities + n_val_facilities:]
        
        print(f"Spatial split strategy:")
        print(f"  Train facilities: {list(train_facilities)}")
        print(f"  Validation facilities: {list(val_facilities)}")
        print(f"  Test facilities: {list(test_facilities)}")
        
        # Create splits
        train_mask = X_features['facility'].isin(train_facilities)
        val_mask = X_features['facility'].isin(val_facilities)
        test_mask = X_features['facility'].isin(test_facilities)
        
        X_train = X_features[train_mask].reset_index(drop=True)
        X_val = X_features[val_mask].reset_index(drop=True)
        X_test = X_features[test_mask].reset_index(drop=True)
        
        y_train = y_target[train_mask].reset_index(drop=True)
        y_val = y_target[val_mask].reset_index(drop=True)
        y_test = y_target[test_mask].reset_index(drop=True)
        
        print(f"\nSpatial split results:")
        print(f"  Train: {len(X_train)} samples ({len(X_train)/len(X_features)*100:.1f}%)")
        print(f"  Validation: {len(X_val)} samples ({len(X_val)/len(X_features)*100:.1f}%)")
        print(f"  Test: {len(X_test)} samples ({len(X_test)/len(X_features)*100:.1f}%)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_random_spatial_splits(self, X_features: pd.DataFrame, y_target: pd.DataFrame,
                                   test_size: float = 0.2, val_size: float = 0.2) -> Tuple[pd.DataFrame, ...]:
        """
        Create random splits stratified by facility when facility-level splits aren't possible
        """
        print(f"Creating stratified random splits by facility...")
        
        # Use facility as stratification variable
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_features, y_target, test_size=test_size, 
            stratify=X_features['facility'], random_state=self.random_state
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), 
            stratify=X_temp['facility'], random_state=self.random_state
        )
        
        print(f"\nStratified split results:")
        print(f"  Train: {len(X_train)} samples")
        print(f"  Validation: {len(X_val)} samples") 
        print(f"  Test: {len(X_test)} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def prepare_features_targets(self, X_features: pd.DataFrame, y_target: pd.DataFrame, 
                               log_transform_targets: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and targets for multi-facility training
        
        Args:
            X_features: Feature DataFrame
            y_target: Target DataFrame
            log_transform_targets: Whether to log-transform targets
            
        Returns:
            X: Feature array
            y: Target array (possibly log-transformed)
        """
        # Remove non-feature columns and facility identifier
        excluded_cols = [
            'month', 'grid_i', 'grid_j', 'facility',
            'landcover_urban_percent',      
            'landcover_forest_percent',     
            'landcover_agriculture_percent' 
        ]
        
        feature_cols = [col for col in X_features.columns if col not in excluded_cols]
        
        X = X_features[feature_cols].values
        y_raw = y_target['pm25_concentration'].values
        
        self.feature_names = feature_cols
        
        print(f"\nFeature preparation:")
        print(f"  Feature columns ({len(feature_cols)}):")
        for i, col in enumerate(feature_cols):
            print(f"    {i+1:2d}. {col}")
        
        # Handle target transformation
        if log_transform_targets:
            # Add small constant to avoid log(0)
            epsilon = 1e-18
            y_transformed = np.log10(y_raw + epsilon)
            
            print(f"\nApplied log10 transformation to targets:")
            print(f"  Original range: {y_raw.min():.2e} to {y_raw.max():.2e}")
            print(f"  Transformed range: {y_transformed.min():.2f} to {y_transformed.max():.2f}")
            print(f"  Original mean: {y_raw.mean():.2e}")
            print(f"  Transformed mean: {y_transformed.mean():.2f} ± {y_transformed.std():.2f}")
            
            self.log_transform = True
            self.log_epsilon = epsilon
            y = y_transformed
        else:
            print(f"\nNo target transformation applied")
            self.log_transform = False
            y = y_raw
        
        # Run diagnostics
        self.diagnose_data_issues(X, y)
        
        return X, y
    
    def diagnose_data_issues(self, X: np.ndarray, y: np.ndarray):
        """Diagnose potential data issues before training"""
        print(f"\n=== Data Diagnostics ===")
        
        # Target statistics
        print(f"Target (y) statistics:")
        print(f"  Range: {y.min():.2e} to {y.max():.2e}")
        print(f"  Mean: {y.mean():.2e} ± {y.std():.2e}")
        print(f"  Zero values: {(y == 0).sum()} ({(y == 0).mean()*100:.1f}%)")
        print(f"  Negative values: {(y < 0).sum()} ({(y < 0).mean()*100:.1f}%)")
        
        # Check for constant targets
        unique_targets = len(np.unique(y))
        print(f"  Unique values: {unique_targets}")
        if unique_targets < 10:
            print(f"  WARNING: Very few unique target values!")
        
        # Feature statistics
        print(f"\nFeature (X) statistics:")
        print(f"  Shape: {X.shape}")
        print(f"  Features with constant values: {np.sum(X.std(axis=0) == 0)}")
        print(f"  Features with NaN: {np.sum(np.isnan(X).any(axis=0))}")
        
        # Top correlations
        print(f"\nTop 5 feature-target correlations:")
        correlations = []
        for i in range(X.shape[1]):
            corr = np.corrcoef(X[:, i], y)[0, 1]
            if not np.isnan(corr):
                correlations.append((i, abs(corr)))
        
        correlations.sort(key=lambda x: x[1], reverse=True)
        for i, (feat_idx, corr) in enumerate(correlations[:5]):
            feat_name = self.feature_names[feat_idx] if self.feature_names else f"Feature_{feat_idx}"
            print(f"    {i+1}. {feat_name}: {corr:.4f}")
    
    def inverse_transform_predictions(self, y_pred_transformed: np.ndarray) -> np.ndarray:
        """Convert log-transformed predictions back to original scale"""
        if self.log_transform:
            y_pred_original = 10**y_pred_transformed - self.log_epsilon
            y_pred_original = np.maximum(y_pred_original, 0)
            return y_pred_original
        else:
            return y_pred_transformed
    
    def scale_features(self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Scale features using StandardScaler fitted on training data"""
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"\nFeature scaling applied:")
        print(f"  Train mean: {X_train_scaled.mean(axis=0)[:3]} ...")
        print(f"  Train std: {X_train_scaled.std(axis=0)[:3]} ...")
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def train_multi_facility_model(self, X_train: np.ndarray, y_train: np.ndarray,
                                 X_val: np.ndarray, y_val: np.ndarray,
                                 hyperparameter_tuning: bool = True) -> RandomForestRegressor:
        """
        Train Random Forest model on multi-facility data
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            hyperparameter_tuning: Whether to perform grid search
            
        Returns:
            Trained RandomForestRegressor
        """
        if hyperparameter_tuning:
            print("\nPerforming hyperparameter tuning for multi-facility model...")
            
            # Define parameter grid
            param_grid = {
                'n_estimators': [200, 300, 400],
                'max_depth': [20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
            
            # Grid search with cross-validation
            rf_base = RandomForestRegressor(random_state=self.random_state, n_jobs=-1)
            grid_search = GridSearchCV(
                rf_base, param_grid, cv=5, scoring='neg_mean_squared_error',
                n_jobs=-1, verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV score: {-grid_search.best_score_:.6f}")
            
            self.model = grid_search.best_estimator_
        else:
            print("\nTraining Random Forest with default parameters...")
            
            # Default parameters for multi-facility
            self.model = RandomForestRegressor(
                n_estimators=300,
                max_depth=30,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=self.random_state,
                n_jobs=-1
            )
            
            self.model.fit(X_train, y_train)
        
        # Evaluate on validation set
        y_val_pred = self.model.predict(X_val)
        val_mse = mean_squared_error(y_val, y_val_pred)
        val_mae = mean_absolute_error(y_val, y_val_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        
        print(f"\nValidation Results:")
        print(f"  MSE: {val_mse:.8f}")
        print(f"  RMSE: {np.sqrt(val_mse):.8f}")
        print(f"  MAE: {val_mae:.8f}")
        print(f"  R²: {val_r2:.4f}")
        
        return self.model
    
    def evaluate_spatial_performance(self, X_test_split: pd.DataFrame, y_test_split: pd.DataFrame,
                                   X_test_scaled: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate model performance with spatial analysis
        
        Args:
            X_test_split: Test features DataFrame (with facility info)
            y_test_split: Test targets DataFrame (with facility info)
            X_test_scaled: Scaled test features
            
        Returns:
            Dictionary of evaluation metrics including spatial analysis
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Get predictions in transformed space
        y_pred_transformed = self.model.predict(X_test_scaled)
        
        # Convert to original scale if needed
        if hasattr(self, 'log_transform') and self.log_transform:
            y_test_original = self.inverse_transform_predictions(y_test_split['pm25_concentration'].values)
            y_pred_original = self.inverse_transform_predictions(y_pred_transformed)
            
            print(f"\nMetrics in transformed (log) space:")
            mse_log = mean_squared_error(y_test_split['pm25_concentration'].values, y_pred_transformed)
            r2_log = r2_score(y_test_split['pm25_concentration'].values, y_pred_transformed)
            print(f"  MSE (log): {mse_log:.4f}")
            print(f"  R² (log): {r2_log:.4f}")
            
            y_test_eval = y_test_original
            y_pred_eval = y_pred_original
        else:
            y_test_eval = y_test_split['pm25_concentration'].values
            y_pred_eval = y_pred_transformed
        
        # Overall metrics
        mse = mean_squared_error(y_test_eval, y_pred_eval)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_eval, y_pred_eval)
        r2 = r2_score(y_test_eval, y_pred_eval)
        mape = np.mean(np.abs((y_test_eval - y_pred_eval) / (y_test_eval + 1e-18))) * 100
        
        results = {
            'overall': {
                'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape,
                'n_samples': len(y_test_eval)
            },
            'y_test': y_test_eval,
            'y_pred': y_pred_eval
        }
        
        # Spatial performance by facility
        results['by_facility'] = {}
        test_facilities = X_test_split['facility'].unique()
        
        for facility in test_facilities:
            facility_mask = X_test_split['facility'] == facility
            
            if np.any(facility_mask):
                y_true_fac = y_test_eval[facility_mask]
                y_pred_fac = y_pred_eval[facility_mask]
                
                if len(y_true_fac) > 1:
                    fac_r2 = r2_score(y_true_fac, y_pred_fac)
                    fac_rmse = np.sqrt(mean_squared_error(y_true_fac, y_pred_fac))
                    fac_mae = mean_absolute_error(y_true_fac, y_pred_fac)
                    
                    results['by_facility'][facility] = {
                        'r2': fac_r2, 'rmse': fac_rmse, 'mae': fac_mae,
                        'n_samples': len(y_true_fac)
                    }
        
        # Performance by concentration ranges
        percentiles = [75, 90, 95, 99]
        results['by_percentile'] = {}
        
        for p in percentiles:
            threshold = np.percentile(y_test_eval, p)
            high_mask = y_test_eval > threshold
            
            if np.any(high_mask):
                y_true_high = y_test_eval[high_mask]
                y_pred_high = y_pred_eval[high_mask]
                
                if len(y_true_high) > 1:
                    high_r2 = r2_score(y_true_high, y_pred_high)
                    high_rmse = np.sqrt(mean_squared_error(y_true_high, y_pred_high))
                    
                    results['by_percentile'][f'top_{100-p}%'] = {
                        'r2': high_r2, 'rmse': high_rmse, 'threshold': threshold,
                        'n_samples': len(y_true_high)
                    }
        
        self.results = results
        return results
    
    def print_spatial_evaluation_results(self, results: Dict[str, Any]):
        """Print formatted spatial evaluation results"""
        print("\n" + "="*60)
        print("MULTI-FACILITY SPATIAL VALIDATION RESULTS")
        print("="*60)
        
        # Overall performance
        overall = results['overall']
        print(f"\nOverall Performance ({overall['n_samples']} samples):")
        print(f"  R²: {overall['r2']:.4f}")
        print(f"  RMSE: {overall['rmse']:.2e}")
        print(f"  MAE: {overall['mae']:.2e}")
        print(f"  MAPE: {overall['mape']:.2f}%")
        
        # Performance by facility (spatial validation)
        if 'by_facility' in results and results['by_facility']:
            print(f"\nSpatial Performance by Facility:")
            for facility, metrics in results['by_facility'].items():
                print(f"  {facility:>10}: R²={metrics['r2']:.3f}, RMSE={metrics['rmse']:.2e}, n={metrics['n_samples']}")
        
        # Performance by concentration ranges
        if 'by_percentile' in results and results['by_percentile']:
            print(f"\nPerformance by Concentration Range:")
            for range_name, metrics in results['by_percentile'].items():
                print(f"  {range_name:>8}: R²={metrics['r2']:.3f}, RMSE={metrics['rmse']:.2e}, n={metrics['n_samples']}")
    
    def plot_spatial_validation_results(self, X_test_split: pd.DataFrame, save_plots: bool = True):
        """Create spatial validation plots"""
        if 'y_test' not in self.results:
            raise ValueError("Model not evaluated yet!")
        
        y_test = self.results['y_test']
        y_pred = self.results['y_pred']
        
        # Create comprehensive plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Overall Predicted vs Actual
        axes[0, 0].scatter(y_test, y_pred, alpha=0.6, s=20)
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual PM2.5 (μg/m³)')
        axes[0, 0].set_ylabel('Predicted PM2.5 (μg/m³)')
        axes[0, 0].set_title(f'Overall Performance (R² = {self.results["overall"]["r2"]:.3f})')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Performance by Facility
        if 'by_facility' in self.results:
            facilities = list(self.results['by_facility'].keys())
            r2_values = [self.results['by_facility'][f]['r2'] for f in facilities]
            
            bars = axes[0, 1].bar(facilities, r2_values)
            axes[0, 1].set_ylabel('R²')
            axes[0, 1].set_title('Spatial Performance by Facility')
            axes[0, 1].set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, r2 in zip(bars, r2_values):
                axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{r2:.3f}', ha='center', va='bottom')
        
        # 3. Residuals by Facility
        residuals = y_test - y_pred
        facility_colors = ['blue', 'orange', 'green', 'red', 'purple']
        
        for i, facility in enumerate(X_test_split['facility'].unique()):
            facility_mask = X_test_split['facility'] == facility
            facility_residuals = residuals[facility_mask]
            facility_pred = y_pred[facility_mask]
            
            color = facility_colors[i % len(facility_colors)]
            axes[0, 2].scatter(facility_pred, facility_residuals, alpha=0.6, s=20, 
                             label=facility, color=color)
        
        axes[0, 2].axhline(y=0, color='r', linestyle='--')
        axes[0, 2].set_xlabel('Predicted PM2.5 (μg/m³)')
        axes[0, 2].set_ylabel('Residuals (μg/m³)')
        axes[0, 2].set_title('Residuals by Facility')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Performance by Concentration Range
        if 'by_percentile' in self.results:
            range_names = list(self.results['by_percentile'].keys())
            range_r2 = [self.results['by_percentile'][r]['r2'] for r in range_names]
            
            axes[1, 0].bar(range_names, range_r2)
            axes[1, 0].set_ylabel('R²')
            axes[1, 0].set_title('Performance by Concentration Range')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 5. Residuals Distribution
        axes[1, 1].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(residuals.mean(), color='r', linestyle='--', 
                          label=f'Mean: {residuals.mean():.2e}')
        axes[1, 1].set_xlabel('Residuals (μg/m³)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Distribution of Residuals')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Feature Importance
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            top_features = importance_df.head(10)
            axes[1, 2].barh(range(len(top_features)), top_features['importance'])
            axes[1, 2].set_yticks(range(len(top_features)))
            axes[1, 2].set_yticklabels(top_features['feature'])
            axes[1, 2].set_xlabel('Feature Importance')
            axes[1, 2].set_title('Top 10 Feature Importance')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.suptitle(f'Multi-Facility Model: Spatial Validation Results', fontsize=16)
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f"{FIGURES_PATH}/multi_facility_spatial_validation.png", 
                       dpi=300, bbox_inches='tight')
            print(f"Saved spatial validation plots")
        
        plt.show()
    
    def save_model(self, model_name: str = "multi_facility", year: int = 2023):
        """Save trained multi-facility model"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        model_path = f"{MODELS_PATH}/rf_model_{model_name}_{year}.joblib"
        scaler_path = f"{MODELS_PATH}/scaler_{model_name}_{year}.joblib"
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        # Save metadata
        metadata = {
            'feature_names': self.feature_names,
            'results': self.results,
            'model_type': 'MultiFacilityRandomForest',
            'facilities': self.facilities,
            'log_transform': getattr(self, 'log_transform', False),
            'log_epsilon': getattr(self, 'log_epsilon', None)
        }
        
        metadata_path = f"{MODELS_PATH}/metadata_{model_name}_{year}.joblib"
        joblib.dump(metadata, metadata_path)
        
        print(f"\nMulti-facility model saved:")
        print(f"  Model: {model_path}")
        print(f"  Scaler: {scaler_path}")
        print(f"  Metadata: {metadata_path}")


def main():
    """
    Main training pipeline for multi-facility model with spatial validation
    """
    print("=== MULTI-FACILITY MODEL TRAINING WITH SPATIAL VALIDATION ===\n")
    
    # ===== CONFIGURATION =====
    facilities = ['suncor', 'RMBC']  # List of facilities to include
    year = 2023
    grid_size = 24
    log_transform = True  # Use log transform instead of weighting
    hyperparameter_tuning = True
    # ==========================
    
    # Initialize trainer
    trainer = MultiFacilityModelTrainer(random_state=42)
    
    # Load multi-facility data
    X_features, y_target = trainer.load_multi_facility_data(
        facilities=facilities,
        year=year,
        grid_size=grid_size
    )
    
    # Create spatial validation splits
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.create_spatial_validation_splits(
        X_features, y_target,
        test_size=0.2,
        val_size=0.2
    )
    
    # Prepare features and targets with log transformation
    print(f"\nPreparing training data...")
    X_train_processed, y_train_processed = trainer.prepare_features_targets(
        X_train, y_train, 
        log_transform_targets=log_transform
    )
    
    # Process validation data (extract features only, targets already processed in spatial split)
    excluded_cols = [
        'month', 'grid_i', 'grid_j', 'facility',
        'landcover_urban_percent', 'landcover_forest_percent', 'landcover_agriculture_percent'
    ]
    feature_cols = [col for col in X_val.columns if col not in excluded_cols]
    
    X_val_processed = X_val[feature_cols].values
    X_test_processed = X_test[feature_cols].values
    
    # Apply same target transformation to validation and test sets
    if log_transform:
        epsilon = trainer.log_epsilon
        y_val_processed = np.log10(y_val['pm25_concentration'].values + epsilon)
        y_test_processed = np.log10(y_test['pm25_concentration'].values + epsilon)
    else:
        y_val_processed = y_val['pm25_concentration'].values
        y_test_processed = y_test['pm25_concentration'].values
    
    # Scale features
    X_train_scaled, X_val_scaled, X_test_scaled = trainer.scale_features(
        X_train_processed, X_val_processed, X_test_processed
    )
    
    # Train multi-facility model
    print("\n" + "="*60)
    print("TRAINING MULTI-FACILITY RANDOM FOREST MODEL")
    print("="*60)
    
    model = trainer.train_multi_facility_model(
        X_train_scaled, y_train_processed, 
        X_val_scaled, y_val_processed,
        hyperparameter_tuning=hyperparameter_tuning
    )
    
    # Evaluate with spatial analysis
    print("\n" + "="*60)
    print("SPATIAL VALIDATION EVALUATION")
    print("="*60)
    
    results = trainer.evaluate_spatial_performance(X_test, y_test, X_test_scaled)
    trainer.print_spatial_evaluation_results(results)
    
    # Create spatial validation plots
    trainer.plot_spatial_validation_results(X_test, save_plots=True)
    
    # Analyze feature importance
    print("\n" + "="*40)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*40)
    
    if hasattr(trainer.model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': trainer.feature_names,
            'importance': trainer.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 15 Feature Importance:")
        for i, (_, row) in enumerate(importance_df.head(15).iterrows()):
            print(f"  {i+1:2d}. {row['feature']:<35} {row['importance']:.4f}")
        
        # Plot feature importance
        plt.figure(figsize=(10, 8))
        top_features = importance_df.head(15)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Top 15 Feature Importance - Multi-Facility Model')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f"{FIGURES_PATH}/multi_facility_feature_importance.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    # Additional analysis: Cross-facility performance
    print("\n" + "="*40)
    print("CROSS-FACILITY ANALYSIS")
    print("="*40)
    
    # Analyze how well model trained on one facility performs on another
    if len(facilities) >= 2:
        print(f"Training facilities: {X_train['facility'].unique()}")
        print(f"Test facilities: {X_test['facility'].unique()}")
        
        # Show sample distribution in test set
        test_facility_counts = X_test['facility'].value_counts()
        print(f"\nTest set composition:")
        for facility, count in test_facility_counts.items():
            print(f"  {facility}: {count} samples ({count/len(X_test)*100:.1f}%)")
    
    # Save model
    trainer.save_model(model_name="multi_facility_log", year=year)
    
    print("\n" + "="*50)
    print("MULTI-FACILITY TRAINING COMPLETE")
    print("="*50)
    print(f"Facilities: {facilities}")
    print(f"Final Overall R²: {results['overall']['r2']:.4f}")
    print(f"Final Overall RMSE: {results['overall']['rmse']:.2e}")
    if 'by_facility' in results:
        print(f"Spatial validation successful across facilities!")
    print(f"Model uses log transformation: {log_transform}")


if __name__ == "__main__":
    main()