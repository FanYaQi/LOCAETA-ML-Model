"""
Multi-facility model trainer with spatial validation
"""
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
        """Initialize multi-facility model trainer"""
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.results = {}
        self.facilities = []
        
    def load_multi_facility_data(self, facilities: List[str], year: int = 2023, 
                                grid_size: int = 24) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and combine data from multiple facilities"""
        self.facilities = facilities
        all_X = []
        all_y = []
        
        for facility in facilities:
            try:
                X_path = f"{DATA_PATH}/processed_data/by_facility/X_features_{facility}_{year}_grid{grid_size}.csv"
                y_path = f"{DATA_PATH}/processed_data/by_facility/y_target_method1_{facility}_{year}_grid{grid_size}.csv"
                
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
        """Create spatial validation splits - hold out entire facilities for validation"""
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
        """Create random splits stratified by facility when facility-level splits aren't possible"""
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
        """Prepare features and targets for multi-facility training"""
        # Remove non-feature columns and facility identifier
        excluded_cols = [
            'month', 'grid_i', 'grid_j', 'facility', 'facility_id',
            'landcover_urban_percent',      
            'landcover_forest_percent',     
            'landcover_agriculture_percent' 
        ]
        
        # Get available feature columns (handle missing columns gracefully)
        feature_cols = [col for col in X_features.columns if col not in excluded_cols]
        print(f"Available columns: {list(X_features.columns)}")
        print(f"Excluded columns found: {[col for col in excluded_cols if col in X_features.columns]}")
        print(f"Using {len(feature_cols)} features: {feature_cols[:5]}..." if len(feature_cols) > 5 else f"Using features: {feature_cols}")
        
        # Convert to numpy arrays
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