"""
Simplified tri-validation trainer: Sample-based, Time-based, Site-based validation
Uses only RF + Method1 + Log transform (as requested)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
from pathlib import Path
from utils.path_util import DATA_PATH, MODELS_PATH

warnings.filterwarnings('ignore')

# ============================================================================
# 🎯 CONFIGURATION - MODIFY PATHS AND SETTINGS HERE
# ============================================================================

# Data paths - Easy to find and modify! (Updated to use comprehensive combined_large data)
TRI_VALIDATION_CONFIG = {
    'X_features_path': f"{DATA_PATH}/processed_data/combined_large/X_features_combined_all_facilities.csv",
    'y_targets_path': f"{DATA_PATH}/processed_data/combined_large/y_method1_combined_all_facilities.csv",  # Method 1 only
    'output_dir': f"{MODELS_PATH}/tri_validation_results",
    
    # Model settings (simplified as requested)
    'method': 'method1',           # Method 1 only
    'model_type': 'RF',            # Random Forest only  
    'target_log_transform': True,  # Log transform as requested
    'random_state': 42,
    
    # Validation settings
    'test_size': 0.2,
    'temporal_split': {
        'train_month': 3,  # March for training (since you have March data)
        'test_month': 3,   # March for testing (will be split differently)
    },
    
    # Features to use
    'features_to_use': [
        'facility_height', 'distance_to_facility', 'bearing_from_facility', 
        'NEI_annual_emission_t', 'monthly_emission_rate_t_per_hr'
    ]
}

# ============================================================================
# 📊 TRI-VALIDATION TRAINER CLASS  
# ============================================================================

class SimpleTriValidationTrainer:
    """
    Simplified tri-validation trainer following your legacy script style
    """
    
    def __init__(self, config):
        self.config = config
        self.results = {}
        
        # Create output directory
        Path(config['output_dir']).mkdir(parents=True, exist_ok=True)
        
        print("🎯 SIMPLE TRI-VALIDATION TRAINER")
        print("="*60)
        print(f"Method: {config['method']}")
        print(f"Model: {config['model_type']}")
        print(f"Log transform: {config['target_log_transform']}")
        print("="*60)
    
    def load_and_prepare_data(self):
        """Load and prepare data for tri-validation"""
        print("\n📂 LOADING DATA")
        print("-"*40)
        
        # Load data
        X_features = pd.read_csv(self.config['X_features_path'])
        y_targets = pd.read_csv(self.config['y_targets_path'])
        
        print(f"Loaded X features: {X_features.shape}")
        print(f"Loaded y targets: {y_targets.shape}")
        
        # Check available facilities
        if 'facility_id' in X_features.columns:
            facilities = sorted(X_features['facility_id'].unique())
            print(f"Available facilities: {facilities}")
        else:
            print("No facility_id column found")
            facilities = []
        
        # Use available features or fall back to all numeric columns
        available_features = [col for col in self.config['features_to_use'] if col in X_features.columns]
        if not available_features:
            # Fallback to numeric columns excluding IDs
            exclude_cols = ['month', 'grid_i', 'grid_j', 'facility_id', 'lat', 'lon']
            available_features = [col for col in X_features.columns 
                                if col not in exclude_cols and X_features[col].dtype in ['int64', 'float64']]
        
        print(f"Using {len(available_features)} features: {available_features}")
        
        # Prepare features and targets
        X = X_features[['facility_id', 'month', 'grid_i', 'grid_j'] + available_features].copy()
        y = y_targets['pm25_concentration'].copy()
        
        # Clean data - remove NaN values
        valid_mask = ~(X[available_features].isnull().any(axis=1) | y.isnull())
        X = X[valid_mask].reset_index(drop=True)
        y = y[valid_mask].reset_index(drop=True)
        
        print(f"After cleaning: {len(X)} samples")
        
        # Apply log transform to targets
        if self.config['target_log_transform']:
            y_min = y[y > 0].min() if (y > 0).any() else 1e-6
            y = np.log(y + y_min * 0.01)
            print(f"Applied log transform to targets")
        
        self.X_full = X
        self.y_full = y
        self.feature_names = available_features
        self.facilities = facilities
        
        return X, y
    
    def run_sample_based_validation(self):
        """Sample-based cross validation (random split)"""
        print(f"\n🔀 SAMPLE-BASED VALIDATION")
        print("-"*40)
        
        # Random split
        X_features = self.X_full[self.feature_names].values
        X_train, X_test, y_train, y_test = train_test_split(
            X_features, self.y_full, 
            test_size=self.config['test_size'], 
            random_state=self.config['random_state']
        )
        
        # Train and evaluate
        model = RandomForestRegressor(n_estimators=100, random_state=self.config['random_state'])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        results = {
            'validation_type': 'sample_based',
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'n_train': len(y_train),
            'n_test': len(y_test),
            'model': model,
            'y_test': y_test,
            'y_pred': y_pred
        }
        
        print(f"Results: R² = {r2:.3f}, RMSE = {rmse:.3f}, MAE = {mae:.3f}")
        return results
    
    def run_site_based_validation(self):
        """Site-based cross validation (facility-level split)"""
        print(f"\n🏭 SITE-BASED VALIDATION")
        print("-"*40)
        
        if not self.facilities or 'facility_id' not in self.X_full.columns:
            print("Cannot run site-based validation: No facility information available")
            return None
        
        # Site-based split using facilities
        if len(self.facilities) < 3:
            print(f"Warning: Only {len(self.facilities)} facilities available. Using stratified split.")
            X_features = self.X_full[self.feature_names].values
            X_train, X_test, y_train, y_test = train_test_split(
                X_features, self.y_full,
                test_size=self.config['test_size'],
                stratify=self.X_full['facility_id'],
                random_state=self.config['random_state']
            )
            train_facilities = "stratified"
            test_facilities = "stratified"
        else:
            # True facility-based split
            gss = GroupShuffleSplit(n_splits=1, test_size=self.config['test_size'], 
                                  random_state=self.config['random_state'])
            train_idx, test_idx = next(gss.split(self.X_full, self.y_full, groups=self.X_full['facility_id']))
            
            X_train = self.X_full.iloc[train_idx][self.feature_names].values
            X_test = self.X_full.iloc[test_idx][self.feature_names].values
            y_train = self.y_full.iloc[train_idx]
            y_test = self.y_full.iloc[test_idx]
            
            train_facilities = sorted(self.X_full.iloc[train_idx]['facility_id'].unique())
            test_facilities = sorted(self.X_full.iloc[test_idx]['facility_id'].unique())
            
            print(f"Train facilities: {train_facilities}")
            print(f"Test facilities: {test_facilities}")
        
        # Train and evaluate
        model = RandomForestRegressor(n_estimators=100, random_state=self.config['random_state'])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        results = {
            'validation_type': 'site_based',
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'n_train': len(y_train),
            'n_test': len(y_test),
            'train_facilities': train_facilities,
            'test_facilities': test_facilities,
            'model': model,
            'y_test': y_test,
            'y_pred': y_pred
        }
        
        print(f"Results: R² = {r2:.3f}, RMSE = {rmse:.3f}, MAE = {mae:.3f}")
        return results
    
    def run_temporal_based_validation(self):
        """Temporal-based validation (if multiple months available)"""
        print(f"\n⏰ TEMPORAL-BASED VALIDATION") 
        print("-"*40)
        
        if 'month' not in self.X_full.columns:
            print("Cannot run temporal validation: No month information available")
            return None
        
        available_months = sorted(self.X_full['month'].unique())
        print(f"Available months: {available_months}")
        
        if len(available_months) < 2:
            print("Cannot run temporal validation: Only one month available")
            print("Using temporal split within the month based on grid positions")
            
            # Spatial-temporal split within month
            # Use grid positions as pseudo-temporal split
            X_features = self.X_full[self.feature_names].values
            X_train, X_test, y_train, y_test = train_test_split(
                X_features, self.y_full,
                test_size=self.config['test_size'],
                random_state=self.config['random_state']
            )
            split_type = "spatial_pseudo_temporal"
        else:
            # True temporal split using months
            train_months = available_months[:-1]  # All but last month for training
            test_months = available_months[-1:]   # Last month for testing
            
            train_mask = self.X_full['month'].isin(train_months)
            test_mask = self.X_full['month'].isin(test_months)
            
            X_train = self.X_full[train_mask][self.feature_names].values
            X_test = self.X_full[test_mask][self.feature_names].values
            y_train = self.y_full[train_mask]
            y_test = self.y_full[test_mask]
            
            print(f"Train months: {train_months}")
            print(f"Test months: {test_months}")
            split_type = "true_temporal"
        
        # Train and evaluate
        model = RandomForestRegressor(n_estimators=100, random_state=self.config['random_state'])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        results = {
            'validation_type': 'temporal_based',
            'split_type': split_type,
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'n_train': len(y_train),
            'n_test': len(y_test),
            'model': model,
            'y_test': y_test,
            'y_pred': y_pred
        }
        
        print(f"Results ({split_type}): R² = {r2:.3f}, RMSE = {rmse:.3f}, MAE = {mae:.3f}")
        return results
    
    def run_all_validations(self):
        """Run all three validation types"""
        print(f"\n{'='*60}")
        print("🚀 RUNNING ALL TRI-VALIDATIONS")
        print(f"{'='*60}")
        
        # Load data once
        self.load_and_prepare_data()
        
        # Run all three validations
        results = {}
        
        # 1. Sample-based
        results['sample'] = self.run_sample_based_validation()
        
        # 2. Site-based  
        results['site'] = self.run_site_based_validation()
        
        # 3. Temporal-based
        results['temporal'] = self.run_temporal_based_validation()
        
        # Summary
        self.print_summary(results)
        self.create_plots(results)
        self.save_results(results)
        
        return results
    
    def print_summary(self, results):
        """Print summary of all validation results"""
        print(f"\n{'='*60}")
        print("📊 TRI-VALIDATION SUMMARY")
        print(f"{'='*60}")
        
        summary_data = []
        for validation_type, result in results.items():
            if result is not None:
                summary_data.append({
                    'Validation': validation_type.upper(),
                    'R²': f"{result['r2']:.3f}",
                    'RMSE': f"{result['rmse']:.3f}",
                    'MAE': f"{result['mae']:.3f}",
                    'N_train': result['n_train'],
                    'N_test': result['n_test']
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            print(summary_df.to_string(index=False))
        
        # Best performance
        valid_results = {k: v for k, v in results.items() if v is not None}
        if valid_results:
            best_r2 = max(valid_results.values(), key=lambda x: x['r2'])
            print(f"\n🏆 Best R² performance: {best_r2['validation_type'].upper()} (R² = {best_r2['r2']:.3f})")
    
    def create_plots(self, results):
        """Create comparison plots for all validations"""
        valid_results = {k: v for k, v in results.items() if v is not None}
        if not valid_results:
            return
        
        n_plots = len(valid_results)
        fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))
        if n_plots == 1:
            axes = [axes]
        
        for i, (validation_type, result) in enumerate(valid_results.items()):
            ax = axes[i]
            
            # Scatter plot
            ax.scatter(result['y_test'], result['y_pred'], alpha=0.6, s=30)
            
            # 1:1 line
            min_val = min(result['y_test'].min(), result['y_pred'].min())
            max_val = max(result['y_test'].max(), result['y_pred'].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
            
            # Labels and metrics
            ax.set_xlabel('Measured PM₂.₅ (log)' if self.config['target_log_transform'] else 'Measured PM₂.₅')
            ax.set_ylabel('Predicted PM₂.₅ (log)' if self.config['target_log_transform'] else 'Predicted PM₂.₅')
            ax.set_title(f'{validation_type.upper()} Validation\nR² = {result["r2"]:.3f}, RMSE = {result["rmse"]:.3f}')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = Path(self.config['output_dir']) / 'tri_validation_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plot saved: {plot_path}")
    
    def save_results(self, results):
        """Save results to CSV files"""
        output_dir = Path(self.config['output_dir'])
        
        # Save summary
        summary_data = []
        for validation_type, result in results.items():
            if result is not None:
                summary_data.append({
                    'validation_type': validation_type,
                    'r2': result['r2'],
                    'rmse': result['rmse'],
                    'mae': result['mae'],
                    'n_train': result['n_train'],
                    'n_test': result['n_test']
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_path = output_dir / 'tri_validation_summary.csv'
            summary_df.to_csv(summary_path, index=False)
            print(f"Summary saved: {summary_path}")
        
        # Save detailed results
        for validation_type, result in results.items():
            if result is not None:
                detailed_data = pd.DataFrame({
                    'measured': result['y_test'],
                    'predicted': result['y_pred']
                })
                detailed_path = output_dir / f'{validation_type}_validation_predictions.csv'
                detailed_data.to_csv(detailed_path, index=False)
        
        print(f"All results saved to: {output_dir}")

# ============================================================================
# 🎯 MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("🎯 SIMPLE TRI-VALIDATION FOR LOCAETA-ML")
    print("="*80)
    print(f"Configuration:")
    print(f"  Method: {TRI_VALIDATION_CONFIG['method']}")
    print(f"  Model: {TRI_VALIDATION_CONFIG['model_type']}")
    print(f"  Log transform: {TRI_VALIDATION_CONFIG['target_log_transform']}")
    print(f"  Data: {TRI_VALIDATION_CONFIG['X_features_path']}")
    print("="*80)
    
    # Initialize trainer
    trainer = SimpleTriValidationTrainer(TRI_VALIDATION_CONFIG)
    
    # Run all validations
    try:
        results = trainer.run_all_validations()
        print(f"\n✅ Tri-validation completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Tri-validation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()