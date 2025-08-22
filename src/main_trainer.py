"""
Main training script that consolidates all training functionality
"""
import pandas as pd
from trainers.configurable_trainer import ConfigurableMLTrainer
from trainers.multi_facility_trainer import MultiFacilityModelTrainer
from simple_tri_validation import SimpleTriValidationTrainer, TRI_VALIDATION_CONFIG
from utils.path_util import DATA_PATH, MODELS_PATH

# ============================================================================
# 🎯 CONFIGURATION - MODIFY PATHS AND SETTINGS HERE
# ============================================================================

DATA_PATHS = {
    'X_features': f"{DATA_PATH}/processed_data/combined_large/X_features_combined_all_facilities.csv",
    'y_method1': f"{DATA_PATH}/processed_data/combined_large/y_method1_combined_all_facilities.csv",
    'y_method2': f"{DATA_PATH}/processed_data/combined_large/y_method2_combined_all_facilities.csv", 
    'y_method3': f"{DATA_PATH}/processed_data/combined_large/y_method3_combined_all_facilities.csv",
    'output_dir': f"{MODELS_PATH}/results"
}

# Multi-facility training config (updated to use combined_large data)
MULTI_FACILITY_CONFIG = {
    'facilities': ['suncor', 'rmbc', 'bluespruce', 'cherokee', 'cig', 'coors', 'denversteam', 'fortstvrain', 'rmec'],  # All 9 facilities
    'year': 2023,
    'grid_size': 24,
    'log_transform': True,
    'hyperparameter_tuning': False,  # Set to True for full hyperparameter search
    'use_combined_data': True,  # Use combined_large dataset instead of individual facility files
    'combined_data_paths': {
        'X_features': f"{DATA_PATH}/processed_data/combined_large/X_features_combined_all_facilities.csv",
        'y_method1': f"{DATA_PATH}/processed_data/combined_large/y_method1_combined_all_facilities.csv"
    }
}

# Training configurations - Easy to modify!
TRAINING_CONFIGS = [
    {
        'method': 'method2',
        'target_log_transform': True,
        'validation_type': 'site_based',
        'model_type': 'RF',
        'test_size': 0.2,
        'random_state': 42,
        'figure_suffix': 'march_test',
        'test_facilities': None,  # Auto-select
        'generate_heatmaps': True,
        'heatmap_facilities': 'auto',
        'heatmap_months': [3],
        'heatmap_extent_km': 50,
        'heatmap_style': 'side_by_side',
    },
    {
        'method': 'method3', 
        'target_log_transform': True,
        'validation_type': 'site_based',
        'model_type': 'RF',
        'test_size': 0.2,
        'random_state': 42,
        'figure_suffix': 'march_test',
        'test_facilities': None,
        'generate_heatmaps': True,
        'heatmap_facilities': 'auto',
        'heatmap_months': [3],
        'heatmap_extent_km': 50,
        'heatmap_style': 'side_by_side',
    }
]

# ============================================================================
# 📋 FEATURE DEFINITIONS
# ============================================================================

def get_default_features():
    """Get default feature set"""
    return [
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
    ]

# ============================================================================
# 🚀 TRAINING FUNCTIONS
# ============================================================================

def run_configurable_training():
    """Run configurable training with enhanced features"""
    print("="*80)
    print("CONFIGURABLE TRAINING")
    print("="*80)
    print(f"Data source: {DATA_PATHS['X_features']}")
    print(f"Output directory: {DATA_PATHS['output_dir']}")
    print("="*80)
    
    features = get_default_features()
    
    # Prepare configurations
    configs = []
    for base_config in TRAINING_CONFIGS:
        config = base_config.copy()
        config['features'] = features
        config['output_dir'] = DATA_PATHS['output_dir']
        configs.append(config)
    
    # Run batch processing
    for i, config in enumerate(configs):
        print(f"\n{'='*80}")
        print(f"Running Configuration {i+1}/{len(configs)}")
        print(f"Model: {config['model_type']} - Method: {config['method']} - Validation: {config['validation_type']}")
        print(f"Suffix: {config.get('figure_suffix', 'None')}")
        print(f"Heatmaps: {config.get('generate_heatmaps', False)}")
        method_key = f"y_{config['method']}"
        print(f"Data: {DATA_PATHS[method_key]}")
        print(f"{'='*80}")
        
        try:
            # Initialize trainer
            trainer = ConfigurableMLTrainer(config)
            
            # Load and prepare data
            method_key = f"y_{config['method']}"
            X, y = trainer.load_and_prepare_data(
                DATA_PATHS['X_features'], 
                DATA_PATHS[method_key]
            )
            
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


def run_multi_facility_training():
    """Run multi-facility training with spatial validation"""
    print("="*80)
    print("MULTI-FACILITY MODEL TRAINING WITH SPATIAL VALIDATION")
    print("="*80)
    print(f"Facilities: {MULTI_FACILITY_CONFIG['facilities']}")
    print(f"Data source: combined_large (25,920 samples, 12 months)")
    print(f"Using combined dataset: {MULTI_FACILITY_CONFIG['use_combined_data']}")
    print("="*80)
    
    # Initialize trainer
    trainer = MultiFacilityModelTrainer(random_state=42)
    
    try:
        if MULTI_FACILITY_CONFIG['use_combined_data']:
            # Load from combined dataset directly
            print("Loading from combined_large dataset...")
            X_features = pd.read_csv(MULTI_FACILITY_CONFIG['combined_data_paths']['X_features'])
            y_target = pd.read_csv(MULTI_FACILITY_CONFIG['combined_data_paths']['y_method1'])
            
            print(f"Loaded combined data:")
            print(f"  X_features shape: {X_features.shape}")
            print(f"  y_target shape: {y_target.shape}")
            
            if 'facility_id' in X_features.columns:
                facilities = sorted(X_features['facility_id'].unique())
                print(f"  Facilities in data: {facilities}")
                
            if 'month' in X_features.columns:
                months = sorted(X_features['month'].unique())
                print(f"  Months in data: {months}")
        else:
            # Load multi-facility data (original method)
            X_features, y_target = trainer.load_multi_facility_data(
                facilities=MULTI_FACILITY_CONFIG['facilities'],
                year=MULTI_FACILITY_CONFIG['year'],
                grid_size=MULTI_FACILITY_CONFIG['grid_size']
            )
        
        # Create spatial validation splits
        X_train, X_val, X_test, y_train, y_val, y_test = trainer.create_spatial_validation_splits(
            X_features, y_target,
            test_size=0.2,
            val_size=0.2
        )
        
        # Prepare features and targets
        X_train_processed, y_train_processed = trainer.prepare_features_targets(
            X_train, y_train, 
            log_transform_targets=MULTI_FACILITY_CONFIG['log_transform']
        )
        
        print("\n✓ Multi-facility training setup completed")
        
    except Exception as e:
        print(f"✗ Multi-facility training failed: {e}")
        import traceback
        traceback.print_exc()


def run_tri_validation():
    """Run simplified tri-validation (Sample + Site + Time based)"""
    print("="*80)
    print("TRI-VALIDATION: SAMPLE + SITE + TIME BASED")
    print("="*80)
    print(f"Method: {TRI_VALIDATION_CONFIG['method']}")
    print(f"Model: {TRI_VALIDATION_CONFIG['model_type']}")
    print(f"Log transform: {TRI_VALIDATION_CONFIG['target_log_transform']}")
    print(f"Data: {TRI_VALIDATION_CONFIG['X_features_path']}")
    print("="*80)
    
    try:
        # Initialize tri-validation trainer
        trainer = SimpleTriValidationTrainer(TRI_VALIDATION_CONFIG)
        
        # Run all three validation types
        results = trainer.run_all_validations()
        
        print(f"\n✓ Tri-validation completed successfully!")
        print(f"Results saved to: {TRI_VALIDATION_CONFIG['output_dir']}")
        
    except Exception as e:
        print(f"✗ Tri-validation failed: {e}")
        import traceback
        traceback.print_exc()


def print_configuration():
    """Print current configuration for verification"""
    print("="*80)
    print("🔧 CURRENT CONFIGURATION")
    print("="*80)
    
    print(f"\n📁 DATA PATHS (COMBINED_LARGE DATASET):")
    for key, path in DATA_PATHS.items():
        print(f"  {key}: {path}")
    
    print(f"\n📊 COMBINED_LARGE DATASET SUMMARY:")
    print(f"  Total samples: 25,920 (vs 5,184 in March-only)")
    print(f"  Facilities: 9 (bluespruce, cherokee, cig, coors, denversteam, fortstvrain, rmbc, rmec, suncor)")
    print(f"  Time coverage:")
    print(f"    - 12 months: suncor, rmbc, bluespruce (7,488 samples each)")
    print(f"    - March only: cherokee, cig, coors, denversteam, fortstvrain, rmec (576 samples each)")
    
    print(f"\n🏭 MULTI-FACILITY CONFIG:")
    for key, value in MULTI_FACILITY_CONFIG.items():
        if key == 'combined_data_paths':
            print(f"  {key}: [X_features + y_method1 paths]")
        else:
            print(f"  {key}: {value}")
    
    print(f"\n⚙️ TRAINING CONFIGS: {len(TRAINING_CONFIGS)} configurations")
    for i, config in enumerate(TRAINING_CONFIGS):
        print(f"  Config {i+1}: {config['model_type']} + {config['method']} + {config['validation_type']}")
    
    features = get_default_features()
    print(f"\n📊 FEATURES: {len(features)} features")
    for i, feature in enumerate(features[:5]):
        print(f"  {i+1}. {feature}")
    if len(features) > 5:
        print(f"  ... and {len(features)-5} more")
    
    print("="*80)


def main():
    """Main function to run training"""
    print("="*80)
    print("🎯 LOCAETA-ML CONSOLIDATED TRAINING SYSTEM")
    print("="*80)
    
    # Print configuration for verification
    print_configuration()
    
    # Ask user what to run
    print("\n🚀 AVAILABLE OPTIONS:")
    print("1. Configurable Training (Enhanced with heatmaps)")
    print("2. Multi-Facility Training (Spatial validation)")
    print("3. Tri-Validation (Sample + Site + Time based - RF + Method1 + Log)")
    print("4. All training options (1 + 2 + 3)")
    print("0. Just show configuration (no training)")
    
    try:
        choice = input("\nEnter your choice (0-4): ").strip()
        
        if choice == "1":
            run_configurable_training()
            print("✓ Configurable training completed")
            
        elif choice == "2":
            run_multi_facility_training()
            print("✓ Multi-facility training completed")
            
        elif choice == "3":
            print("\n3. Running Tri-Validation...")
            run_tri_validation()
            print("✓ Tri-validation completed")
            
        elif choice == "4":
            print("\n1. Running Configurable Training...")
            run_configurable_training()
            print("✓ Configurable training completed")
            
            print("\n2. Running Multi-Facility Training...")
            run_multi_facility_training()
            print("✓ Multi-facility training completed")
            
            print("\n3. Running Tri-Validation...")
            run_tri_validation()
            print("✓ Tri-validation completed")
            
        elif choice == "0":
            print("✓ Configuration displayed only")
            
        else:
            print("❌ Invalid choice. Please run again and select 0-4.")
            
    except KeyboardInterrupt:
        print("\n\n🛑 Training interrupted by user")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("🏁 TRAINING SESSION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()