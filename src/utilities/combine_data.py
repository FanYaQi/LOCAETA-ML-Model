import pandas as pd
import numpy as np
from pathlib import Path
import sys
## add sys path for local functions
path_to_add = '/Users/yaqifan/Documents/Github/LOCAETA-ML/src'
if path_to_add not in sys.path:
    sys.path.insert(0, path_to_add)
from utils.path_util import DATA_PATH
import warnings
warnings.filterwarnings('ignore')

class DataCombiner:
    """
    Combine two datasets with overlapping facilities while avoiding duplicates
    """
    
    def __init__(self, config):
        """
        Initialize data combiner
        
        config = {
            'dataset1_path': path to 12-month 3-facility data
            'dataset2_path': path to March 9-facility data
            'overlapping_facilities': ['suncor', 'rmbc', 'bluespruce']
            'overlap_month': 3  # March
            'output_path': where to save combined data
            'methods': ['method1', 'method2', 'method3']
        }
        """
        self.config = config
        self.dataset1_info = {}
        self.dataset2_info = {}
        self.combined_info = {}
        
        # Create output directory
        Path(config['output_path']).mkdir(parents=True, exist_ok=True)
        
    def analyze_datasets(self):
        """Analyze both datasets to understand structure and overlaps"""
        print("="*80)
        print("ANALYZING DATASETS")
        print("="*80)
        
        # Load dataset 1 (12-month, 3 facilities)
        print(f"\nLoading Dataset 1: {self.config['dataset1_path']}")
        dataset1_X = pd.read_csv(f"{self.config['dataset1_path']}/X_features_all_facilities_2023_grid24.csv")
        
        self.dataset1_info = {
            'X_shape': dataset1_X.shape,
            'facilities': sorted(dataset1_X['facility_id'].unique()),
            'months': sorted(dataset1_X['month'].unique()) if 'month' in dataset1_X.columns else ['unknown'],
            'unique_combinations': len(dataset1_X[['facility_id', 'month', 'grid_i', 'grid_j']].drop_duplicates())
        }
        
        print(f"  Shape: {self.dataset1_info['X_shape']}")
        print(f"  Facilities: {self.dataset1_info['facilities']}")
        print(f"  Months: {self.dataset1_info['months']}")
        print(f"  Unique location-time combinations: {self.dataset1_info['unique_combinations']}")
        
        # Load dataset 2 (March, 9 facilities)
        print(f"\nLoading Dataset 2: {self.config['dataset2_path']}")
        dataset2_X = pd.read_csv(f"{self.config['dataset2_path']}/X_features_all_facilities_mar2023.csv")
        
        self.dataset2_info = {
            'X_shape': dataset2_X.shape,
            'facilities': sorted(dataset2_X['facility_id'].unique()),
            'months': sorted(dataset2_X['month'].unique()) if 'month' in dataset2_X.columns else ['unknown'],
            'unique_combinations': len(dataset2_X[['facility_id', 'month', 'grid_i', 'grid_j']].drop_duplicates())
        }
        
        print(f"  Shape: {self.dataset2_info['X_shape']}")
        print(f"  Facilities: {self.dataset2_info['facilities']}")
        print(f"  Months: {self.dataset2_info['months']}")
        print(f"  Unique location-time combinations: {self.dataset2_info['unique_combinations']}")
        
        # Analyze overlaps
        overlapping_facilities = set(self.dataset1_info['facilities']) & set(self.dataset2_info['facilities'])
        dataset2_only_facilities = set(self.dataset2_info['facilities']) - set(self.dataset1_info['facilities'])
        
        print(f"\nOVERLAP ANALYSIS:")
        print(f"  Overlapping facilities: {sorted(overlapping_facilities)}")
        print(f"  Dataset 2 only facilities: {sorted(dataset2_only_facilities)}")
        print(f"  Expected overlapping facilities: {self.config['overlapping_facilities']}")
        
        # Verify expected overlaps
        expected_overlaps = set(self.config['overlapping_facilities'])
        if overlapping_facilities != expected_overlaps:
            print(f"  WARNING: Actual overlaps don't match expected!")
            print(f"    Missing from actual: {expected_overlaps - overlapping_facilities}")
            print(f"    Extra in actual: {overlapping_facilities - expected_overlaps}")
        
        return dataset1_X, dataset2_X
    
    def identify_duplicates(self, dataset1_X, dataset2_X):
        """Identify exact duplicate records between datasets"""
        print(f"\n{'-'*60}")
        print("IDENTIFYING DUPLICATES")
        print(f"{'-'*60}")
        
        # Create unique identifiers for each record
        id_cols = ['facility_id', 'month', 'grid_i', 'grid_j']
        
        # Check if all ID columns exist
        missing_cols_d1 = [col for col in id_cols if col not in dataset1_X.columns]
        missing_cols_d2 = [col for col in id_cols if col not in dataset2_X.columns]
        
        if missing_cols_d1 or missing_cols_d2:
            print(f"WARNING: Missing ID columns!")
            print(f"  Dataset 1 missing: {missing_cols_d1}")
            print(f"  Dataset 2 missing: {missing_cols_d2}")
            return pd.DataFrame(), 0, 0
        
        # Create unique identifiers
        dataset1_X['unique_id'] = dataset1_X[id_cols].apply(lambda x: '_'.join(map(str, x)), axis=1)
        dataset2_X['unique_id'] = dataset2_X[id_cols].apply(lambda x: '_'.join(map(str, x)), axis=1)
        
        # Find duplicates
        duplicate_ids = set(dataset1_X['unique_id']) & set(dataset2_X['unique_id'])
        
        print(f"Total unique records in Dataset 1: {len(dataset1_X['unique_id'].unique())}")
        print(f"Total unique records in Dataset 2: {len(dataset2_X['unique_id'].unique())}")
        print(f"Duplicate records found: {len(duplicate_ids)}")
        
        if len(duplicate_ids) > 0:
            # Analyze duplicate patterns
            duplicate_df = dataset1_X[dataset1_X['unique_id'].isin(duplicate_ids)][id_cols + ['unique_id']]
            duplicate_summary = duplicate_df.groupby(['facility_id', 'month']).size().reset_index(name='count')
            
            print(f"\nDuplicate breakdown by facility and month:")
            print(duplicate_summary.to_string(index=False))
            
            # Show sample duplicates
            print(f"\nSample duplicate records:")
            print(duplicate_df.head().to_string(index=False))
        
        return duplicate_df if len(duplicate_ids) > 0 else pd.DataFrame(), len(duplicate_ids), duplicate_ids
    
    def combine_features(self, dataset1_X, dataset2_X, duplicate_ids):
        """Combine X features from both datasets, removing duplicates"""
        print(f"\n{'-'*60}")
        print("COMBINING FEATURES")
        print(f"{'-'*60}")
        
        # Strategy: Keep all data from dataset1, add non-duplicate data from dataset2
        print("Strategy: Keep all Dataset 1 + non-duplicate data from Dataset 2")
        
        # Remove duplicates from dataset2
        dataset2_clean = dataset2_X[~dataset2_X['unique_id'].isin(duplicate_ids)].copy()
        
        print(f"Dataset 1 records to keep: {len(dataset1_X)}")
        print(f"Dataset 2 records after removing duplicates: {len(dataset2_clean)}")
        
        # Check column compatibility
        common_cols = set(dataset1_X.columns) & set(dataset2_clean.columns)
        dataset1_only_cols = set(dataset1_X.columns) - set(dataset2_clean.columns)
        dataset2_only_cols = set(dataset2_clean.columns) - set(dataset1_X.columns)
        
        print(f"\nColumn analysis:")
        print(f"  Common columns: {len(common_cols)}")
        print(f"  Dataset 1 only: {len(dataset1_only_cols)} {list(dataset1_only_cols)[:5]}{'...' if len(dataset1_only_cols) > 5 else ''}")
        print(f"  Dataset 2 only: {len(dataset2_only_cols)} {list(dataset2_only_cols)[:5]}{'...' if len(dataset2_only_cols) > 5 else ''}")
        
        # Handle column differences
        if dataset1_only_cols or dataset2_only_cols:
            print(f"\nHandling column differences...")
            
            # Add missing columns with NaN
            for col in dataset1_only_cols:
                if col != 'unique_id':  # Don't add this helper column
                    dataset2_clean[col] = np.nan
                    print(f"  Added column '{col}' to Dataset 2 (filled with NaN)")
            
            for col in dataset2_only_cols:
                if col != 'unique_id':  # Don't add this helper column
                    dataset1_X[col] = np.nan
                    print(f"  Added column '{col}' to Dataset 1 (filled with NaN)")
        
        # Combine datasets
        final_cols = sorted([col for col in common_cols | dataset1_only_cols | dataset2_only_cols if col != 'unique_id'])
        
        combined_X = pd.concat([
            dataset1_X[final_cols],
            dataset2_clean[final_cols]
        ], ignore_index=True)
        
        print(f"\nCombined dataset shape: {combined_X.shape}")
        
        # Summary statistics
        combined_facilities = sorted(combined_X['facility_id'].unique())
        combined_months = sorted(combined_X['month'].unique()) if 'month' in combined_X.columns else ['unknown']
        
        facility_counts = combined_X['facility_id'].value_counts().sort_index()
        
        print(f"Combined dataset summary:")
        print(f"  Total facilities: {len(combined_facilities)}")
        print(f"  Facilities: {combined_facilities}")
        print(f"  Months: {combined_months}")
        print(f"  Records per facility:")
        for facility in combined_facilities:
            count = facility_counts.get(facility, 0)
            months_for_facility = sorted(combined_X[combined_X['facility_id'] == facility]['month'].unique()) if 'month' in combined_X.columns else ['unknown']
            print(f"    {facility}: {count} records ({len(months_for_facility)} months: {months_for_facility})")
        
        return combined_X
    
    def combine_targets(self, duplicate_ids):
        """Combine Y targets from both datasets for all methods"""
        print(f"\n{'-'*60}")
        print("COMBINING TARGETS")
        print(f"{'-'*60}")
        
        combined_targets = {}
        
        for method in self.config['methods']:
            print(f"\nProcessing {method}...")
            
            # Load target files
            try:
                # Dataset 1: 12-month data
                y1_path = f"{self.config['dataset1_path']}/y_target_{method}_all_facilities_2023_grid24.csv"
                dataset1_y = pd.read_csv(y1_path)
                
                # Dataset 2: March data  
                y2_path = f"{self.config['dataset2_path']}/y_{method}_all_facilities_mar2023.csv"
                dataset2_y = pd.read_csv(y2_path)
                
                print(f"  Dataset 1 shape: {dataset1_y.shape}")
                print(f"  Dataset 2 shape: {dataset2_y.shape}")
                
            except FileNotFoundError as e:
                print(f"  ERROR: Target file not found for {method}: {e}")
                continue
            
            # Create unique identifiers (assuming same structure as X data)
            id_cols = ['facility_id', 'month', 'grid_i', 'grid_j']
            
            # Check if all ID columns exist in target files
            missing_cols_y1 = [col for col in id_cols if col not in dataset1_y.columns]
            missing_cols_y2 = [col for col in id_cols if col not in dataset2_y.columns]
            
            if missing_cols_y1 or missing_cols_y2:
                print(f"  WARNING: Missing ID columns in target files!")
                print(f"    Dataset 1 missing: {missing_cols_y1}")
                print(f"    Dataset 2 missing: {missing_cols_y2}")
                continue
            
            dataset1_y['unique_id'] = dataset1_y[id_cols].apply(lambda x: '_'.join(map(str, x)), axis=1)
            dataset2_y['unique_id'] = dataset2_y[id_cols].apply(lambda x: '_'.join(map(str, x)), axis=1)
            
            # Remove duplicates from dataset2
            dataset2_y_clean = dataset2_y[~dataset2_y['unique_id'].isin(duplicate_ids)].copy()
            
            print(f"  Dataset 1 targets to keep: {len(dataset1_y)}")
            print(f"  Dataset 2 targets after removing duplicates: {len(dataset2_y_clean)}")
            
            # Combine targets
            target_cols = [col for col in dataset1_y.columns if col != 'unique_id']
            combined_y = pd.concat([
                dataset1_y[target_cols],
                dataset2_y_clean[target_cols]
            ], ignore_index=True)
            
            print(f"  Combined {method} shape: {combined_y.shape}")
            
            combined_targets[method] = combined_y
        
        return combined_targets
    
    def save_combined_data(self, combined_X, combined_targets):
        """Save combined datasets to output directory"""
        print(f"\n{'-'*60}")
        print("SAVING COMBINED DATA")
        print(f"{'-'*60}")
        
        output_path = Path(self.config['output_path'])
        
        # Save combined features
        X_output_path = output_path / "X_features_combined_all_facilities.csv"
        combined_X.to_csv(X_output_path, index=False)
        print(f"Saved combined features: {X_output_path}")
        print(f"  Shape: {combined_X.shape}")
        
        # Save combined targets
        for method, combined_y in combined_targets.items():
            y_output_path = output_path / f"y_{method}_combined_all_facilities.csv"
            combined_y.to_csv(y_output_path, index=False)
            print(f"Saved combined {method}: {y_output_path}")
            print(f"  Shape: {combined_y.shape}")
        
        # Save summary report
        self._save_combination_report(combined_X, combined_targets, output_path)
        
        return X_output_path, {method: output_path / f"y_{method}_combined_all_facilities.csv" for method in combined_targets.keys()}
    
    def _save_combination_report(self, combined_X, combined_targets, output_path):
        """Save detailed combination report"""
        report_path = output_path / "data_combination_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("DATA COMBINATION REPORT\n")
            f.write("="*50 + "\n\n")
            
            f.write("CONFIGURATION:\n")
            f.write(f"Dataset 1 path: {self.config['dataset1_path']}\n")
            f.write(f"Dataset 2 path: {self.config['dataset2_path']}\n")
            f.write(f"Overlapping facilities: {self.config['overlapping_facilities']}\n")
            f.write(f"Output path: {self.config['output_path']}\n\n")
            
            f.write("DATASET INFORMATION:\n")
            f.write(f"Dataset 1 - Shape: {self.dataset1_info['X_shape']}, Facilities: {self.dataset1_info['facilities']}, Months: {self.dataset1_info['months']}\n")
            f.write(f"Dataset 2 - Shape: {self.dataset2_info['X_shape']}, Facilities: {self.dataset2_info['facilities']}, Months: {self.dataset2_info['months']}\n\n")
            
            f.write("COMBINED DATASET:\n")
            f.write(f"Features shape: {combined_X.shape}\n")
            f.write(f"Facilities: {sorted(combined_X['facility_id'].unique())}\n")
            f.write(f"Months: {sorted(combined_X['month'].unique()) if 'month' in combined_X.columns else ['unknown']}\n\n")
            
            f.write("RECORDS PER FACILITY:\n")
            facility_counts = combined_X['facility_id'].value_counts().sort_index()
            for facility in sorted(combined_X['facility_id'].unique()):
                count = facility_counts.get(facility, 0)
                months = sorted(combined_X[combined_X['facility_id'] == facility]['month'].unique()) if 'month' in combined_X.columns else ['unknown']
                f.write(f"  {facility}: {count} records ({len(months)} months: {months})\n")
            
            f.write(f"\nTARGET DATASETS:\n")
            for method, target_df in combined_targets.items():
                f.write(f"  {method}: {target_df.shape}\n")
        
        print(f"Saved combination report: {report_path}")
    
    def run_combination(self):
        """Run the complete data combination process"""
        print("STARTING DATA COMBINATION PROCESS")
        print("="*80)
        
        # Step 1: Analyze datasets
        dataset1_X, dataset2_X = self.analyze_datasets()
        
        # Step 2: Identify duplicates
        duplicate_df, num_duplicates, duplicate_ids = self.identify_duplicates(dataset1_X, dataset2_X)
        
        # Step 3: Combine features
        combined_X = self.combine_features(dataset1_X, dataset2_X, duplicate_ids)
        
        # Step 4: Combine targets
        combined_targets = self.combine_targets(duplicate_ids)
        
        # Step 5: Save combined data
        X_path, y_paths = self.save_combined_data(combined_X, combined_targets)
        
        print(f"\n{'='*80}")
        print("DATA COMBINATION COMPLETED!")
        print(f"{'='*80}")
        print(f"Combined features: {X_path}")
        print(f"Combined targets: {list(y_paths.values())}")
        print(f"Total records: {len(combined_X)}")
        print(f"Total facilities: {len(combined_X['facility_id'].unique())}")
        print(f"Duplicates removed: {num_duplicates}")
        
        return X_path, y_paths, combined_X, combined_targets


# Example usage
if __name__ == "__main__":
    # Configuration for data combination
    COMBINE_CONFIG = {
        'dataset1_path': f"{DATA_PATH}/processed_data/combined",  # 12-month 3-facility data
        'dataset2_path': f"{DATA_PATH}/processed_data/climate_trace_9",  # March 9-facility data
        'overlapping_facilities': ['suncor', 'rmbc', 'bluespruce'],
        'overlap_month': 3,  # March
        'output_path': f"{DATA_PATH}/processed_data/combined_large",
        'methods': ['method1', 'method2', 'method3']
    }
    
    # Run combination
    combiner = DataCombiner(COMBINE_CONFIG)
    X_path, y_paths, combined_X, combined_targets = combiner.run_combination()
    
    print(f"\nReady for enhanced experiments with {len(combined_X)} total samples!")