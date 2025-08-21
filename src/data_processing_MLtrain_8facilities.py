import pandas as pd
import numpy as np
import netCDF4 as nc
import warnings
warnings.filterwarnings('ignore')

# Import the existing processor and path utilities
from data_processing_MLtrain_multifacility import MultiFacilityAirQualityProcessor
from utils.path_util import DATA_PATH

# Scale factors for each facility (exactly as you provided)
FACILITY_SCALE_FACTORS = {
    'suncor': 1.0,
    'bluespruce': 12.7001/16.93,
    'cherokee': 2.0001/29.49, 
    'cig': 1.0,
    'coors': 1.0,
    'denversteam': 1.0,
    'fortstvrain': 30.38/58.6651,
    'rmbc': 1.0,
    'rmec': 32.4075/17
}

# Base scaling factor and Suncor additional scale
BASE_SCALING_FACTOR = 907184740000  # uston2miug conversion
SUNCOR_ADDITIONAL_SCALE = 5.711149751263955  # Additional scaling for Suncor

class HYSPLITSingleMonthProcessor(MultiFacilityAirQualityProcessor):
    """
    Clean processor for single-month HYSPLIT data with scale factors
    """
    
    def __init__(self, grid_size: int = 24):
        """Initialize processor"""
        super().__init__(grid_size=grid_size)
        self.month = 3  # March 2023
        self.year = 2023
        
        # Define paths using DATA_PATH
        self.facility_info_csv = f"{DATA_PATH}/processed_data/metadata/facility_info.csv"
        self.netcdf_directory = f"{DATA_PATH}/HYSPLIT_Output/climate_trace_9"
        
        # Load facility information from CSV
        self.facility_info = self.load_facility_info()
        
    def load_facility_info(self) -> pd.DataFrame:
        """Load facility info from CSV and filter for March"""
        try:
            facility_info = pd.read_csv(self.facility_info_csv)
            print(f"Loaded facility info from: {self.facility_info_csv}")
            
            # Filter for March data (month=3)
            march_data = facility_info[facility_info['month'] == self.month].copy()
            print(f"March data: {len(march_data)} facilities")
            
            return march_data
            
        except FileNotFoundError:
            print(f"Facility info file not found: {self.facility_info_csv}")
            print("Using empty facility info - will extract from NetCDF files")
            return pd.DataFrame()
        except Exception as e:
            print(f"Error loading facility info: {e}")
            return pd.DataFrame()
    
    def discover_netcdf_files(self) -> dict:
        """Find all NetCDF files in HYSPLIT_Output/climate_trace_9 directory"""
        facility_files = {}
        
        import glob
        nc_pattern = f"{self.netcdf_directory}/pm25_*.nc"
        nc_files = glob.glob(nc_pattern)
        
        print(f"\nSearching in: {self.netcdf_directory}")
        print(f"Found {len(nc_files)} NetCDF files:")
        
        for file_path in nc_files:
            filename = file_path.split('/')[-1]  # Get filename from full path
            filename_stem = filename.replace('.nc', '')  # Remove extension
            parts = filename_stem.split('_')
            
            if len(parts) >= 2:
                facility_name = parts[1].lower()  # Get facility name and convert to lowercase
                
                # Handle name variations
                if 'fortst' in facility_name or 'fortstv' in facility_name:
                    facility_name = 'fortstvrain'
                
                # Check if this facility has a scale factor
                if facility_name in FACILITY_SCALE_FACTORS:
                    facility_files[facility_name] = file_path
                    print(f"  ✓ {facility_name}: {filename}")
                else:
                    print(f"  ? Unknown facility (no scale factor): {facility_name} in {filename}")
                    print(f"    Available scale factors: {list(FACILITY_SCALE_FACTORS.keys())}")
            else:
                print(f"  ✗ Could not parse: {filename}")
        
        return facility_files
    
    def get_scale_factor(self, facility_name: str) -> float:
        """Get total scale factor for facility"""
        facility_key = facility_name.lower()
        scale_multiplier = FACILITY_SCALE_FACTORS.get(facility_key, 1.0)
        
        # Apply additional Suncor scaling
        if facility_key == 'suncor':
            scale_multiplier *= SUNCOR_ADDITIONAL_SCALE
        
        total_scale = BASE_SCALING_FACTOR * scale_multiplier
        return total_scale
    
    def create_facility_metadata(self, facility_files: dict) -> pd.DataFrame:
        """Create metadata combining CSV data with NetCDF coordinates"""
        metadata_list = []
        
        print(f"\nCreating facility metadata:")
        
        for facility_name, file_path in facility_files.items():
            try:
                # Try to get info from CSV first
                if len(self.facility_info) > 0:
                    # Look for this facility in CSV (case-insensitive)
                    csv_matches = self.facility_info[
                        self.facility_info['name'].str.lower() == facility_name.lower()
                    ]
                    
                    if len(csv_matches) > 0:
                        csv_row = csv_matches.iloc[0]
                        lat = float(csv_row['lat'])
                        lon = float(csv_row['lon'])
                        height = float(csv_row['height'])
                        nei_emission = float(csv_row['NEI_annual_emission_t'])
                        activity = float(csv_row['activity'])
                        
                        print(f"  ✓ {facility_name}: Using CSV data")
                    else:
                        # Fallback to NetCDF extraction
                        lat, lon, height, nei_emission, activity = self.extract_from_netcdf(file_path)
                        print(f"  ✓ {facility_name}: Using NetCDF data (no CSV match)")
                else:
                    # No CSV data, extract from NetCDF
                    lat, lon, height, nei_emission, activity = self.extract_from_netcdf(file_path)
                    print(f"  ✓ {facility_name}: Using NetCDF data (no CSV file)")
                
                # Get scale factor
                scale_factor = self.get_scale_factor(facility_name)
                
                # Calculate monthly emission rate (consistent with original code)
                monthly_emission_rate = nei_emission / (365.25 * 24) * 12
                
                metadata_list.append({
                    'name': facility_name,
                    'lat': lat,
                    'lon': lon,
                    'height': height,
                    'NEI_annual_emission_t': nei_emission,
                    'scale_factor': scale_factor,
                    'month': self.month,
                    'activity': activity,
                    'monthly_emission_rate_t_per_hr': monthly_emission_rate,
                    'file_path': file_path
                })
                
                print(f"    Coords: ({lat:.5f}, {lon:.5f}, {height}m)")
                print(f"    Scale factor: {scale_factor:.2e}")
                
            except Exception as e:
                print(f"  ✗ Error processing {facility_name}: {e}")
                continue
        
        return pd.DataFrame(metadata_list)
    
    def extract_from_netcdf(self, file_path: str) -> tuple:
        """Extract basic info from NetCDF file"""
        try:
            with nc.Dataset(file_path, 'r') as ds:
                if 'origins' in ds.variables:
                    origins = ds.variables['origins'][:]
                    lat = float(origins[0])
                    lon = float(origins[1])
                    height = float(origins[2]) if len(origins) > 2 else 100.0
                else:
                    lat, lon, height = 39.7392, -104.9903, 100.0  # Denver default
                
                nei_emission = 100.0  # Default
                activity = 1.0  # Default
                
                return lat, lon, height, nei_emission, activity
        except:
            return 39.7392, -104.9903, 100.0, 100.0, 1.0
    
    def process_all_facilities(self):
        """Main processing function"""
        print("="*60)
        print("PROCESSING SINGLE-MONTH HYSPLIT DATA")
        print("="*60)
        
        # Discover files
        facility_files = self.discover_netcdf_files()
        if not facility_files:
            print("No NetCDF files found!")
            return {}
        
        # Create metadata
        metadata = self.create_facility_metadata(facility_files)
        if len(metadata) == 0:
            print("No valid facility metadata created!")
            return {}
        
        # Set up facilities for parent class
        self.facilities = {}
        self.facility_metadata = metadata
        for _, row in metadata.iterrows():
            self.facilities[row['name']] = {
                'coords': (row['lat'], row['lon'], row['height']),
                'scale_factor': row['scale_factor'],
                'NEI_annual_emission_t': row['NEI_annual_emission_t']
            }
        
        # Load shared data (if available)
        print(f"\n{'='*60}")
        print("LOADING SHARED DATA")
        print(f"{'='*60}")
        
        try:
            satellite_data = self.load_satellite_data(self.year)
            meteorology_data = self.load_meteorology_data()
            print("✓ Loaded atmospheric data")
        except:
            print("Warning: Using dummy atmospheric data")
            satellite_data = {}
            meteorology_data = None
        
        try:
            elevation_data = self.load_elevation_data()
            landcover_data = self.load_landcover_data()
            roads_data = self.load_roads_data()
            print("✓ Loaded topographical data")
        except:
            print("Warning: No topographical data")
            elevation_data = None
            landcover_data = None
            roads_data = None
        
        # Process each facility
        print(f"\n{'='*60}")
        print("PROCESSING FACILITIES")
        print(f"{'='*60}")
        
        all_results = []
        
        for facility_name, file_path in facility_files.items():
            print(f"\n{'-'*40}")
            print(f"Processing: {facility_name.upper()}")
            print(f"{'-'*40}")
            
            try:
                result = self.process_single_facility(
                    facility_name, file_path, satellite_data, meteorology_data,
                    elevation_data, landcover_data, roads_data
                )
                
                if result:
                    # Add facility ID
                    for key, df in result.items():
                        df['facility_id'] = facility_name
                    
                    all_results.append(result)
                    print(f"✓ Processed {len(result['X_features'])} samples")
                else:
                    print(f"✗ Failed to process")
                    
            except Exception as e:
                print(f"✗ Error: {e}")
                continue
        
        # Combine results
        if all_results:
            print(f"\n{'='*60}")
            print("COMBINING RESULTS")
            print(f"{'='*60}")
            
            combined = {
                'X_features': pd.concat([r['X_features'] for r in all_results], ignore_index=True),
                'y_method1': pd.concat([r['y_method1'] for r in all_results], ignore_index=True),
                'y_method2': pd.concat([r['y_method2'] for r in all_results], ignore_index=True),
                'y_method3': pd.concat([r['y_method3'] for r in all_results], ignore_index=True)
            }
            
            self.print_summary(combined)
            return combined
        else:
            print("✗ No facilities processed successfully")
            return {}
    
    def process_single_facility(self, facility_name, file_path, satellite_data, 
                              meteorology_data, elevation_data, landcover_data, roads_data):
        """Process a single facility"""
        
        # Load HYSPLIT data
        try:
            hysplit_ds = nc.Dataset(file_path, 'r')
            hysplit_data = {self.month: hysplit_ds}
        except Exception as e:
            print(f"Failed to load NetCDF: {e}")
            return None
        
        # Set facility info for parent class methods
        facility_row = self.facility_metadata[self.facility_metadata['name'] == facility_name].iloc[0]
        self.facility_lat = facility_row['lat']
        self.facility_lon = facility_row['lon'] 
        self.facility_height = facility_row['height']
        self.scale_factor = facility_row['scale_factor']
        self.NEI_annual_emission = facility_row['NEI_annual_emission_t']
        
        # Get grid coordinates
        target_coords = self.get_target_grid_coordinates(hysplit_data)
        if len(target_coords) == 0:
            print("No target coordinates")
            hysplit_ds.close()
            return None
        
        # Extract features
        facility_features = self.extract_facility_features_simple(target_coords, facility_row)
        
        feature_list = [('facility', facility_features)]
        
        # Add other features if available
        if satellite_data and meteorology_data is not None:
            try:
                sat_features = self.extract_satellite_features_grid(satellite_data, target_coords, meteorology_data)
                met_features = self.extract_meteorology_features_grid(meteorology_data, target_coords)
                feature_list.extend([('satellite', sat_features), ('meteorology', met_features)])
            except Exception as e:
                print(f"Warning: Could not extract atmospheric features: {e}")
        
        if elevation_data is not None:
            try:
                elev_features = self.extract_elevation_features(elevation_data, target_coords)
                land_features = self.extract_landcover_features(landcover_data, target_coords)
                road_features = self.extract_roads_features(roads_data, target_coords)
                feature_list.extend([('elevation', elev_features), ('landcover', land_features), ('roads', road_features)])
            except Exception as e:
                print(f"Warning: Could not extract topographical features: {e}")
        
        # Combine features
        X_features = self.combine_features(feature_list)
        
        # Extract targets with proper scaling
        y_method1 = self.extract_targets_scaled(hysplit_data, method="method1")
        y_method2 = self.extract_targets_scaled(hysplit_data, method="method2")  
        y_method3 = self.extract_targets_scaled(hysplit_data, method="method3")
        
        # Clean data
        X_features, y_method1, y_method2, y_method3 = self.clean_data(X_features, y_method1, y_method2, y_method3)
        
        # Close dataset
        hysplit_ds.close()
        
        if len(X_features) == 0:
            print("No valid samples after cleaning")
            return None
        
        return {
            'X_features': X_features,
            'y_method1': y_method1,
            'y_method2': y_method2,
            'y_method3': y_method3
        }
    
    def extract_facility_features_simple(self, target_coords, facility_row):
        """Extract facility features"""
        features_list = []
        
        for _, row in target_coords.iterrows():
            # Calculate distance and bearing
            fwd_azimuth, back_azimuth, distance_m = self.geodesic.inv(
                self.facility_lon, self.facility_lat, row['lon'], row['lat']
            )
            
            distance_km = distance_m / 1000.0
            bearing = (fwd_azimuth + 360) % 360
            
            features_list.append({
                'month': self.month,
                'grid_i': row['grid_i'],
                'grid_j': row['grid_j'],
                'facility_lat': self.facility_lat,
                'facility_lon': self.facility_lon,
                'facility_height': self.facility_height,
                'distance_to_facility': distance_km,
                'bearing_from_facility': bearing,
                'NEI_annual_emission_t': facility_row['NEI_annual_emission_t'],
                'monthly_emission_rate_t_per_hr': facility_row['monthly_emission_rate_t_per_hr']
            })
        
        return pd.DataFrame(features_list)
    
    def extract_targets_scaled(self, hysplit_data, method="method1"):
        """Extract target values with proper scaling"""
        targets_list = []
        
        for month, hysplit_ds in hysplit_data.items():
            pm25_values = hysplit_ds.variables['pm25'][0, 0, :, :]  # First time and level
            lats = hysplit_ds.variables['latitude'][:]
            lons = hysplit_ds.variables['longitude'][:]
            
            # Grid subsetting (same as parent class)
            center_lat_idx = len(lats) // 2
            center_lon_idx = len(lons) // 2
            half_size = self.grid_size // 2
            
            lat_start = center_lat_idx - half_size
            lat_end = center_lat_idx + half_size
            lon_start = center_lon_idx - half_size
            lon_end = center_lon_idx + half_size
            
            lats_subset = lats[lat_start:lat_end]
            lons_subset = lons[lon_start:lon_end]
            pm25_subset = pm25_values[lat_start:lat_end, lon_start:lon_end]
            
            # Apply scaling based on method
            if method == "method1" or method == "method2":
                total_scale = self.scale_factor
            elif method == "method3":
                total_scale = self.scale_factor * (7 * 24)  # Weekly accumulation
            else:
                total_scale = self.scale_factor
            
            for i, lat in enumerate(lats_subset):
                for j, lon in enumerate(lons_subset):
                    pm25_val = float(pm25_subset[i, j])
                    
                    if np.ma.is_masked(pm25_val) or not np.isfinite(pm25_val):
                        pm25_val = np.nan
                    elif not np.isnan(pm25_val):
                        pm25_val = pm25_val * total_scale
                    
                    targets_list.append({
                        'month': month,
                        'grid_i': i,
                        'grid_j': j,
                        'lat': float(lat),
                        'lon': float(lon),
                        'pm25_concentration': pm25_val
                    })
        
        df = pd.DataFrame(targets_list)
        
        # Print statistics
        valid_values = df['pm25_concentration'].dropna()
        if len(valid_values) > 0:
            print(f"{method}: {valid_values.min():.2e} to {valid_values.max():.2e} (scale: {total_scale:.2e})")
        
        return df
    
    def print_summary(self, combined_results):
        """Print summary statistics"""
        X_features = combined_results['X_features']
        
        print(f"✓ Total samples: {len(X_features):,}")
        print(f"✓ Total features: {len(X_features.columns)-1}")
        print(f"✓ Facilities: {list(X_features['facility_id'].unique())}")
        
        # Show scale factors used
        print(f"\n=== SCALE FACTORS USED ===")
        for facility in X_features['facility_id'].unique():
            scale_factor = self.get_scale_factor(facility)
            base_multiplier = scale_factor / BASE_SCALING_FACTOR
            print(f"{facility}: {base_multiplier:.6f} (Total: {scale_factor:.2e})")
        
        # Show target statistics
        print(f"\n=== TARGET STATISTICS ===")
        for method in ['method1', 'method2', 'method3']:
            y_data = combined_results[f'y_{method}']['pm25_concentration']
            valid_data = y_data.dropna()
            if len(valid_data) > 0:
                print(f"{method}: {valid_data.min():.2e} to {valid_data.max():.2e}")
    
    def save_results(self, combined_results, output_dir=None):
        """Save processed data"""
        if output_dir is None:
            output_dir = f"{DATA_PATH}/processed_data/climate_trace_9"
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        X_features = combined_results['X_features']
        
        # Save combined data
        X_features.to_csv(f"{output_dir}/X_features_all_facilities_mar2023.csv", index=False)
        combined_results['y_method1'].to_csv(f"{output_dir}/y_method1_all_facilities_mar2023.csv", index=False)
        combined_results['y_method2'].to_csv(f"{output_dir}/y_method2_all_facilities_mar2023.csv", index=False)
        combined_results['y_method3'].to_csv(f"{output_dir}/y_method3_all_facilities_mar2023.csv", index=False)
        
        # Save by facility
        facility_dir = f"{output_dir}/by_facility"
        os.makedirs(facility_dir, exist_ok=True)
        
        for facility_name in X_features['facility_id'].unique():
            mask = X_features['facility_id'] == facility_name
            facility_X = X_features[mask].reset_index(drop=True)
            
            facility_X.to_csv(f"{facility_dir}/X_features_{facility_name}_mar2023.csv", index=False)
            
            for method in ['method1', 'method2', 'method3']:
                facility_y = combined_results[f'y_{method}'][mask].reset_index(drop=True)
                facility_y.to_csv(f"{facility_dir}/y_{method}_{facility_name}_mar2023.csv", index=False)
        
        # Save summary
        summary = {
            'processing_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_samples': len(X_features),
            'total_features': len(X_features.columns) - 1,
            'facilities_processed': list(X_features['facility_id'].unique()),
            'month': self.month,
            'year': self.year
        }
        
        pd.DataFrame([summary]).to_csv(f"{output_dir}/processing_summary.csv", index=False)
        
        print(f"\n✓ Data saved to: {output_dir}")


def main():
    """Main function"""
    print("="*70)
    print("HYSPLIT DATA PROCESSOR - CLIMATE TRACE 9 FACILITIES")
    print("="*70)
    
    # Initialize processor (paths are handled internally)
    processor = HYSPLITSingleMonthProcessor(grid_size=24)
    
    # Check paths
    print(f"Facility info path: {processor.facility_info_csv}")
    print(f"NetCDF directory: {processor.netcdf_directory}")
    
    # Process all facilities
    results = processor.process_all_facilities()
    
    if results:
        # Save results
        processor.save_results(results)
        print("\n✓ Processing completed successfully!")
        print(f"\nProcessed facilities with scale factors:")
        for facility in results['X_features']['facility_id'].unique():
            scale_factor = processor.get_scale_factor(facility)
            base_multiplier = scale_factor / BASE_SCALING_FACTOR
            print(f"  {facility}: {base_multiplier:.6f}")
    else:
        print("\n✗ Processing failed!")
        print("\nCheck that you have:")
        print(f"1. Facility info CSV: {DATA_PATH}/processed_data/metadata/facility_info.csv")
        print(f"2. NetCDF files in: {DATA_PATH}/HYSPLIT_Output/climate_trace_9/")
        print("3. Scale factors defined for all facility names")


if __name__ == "__main__":
    main()