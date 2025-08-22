"""
Multi-facility data processor with enhanced target calculation methods
"""
import pandas as pd
import numpy as np
import netCDF4 as nc
from pathlib import Path
from typing import Dict, Tuple, List, Optional

from data_processing.base_processor import BaseAirQualityProcessor
from utils.path_util import DATA_PATH

# Base scaling factor for all facilities
USTON2MIUG = 907184740000
# Additional scaling factor for Suncor
SUNCOR_ADDITIONAL_SCALE = 5.711149751263955


class MultiFacilityAirQualityProcessor(BaseAirQualityProcessor):
    """
    Process satellite, meteorology, HYSPLIT, and topographical data for multiple facilities
    with three different target value calculation methods and monthly emission rates
    """
    
    def __init__(self, grid_size: int = 24):
        """Initialize processor with configurable grid size"""
        super().__init__(grid_size)
        self.facility_metadata = None
        
        print(f"Initialized multi-facility processor with {grid_size}x{grid_size} grid")
        
    def load_facility_metadata(self, metadata_file: str) -> pd.DataFrame:
        """Load facility metadata CSV"""
        try:
            metadata = pd.read_csv(metadata_file)
            required_cols = ['name', 'lat', 'lon', 'height', 'NEI_annual_emission_t', 'month', 'activity']
            
            missing_cols = [col for col in required_cols if col not in metadata.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            print(f"Loaded facility metadata: {metadata.shape}")
            print(f"Facilities: {metadata['name'].unique()}")
            print(f"Months per facility: {len(metadata['month'].unique())}")
            
            # Create facility configurations with scale factors
            self.facilities = {}
            for facility in metadata['name'].unique():
                facility_data = metadata[metadata['name'] == facility].iloc[0]
                
                # Set scale factor (Suncor gets additional correction)
                scale_factor = USTON2MIUG
                if facility.lower() == 'suncor':
                    scale_factor *= SUNCOR_ADDITIONAL_SCALE
                
                self.facilities[facility] = {
                    'coords': (facility_data['lat'], facility_data['lon'], facility_data['height']),
                    'scale_factor': scale_factor,
                    'NEI_annual_emission_t': facility_data['NEI_annual_emission_t']
                }
            
            # Calculate monthly emission rates (t/hr) using activity scaling
            metadata = self._add_monthly_emission_rates(metadata)
            
            self.facility_metadata = metadata
            
            print("\nFacility configurations:")
            for name, config in self.facilities.items():
                print(f"  {name}: lat={config['coords'][0]:.5f}, lon={config['coords'][1]:.5f}")
                print(f"    Scale factor: {config['scale_factor']:.2e}")
                print(f"    NEI annual: {config['NEI_annual_emission_t']:.1f} t/year")
            
            return metadata
            
        except Exception as e:
            print(f"Error loading facility metadata: {e}")
            raise
    
    def _add_monthly_emission_rates(self, metadata: pd.DataFrame) -> pd.DataFrame:
        """Add monthly emission rate column (t/hr) based on NEI annual emissions distributed by activity"""
        metadata = metadata.copy()
        
        # Hours per year
        hours_per_year = 365.25 * 24  # 8766 hours
        
        # Calculate monthly emission rates
        emission_rates = []
        
        # Group by facility to calculate activity distribution
        for facility in metadata['name'].unique():
            facility_data = metadata[metadata['name'] == facility].copy()
            
            # Calculate total activity sum for this facility
            total_activity = facility_data['activity'].sum()
            
            # Get annual NEI emission for this facility
            annual_emission = facility_data['NEI_annual_emission_t'].iloc[0]
            
            # Calculate monthly emissions distributed by activity
            for _, row in facility_data.iterrows():
                # Monthly emission = (activity / total_activity) * annual_emission
                monthly_emission_fraction = row['activity'] / total_activity
                monthly_emission_t_per_year = monthly_emission_fraction * annual_emission
                
                # Convert to hourly rate (assuming constant rate within month)
                monthly_hourly_rate = monthly_emission_t_per_year / hours_per_year * 12  # *12 to get monthly rate
                
                emission_rates.append(monthly_hourly_rate)
        
        metadata['monthly_emission_rate_t_per_hr'] = emission_rates
        
        print("\nMonthly emission rates calculated (activity-distributed):")
        for facility in metadata['name'].unique():
            facility_data = metadata[metadata['name'] == facility]
            rates = facility_data['monthly_emission_rate_t_per_hr']
            activity_vals = facility_data['activity']
            print(f"  {facility}:")
            print(f"    Activity range: {activity_vals.min():.2f} - {activity_vals.max():.2f}")
            print(f"    Emission rates: {rates.min():.4f} - {rates.max():.4f} t/hr")
            print(f"    Total annual check: {rates.sum() * hours_per_year / 12:.1f} t/year")
        
        return metadata
    
    def process_all_facilities(self, metadata_file: str, year: int = 2023, 
                             include_topographical: bool = True) -> Dict[str, pd.DataFrame]:
        """Process all facilities using metadata file"""
        # Load facility metadata
        print("="*60)
        print("LOADING FACILITY METADATA")
        print("="*60)
        metadata = self.load_facility_metadata(metadata_file)
        
        # Load shared data once (satellite, meteorology, topographical)
        print("\n" + "="*60)
        print("LOADING SHARED ATMOSPHERIC DATA")
        print("="*60)
        satellite_data = self.load_satellite_data(year)
        meteorology_data = self.load_meteorology_data()
        
        if include_topographical:
            print("\nLoading shared topographical data...")
            elevation_data = self.load_elevation_data()
            landcover_data = self.load_landcover_data()
            roads_data = self.load_roads_data()
        else:
            elevation_data = None
            landcover_data = None
            roads_data = None
        
        # Process each facility
        print("\n" + "="*60)
        print("PROCESSING INDIVIDUAL FACILITIES")
        print("="*60)
        
        all_X_features = []
        all_y_method1 = []
        all_y_method2 = []
        all_y_method3 = []
        
        for facility_name, facility_config in self.facilities.items():
            print(f"\n{'='*50}")
            print(f"Processing facility: {facility_name.upper()}")
            print(f"{'='*50}")
            
            try:
                # Get facility-specific metadata
                facility_metadata = metadata[metadata['name'] == facility_name].copy()
                
                # Process this facility
                facility_results = self.process_single_facility(
                    facility_name=facility_name,
                    facility_config=facility_config,
                    facility_metadata=facility_metadata,
                    year=year,
                    satellite_data=satellite_data,
                    meteorology_data=meteorology_data,
                    elevation_data=elevation_data,
                    landcover_data=landcover_data,
                    roads_data=roads_data
                )
                
                if facility_results:
                    # Add facility ID to all dataframes
                    for key, df in facility_results.items():
                        df['facility_id'] = facility_name
                    
                    all_X_features.append(facility_results['X_features'])
                    all_y_method1.append(facility_results['y_method1'])
                    all_y_method2.append(facility_results['y_method2'])
                    all_y_method3.append(facility_results['y_method3'])
                    
                    print(f"✓ {facility_name}: {len(facility_results['X_features'])} samples processed")
                else:
                    print(f"✗ {facility_name}: Processing failed")
                    
            except Exception as e:
                print(f"✗ {facility_name}: Error - {e}")
                continue
        
        # Combine all facilities
        if all_X_features:
            print(f"\n{'='*60}")
            print("COMBINING ALL FACILITIES")
            print(f"{'='*60}")
            
            combined_results = {
                'X_features': pd.concat(all_X_features, ignore_index=True),
                'y_method1': pd.concat(all_y_method1, ignore_index=True),
                'y_method2': pd.concat(all_y_method2, ignore_index=True),
                'y_method3': pd.concat(all_y_method3, ignore_index=True)
            }
            
            # Print summary statistics
            self.print_combined_summary(combined_results)
            
            return combined_results
        else:
            print("✗ No facilities processed successfully")
            return {}
    
    def process_single_facility(self, facility_name: str, facility_config: Dict,
                              facility_metadata: pd.DataFrame, year: int,
                              satellite_data: Dict, meteorology_data,
                              elevation_data=None, landcover_data=None, roads_data=None) -> Dict:
        """Process a single facility with its metadata"""
        
        self.facility_lat, self.facility_lon, self.facility_height = facility_config['coords']
        self.scale_factor = facility_config['scale_factor']
        self.NEI_annual_emission = facility_config['NEI_annual_emission_t']
        
        print(f"Facility coordinates: ({self.facility_lat:.5f}, {self.facility_lon:.5f}, {self.facility_height}m)")
        print(f"Scale factor: {self.scale_factor:.2e}")
        print(f"NEI annual emission: {self.NEI_annual_emission:.1f} t/year")
        
        # Load HYSPLIT data for this facility
        hysplit_data = self.load_hysplit_data(facility_name, year)
        if not hysplit_data:
            print(f"No HYSPLIT data found for {facility_name}")
            return None
        
        # Get target grid coordinates
        target_coords = self.get_target_grid_coordinates(hysplit_data)
        if len(target_coords) == 0:
            print("No valid target coordinates")
            return None
        
        # Extract features (including monthly emission rates)
        print("Processing features...")
        satellite_features = self.extract_satellite_features_grid(satellite_data, target_coords, meteorology_data)
        meteorology_features = self.extract_meteorology_features_grid(meteorology_data, target_coords)
        facility_features = self.extract_facility_features(target_coords, facility_metadata)
        
        feature_dfs = [
            ('satellite', satellite_features),
            ('meteorology', meteorology_features),
            ('facility', facility_features)
        ]
        
        # Add topographical features if available
        if elevation_data is not None:
            elevation_features = self.extract_elevation_features(elevation_data, target_coords)
            landcover_features = self.extract_landcover_features(landcover_data, target_coords)
            roads_features = self.extract_roads_features(roads_data, target_coords)
            
            feature_dfs.extend([
                ('elevation', elevation_features),
                ('landcover', landcover_features),
                ('roads', roads_features)
            ])
        
        # Combine features
        X_features = self.combine_features(feature_dfs)
        
        # Extract target values using three methods
        # Get activity scaling factors from metadata
        activity_scales = {}
        for _, row in facility_metadata.iterrows():
            month = int(row['month'])
            activity_scales[month] = row['activity']
        
        y_method1 = self.extract_target_values_method1(hysplit_data)
        y_method2 = self.extract_target_values_method2(hysplit_data, activity_scales)
        y_method3 = self.extract_target_values_method3(hysplit_data, activity_scales)
        
        # Clean data (remove NaN values)
        X_features, y_method1, y_method2, y_method3 = self.clean_data(
            X_features, y_method1, y_method2, y_method3
        )
        
        if len(X_features) == 0:
            print("No valid samples after cleaning")
            return None
        
        return {
            'X_features': X_features,
            'y_method1': y_method1,
            'y_method2': y_method2,
            'y_method3': y_method3
        }
    
    def load_hysplit_data(self, facility_name: str, year: int) -> Dict[int, nc.Dataset]:
        """Load HYSPLIT monthly files for a specific facility"""
        hysplit_data = {}
        
        facility_path = f"{DATA_PATH}/HYSPLIT_Output/{facility_name}"
        
        for month in range(1, 13):
            month_str = f"{month:02d}"
            file_path = f"{facility_path}/pm25_conc_output_{year}-{month_str}_netcdf.nc"
            
            try:
                hysplit_data[month] = nc.Dataset(file_path, 'r')
                print(f"  ✓ Loaded {facility_name} {year}-{month_str}")
            except FileNotFoundError:
                print(f"  ✗ Missing: {file_path}")
        
        print(f"Loaded {len(hysplit_data)}/12 HYSPLIT files for {facility_name}")
        return hysplit_data
    
    def extract_target_values_method1(self, hysplit_data: Dict[int, nc.Dataset]) -> pd.DataFrame:
        """Method 1: Original values × uston2miug (base scaling)"""
        return self._extract_target_values_base(hysplit_data, method_name="method1", 
                                               activity_scales=None, weekly_multiplier=1.0)
    
    def extract_target_values_method2(self, hysplit_data: Dict[int, nc.Dataset], 
                                    activity_scales: Dict[int, float]) -> pd.DataFrame:
        """Method 2: Base scaling × activity scaling"""
        return self._extract_target_values_base(hysplit_data, method_name="method2",
                                               activity_scales=activity_scales, weekly_multiplier=1.0)
    
    def extract_target_values_method3(self, hysplit_data: Dict[int, nc.Dataset],
                                    activity_scales: Dict[int, float]) -> pd.DataFrame:
        """Method 3: Method 2 × 7×24 (weekly accumulation for PM2.5 residence time)"""
        weekly_multiplier = 7 * 24  # 168 hours per week
        return self._extract_target_values_base(hysplit_data, method_name="method3",
                                               activity_scales=activity_scales, 
                                               weekly_multiplier=weekly_multiplier)
    
    def _extract_target_values_base(self, hysplit_data: Dict[int, nc.Dataset], 
                                  method_name: str, activity_scales: Optional[Dict[int, float]] = None,
                                  weekly_multiplier: float = 1.0) -> pd.DataFrame:
        """Base method for extracting target values with different scaling approaches"""
        targets_list = []
        
        # If activity_scales provided, calculate total activity for proper distribution
        if activity_scales:
            total_activity = sum(activity_scales.values())
        
        for month, hysplit_ds in hysplit_data.items():
            # HYSPLIT structure: pm25(time, levels, latitude, longitude)
            pm25_values = hysplit_ds.variables['pm25'][0, 0, :, :]  # Take first time and level
            
            lats = hysplit_ds.variables['latitude'][:]
            lons = hysplit_ds.variables['longitude'][:]
            
            # Same grid subsetting as coordinates
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
            
            # Get scaling factors
            base_scale = self.scale_factor
            
            # Calculate activity-based scaling
            if activity_scales:
                # Activity distribution: (monthly_activity / total_activity) 
                activity_fraction = activity_scales.get(month, 1.0) / total_activity
                # Scale by 12 to maintain annual total (since we have 12 months)
                activity_scale = activity_fraction * 12
            else:
                activity_scale = 1.0

            for i, lat in enumerate(lats_subset):
                for j, lon in enumerate(lons_subset):
                    pm25_val = float(pm25_subset[i, j])
                    
                    # Handle potential masked values or invalid data
                    if np.ma.is_masked(pm25_val) or not np.isfinite(pm25_val):
                        pm25_val = np.nan
                    elif not np.isnan(pm25_val):
                        # Apply all scaling factors
                        pm25_val = pm25_val * base_scale * activity_scale * weekly_multiplier
                    
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
            print(f"\n{method_name.upper()} Target values:")
            print(f"  Range: {valid_values.min():.2e} to {valid_values.max():.2e}")
            print(f"  Mean: {valid_values.mean():.2e} ± {valid_values.std():.2e}")
        
        return df
    
    def combine_features(self, feature_dfs: List[Tuple[str, pd.DataFrame]]) -> pd.DataFrame:
        """Combine feature DataFrames with debugging"""
        # Debug each feature DataFrame
        for name, df in feature_dfs:
            print(f"\n{name.upper()} features:")
            print(f"  Shape: {df.shape}")
            if len(df) > 0:
                print(f"  Sample columns: {list(df.columns)[:5]}...")
                nan_counts = df.isnull().sum()
                nan_counts = nan_counts[nan_counts > 0]
                if len(nan_counts) > 0:
                    print(f"  NaN counts: {dict(list(nan_counts.items())[:3])}...")
            else:
                print("  WARNING: Empty DataFrame!")
        
        # Start with first non-empty DataFrame
        X_features = None
        for name, df in feature_dfs:
            if len(df) > 0:
                X_features = df.copy()
                print(f"\nStarting with {name} features: {X_features.shape}")
                break
        
        if X_features is None:
            print("All feature DataFrames are empty!")
            return pd.DataFrame()
        
        # Merge other features
        for name, df in feature_dfs[1:]:
            if len(df) == 0:
                print(f"Skipping empty {name} DataFrame")
                continue
            
            before_shape = X_features.shape
            X_features = X_features.merge(df, on=['month', 'grid_i', 'grid_j'], how='left')
            after_shape = X_features.shape
            
            print(f"After merging {name}: {before_shape} -> {after_shape}")
        
        return X_features
    
    def clean_data(self, X_features: pd.DataFrame, *y_targets) -> Tuple[pd.DataFrame, ...]:
        """Remove rows with NaN values from features and all target DataFrames"""
        print(f"\nBefore cleaning: {len(X_features)} samples")
        
        # Check for NaN values in features
        feature_nan_mask = X_features.isnull().any(axis=1)
        feature_nans = feature_nan_mask.sum()
        print(f"Feature NaN rows: {feature_nans}")
        
        # Check for NaN values in any target
        target_nan_masks = []
        for i, y_target in enumerate(y_targets):
            target_nan_mask = y_target['pm25_concentration'].isnull()
            target_nan_masks.append(target_nan_mask)
            print(f"Target method {i+1} NaN values: {target_nan_mask.sum()}")
        
        # Combined mask: keep rows with no NaN in features or any target
        combined_nan_mask = feature_nan_mask
        for target_nan_mask in target_nan_masks:
            combined_nan_mask = combined_nan_mask | target_nan_mask
        
        valid_mask = ~combined_nan_mask
        valid_count = valid_mask.sum()
        
        print(f"Valid samples: {valid_count}/{len(X_features)} ({valid_count/len(X_features)*100:.1f}%)")
        
        # Apply mask to all DataFrames
        X_features_clean = X_features[valid_mask].reset_index(drop=True)
        y_targets_clean = []
        
        for y_target in y_targets:
            y_clean = y_target[valid_mask].reset_index(drop=True)
            y_targets_clean.append(y_clean)
        
        return X_features_clean, *y_targets_clean
    
    def print_combined_summary(self, combined_results: Dict[str, pd.DataFrame]):
        """Print summary statistics for combined results"""
        X_features = combined_results['X_features']
        
        print(f"Combined dataset: {len(X_features)} samples")
        print(f"Features: {len(X_features.columns)} columns")
        
        # Facility distribution
        facility_counts = X_features['facility_id'].value_counts()
        print(f"Facility distribution: {facility_counts.to_dict()}")
        
        # Monthly distribution
        month_counts = X_features['month'].value_counts().sort_index()
        print(f"Monthly distribution: {dict(list(month_counts.items())[:6])}...")
        
        # Show emission rate ranges
        if 'monthly_emission_rate_t_per_hr' in X_features.columns:
            emission_rates = X_features['monthly_emission_rate_t_per_hr']
            print(f"Emission rates: {emission_rates.min():.4f} - {emission_rates.max():.4f} t/hr")
        
        # Target value statistics
        for method in ['method1', 'method2', 'method3']:
            y_data = combined_results[f'y_{method}']
            valid_values = y_data['pm25_concentration'].dropna()
            
            print(f"\n{method.upper()} target statistics:")
            print(f"  Range: {valid_values.min():.2e} to {valid_values.max():.2e}")
            print(f"  Mean: {valid_values.mean():.2e} ± {valid_values.std():.2e}")
    
    # Placeholder methods - implement these based on your specific needs
    def extract_satellite_features_grid(self, satellite_data, target_coords, meteorology_data):
        """Extract satellite features - implement based on your requirements"""
        return pd.DataFrame()
    
    def extract_meteorology_features_grid(self, meteorology_data, target_coords):
        """Extract meteorology features - implement based on your requirements"""
        return pd.DataFrame()
    
    def extract_facility_features(self, target_coords, facility_metadata):
        """Extract facility features - implement based on your requirements"""
        return pd.DataFrame()
    
    def extract_elevation_features(self, elevation_data, target_coords):
        """Extract elevation features - implement based on your requirements"""
        return pd.DataFrame()
    
    def extract_landcover_features(self, landcover_data, target_coords):
        """Extract landcover features - implement based on your requirements"""
        return pd.DataFrame()
    
    def extract_roads_features(self, roads_data, target_coords):
        """Extract roads features - implement based on your requirements"""
        return pd.DataFrame()