import pandas as pd
import numpy as np
import xarray as xr
import netCDF4 as nc
from sklearn.metrics.pairwise import haversine_distances
import pyproj
from pathlib import Path
import math
from typing import Dict, Tuple, List, Optional
import rasterio
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import box, Point
import warnings
warnings.filterwarnings('ignore')

from utils.path_util import DATA_PATH

# Base scaling factor for all facilities
uston2miug = 907184740000
# Additional scaling factor for Suncor to correct previous error
suncor_additional_scale = 5.711149751263955

class MultiFacilityAirQualityProcessor:
    """
    Process satellite, meteorology, HYSPLIT, and topographical data for multiple facilities
    with three different target value calculation methods and monthly emission rates
    """
    
    def __init__(self, grid_size: int = 24):
        """Initialize processor with configurable grid size"""
        self.geodesic = pyproj.Geod(ellps='WGS84')
        self.grid_size = grid_size
        self.facility_metadata = None
        
        print(f"Initialized multi-facility processor with {grid_size}x{grid_size} grid")
        
    def load_facility_metadata(self, metadata_file: str) -> pd.DataFrame:
        """
        Load facility metadata CSV
        Expected columns: name, lat, lon, height, NEI_annual_emission_t, month, activity
        """
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
                scale_factor = uston2miug
                if facility.lower() == 'suncor':
                    scale_factor *= suncor_additional_scale
                
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
        """
        Add monthly emission rate column (t/hr) based on NEI annual emissions distributed by activity
        """
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
        """
        Process all facilities using metadata file
        
        Args:
            metadata_file: Path to facility metadata CSV
            year: Year to process
            include_topographical: Whether to include topographical features
            
        Returns:
            Dictionary with processed DataFrames
        """
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
                              satellite_data: Dict, meteorology_data: xr.Dataset,
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
    
    def extract_facility_features(self, target_coords: pd.DataFrame, 
                                facility_metadata: pd.DataFrame) -> pd.DataFrame:
        """
        Extract facility-related features including monthly emission rates
        """
        features_list = []
        
        # Create a lookup for monthly emission rates
        emission_rate_lookup = {}
        for _, row in facility_metadata.iterrows():
            month = int(row['month'])
            emission_rate_lookup[month] = row['monthly_emission_rate_t_per_hr']
        
        for _, row in target_coords.iterrows():
            target_lat, target_lon = row['lat'], row['lon']
            month = int(row['month'])
            
            # Calculate distance and bearing using pyproj
            fwd_azimuth, back_azimuth, distance_m = self.geodesic.inv(
                self.facility_lon, self.facility_lat, target_lon, target_lat
            )
            
            distance_km = distance_m / 1000.0  # Convert to km
            bearing = (fwd_azimuth + 360) % 360  # Ensure 0-360 range
            
            # Get monthly emission rate for this month
            monthly_emission_rate = emission_rate_lookup.get(month, 0.0)
            
            features_list.append({
                'month': month,
                'grid_i': row['grid_i'],
                'grid_j': row['grid_j'],
                'facility_lat': self.facility_lat,
                'facility_lon': self.facility_lon,
                'facility_height': self.facility_height,
                'distance_to_facility': distance_km,
                'bearing_from_facility': bearing,
                'NEI_annual_emission_t': self.NEI_annual_emission,
                'monthly_emission_rate_t_per_hr': monthly_emission_rate
            })
        
        return pd.DataFrame(features_list)
    
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
    
    def save_processed_data(self, combined_results: Dict[str, pd.DataFrame], 
                          year: int, output_base_path: str = None):
        """Save processed data in organized folder structure"""
        if output_base_path is None:
            output_base_path = f"{DATA_PATH}/processed_data"
        
        base_path = Path(output_base_path)
        
        # Create directory structure
        combined_dir = base_path / "combined"
        by_facility_dir = base_path / "by_facility"
        metadata_dir = base_path / "metadata"
        
        for dir_path in [combined_dir, by_facility_dir, metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Save combined datasets
        X_features = combined_results['X_features']
        base_filename = f"all_facilities_{year}_grid{self.grid_size}"
        
        print(f"Saving combined datasets to {combined_dir}")
        X_features.to_csv(combined_dir / f"X_features_{base_filename}.csv", index=False)
        
        for method in ['method1', 'method2', 'method3']:
            y_data = combined_results[f'y_{method}']
            y_data.to_csv(combined_dir / f"y_target_{method}_{base_filename}.csv", index=False)
        
        # Save by facility
        print(f"Saving by-facility datasets to {by_facility_dir}")
        for facility_name in X_features['facility_id'].unique():
            facility_mask = X_features['facility_id'] == facility_name
            facility_X = X_features[facility_mask].reset_index(drop=True)
            
            facility_filename = f"{facility_name}_{year}_grid{self.grid_size}"
            facility_X.to_csv(by_facility_dir / f"X_features_{facility_filename}.csv", index=False)
            
            for method in ['method1', 'method2', 'method3']:
                y_data = combined_results[f'y_{method}']
                facility_y = y_data[facility_mask].reset_index(drop=True)
                facility_y.to_csv(by_facility_dir / f"y_target_{method}_{facility_filename}.csv", index=False)
        
        # Save metadata
        print(f"Saving metadata to {metadata_dir}")
        
        # Enhanced facility info with emission data
        facility_info = []
        for name, config in self.facilities.items():
            lat, lon, height = config['coords']
            facility_info.append({
                'facility_id': name,
                'latitude': lat,
                'longitude': lon,
                'height_m_AGL': height,
                'scale_factor': config['scale_factor'],
                'NEI_annual_emission_t': config['NEI_annual_emission_t']
            })
        
        facility_df = pd.DataFrame(facility_info)
        facility_df.to_csv(metadata_dir / "facility_info_processed.csv", index=False)
        
        # Save original facility metadata if available
        if self.facility_metadata is not None:
            self.facility_metadata.to_csv(metadata_dir / "facility_metadata_with_emissions.csv", index=False)
        
        # Processing summary
        summary = {
            'processing_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'grid_size': self.grid_size,
            'year_processed': year,
            'total_samples': len(X_features),
            'total_features': len(X_features.columns) - 1,  # Exclude facility_id
            'facilities_processed': list(self.facilities.keys()),
            'target_methods': ['method1', 'method2', 'method3'],
            'includes_emission_rates': 'monthly_emission_rate_t_per_hr' in X_features.columns,
            'includes_topographical': any('elevation' in col or 'landcover' in col or 'road' in col 
                                        for col in X_features.columns)
        }
        
        summary_df = pd.DataFrame([summary])
        summary_df.to_csv(metadata_dir / f"processing_summary_{year}.csv", index=False)
        
        print(f"\n✓ All data saved successfully!")
        print(f"Combined datasets: {combined_dir}")
        print(f"By-facility datasets: {by_facility_dir}")
        print(f"Metadata: {metadata_dir}")
    # Include all the utility methods from the original class
    def load_satellite_data(self, year: int) -> Dict[int, xr.Dataset]:
        """Load satellite data for each month"""
        satellite_data = {}
        
        for month in range(1, 13):
            month_str = f"{month:02d}"
            file_path = f"{DATA_PATH}/GWRPM25_AOI/GWRPM25_AOI_{year}{month_str}.nc"
            try:
                satellite_data[month] = xr.open_dataset(file_path)
                print(f"  ✓ Satellite month {month}")
            except FileNotFoundError:
                print(f"  ✗ Missing satellite month {month}: {file_path}")
                
        print(f"Loaded {len(satellite_data)}/12 satellite files")
        return satellite_data
    
    def load_meteorology_data(self) -> xr.Dataset:
        """Load ERA5 meteorology data"""
        file_path = f"{DATA_PATH}/ERA5_monthly_raw/data_stream-moda.nc"
        try:
            met_data = xr.open_dataset(file_path)
            print(f"✓ Meteorology data: {list(met_data.data_vars.keys())}")
            return met_data
        except FileNotFoundError:
            print(f"✗ Missing meteorology file: {file_path}")
            raise
    
    def load_elevation_data(self) -> rasterio.DatasetReader:
        """Load and merge elevation HGT files"""
        print("Loading elevation data...")
        
        hgt_files = [
            f"{DATA_PATH}/elevation_raw/N39W105.hgt",
            f"{DATA_PATH}/elevation_raw/N39W106.hgt", 
            f"{DATA_PATH}/elevation_raw/N40W105.hgt",
            f"{DATA_PATH}/elevation_raw/N40W106.hgt"
        ]
        
        datasets = []
        for file_path in hgt_files:
            if Path(file_path).exists():
                datasets.append(rasterio.open(file_path))
                print(f"  ✓ {Path(file_path).name}")
            else:
                print(f"  ✗ Missing {Path(file_path).name}")
        
        if not datasets:
            print("No elevation files found, returning None")
            return None
        
        # Merge into single dataset
        merged_array, merged_transform = merge(datasets)
        
        # Create profile for merged dataset
        profile = datasets[0].profile.copy()
        profile.update({
            'height': merged_array.shape[1],
            'width': merged_array.shape[2], 
            'transform': merged_transform
        })
        
        # Close individual datasets
        for ds in datasets:
            ds.close()
            
        # Create in-memory dataset
        merged_dataset = rasterio.io.MemoryFile().open(**profile)
        merged_dataset.write(merged_array)
        
        print(f"✓ Merged elevation: {merged_array.shape[1]}x{merged_array.shape[2]} pixels")
        return merged_dataset
    
    def load_landcover_data(self) -> rasterio.DatasetReader:
        """Load and merge land cover tiles, reproject to EPSG:4326"""
        print("Loading land cover data...")
        
        landcover_files = [
            f"{DATA_PATH}/landcover_raw/13S_20230101-20240101.tif",
            f"{DATA_PATH}/landcover_raw/13T_20230101-20240101.tif"
        ]
        
        datasets = []
        for file_path in landcover_files:
            if Path(file_path).exists():
                src_ds = rasterio.open(file_path)
                
                # Reproject to EPSG:4326 if needed
                if src_ds.crs != 'EPSG:4326':
                    print(f"  Reprojecting {Path(file_path).name} to EPSG:4326")
                
                    # Calculate transform for reprojection
                    dst_crs = 'EPSG:4326'
                    transform, width, height = calculate_default_transform(
                        src_ds.crs, dst_crs, src_ds.width, src_ds.height, *src_ds.bounds,
                        resolution=0.0001)  # ~10m resolution in degrees
                    
                    # Create output array
                    dst_array = np.zeros((height, width), dtype=src_ds.dtypes[0])
                    
                    # Reproject
                    reproject(
                        source=src_ds.read(1),
                        destination=dst_array,
                        src_transform=src_ds.transform,
                        src_crs=src_ds.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.nearest)  # Use nearest for categorical data
                    
                    # Create reprojected dataset in memory
                    profile = {
                        'driver': 'GTiff',
                        'dtype': src_ds.dtypes[0],
                        'nodata': src_ds.nodata,
                        'width': width,
                        'height': height,
                        'count': 1,
                        'crs': dst_crs,
                        'transform': transform
                    }
                    
                    reprojected_ds = rasterio.io.MemoryFile().open(**profile)
                    reprojected_ds.write(dst_array, 1)
                    datasets.append(reprojected_ds)
                    src_ds.close()
                else:
                    datasets.append(src_ds)
                    print(f"  ✓ {Path(file_path).name} (already EPSG:4326)")
            else:
                print(f"  ✗ Missing {Path(file_path).name}")
        
        if not datasets:
            print("No land cover files found, returning None")
            return None
        
        # Merge tiles (now all in EPSG:4326)
        merged_array, merged_transform = merge(datasets)
        
        # Create profile for merged dataset
        profile = datasets[0].profile.copy()
        profile.update({
            'height': merged_array.shape[1],
            'width': merged_array.shape[2],
            'transform': merged_transform
        })
        
        # Close datasets
        for ds in datasets:
            ds.close()
            
        # Create in-memory dataset
        merged_dataset = rasterio.io.MemoryFile().open(**profile)
        merged_dataset.write(merged_array)
        
        print(f"✓ Merged land cover: {merged_array.shape[1]}x{merged_array.shape[2]} pixels")
        return merged_dataset
    
    def load_roads_data(self) -> gpd.GeoDataFrame:
        """Load primary roads shapefile"""
        print("Loading roads data...")
        
        roads_file = f"{DATA_PATH}/primaryroads_raw/tl_2023_us_primaryroads.shp"
        
        if not Path(roads_file).exists():
            print(f"✗ Roads file not found: {roads_file}")
            return gpd.GeoDataFrame()
        
        roads_gdf = gpd.read_file(roads_file)
        
        # Create bounding box around all facilities (larger buffer for roads)
        all_lats = [config['coords'][0] for config in self.facilities.values()]
        all_lons = [config['coords'][1] for config in self.facilities.values()]
        
        min_lat, max_lat = min(all_lats) - 0.5, max(all_lats) + 0.5
        min_lon, max_lon = min(all_lons) - 0.5, max(all_lons) + 0.5
        
        bbox = box(min_lon, min_lat, max_lon, max_lat)
        
        # Clip roads to study area
        roads_clipped = roads_gdf[roads_gdf.geometry.intersects(bbox)].copy()
        
        print(f"✓ Clipped roads: {len(roads_clipped)} features")
        if 'RTTYP' in roads_clipped.columns:
            road_types = roads_clipped['RTTYP'].value_counts().to_dict()
            print(f"  Road types: {road_types}")
        
        return roads_clipped
    
    def get_target_grid_coordinates(self, hysplit_data: Dict[int, nc.Dataset]) -> pd.DataFrame:
        """Extract coordinate grid from HYSPLIT data with configurable grid size"""
        coords_list = []
        
        # Use first available month to get spatial grid
        first_month_data = list(hysplit_data.values())[0]
        lats = first_month_data.variables['latitude'][:]
        lons = first_month_data.variables['longitude'][:]
        
        # Take center portion of grid based on self.grid_size
        center_lat_idx = len(lats) // 2
        center_lon_idx = len(lons) // 2
        half_size = self.grid_size // 2
        
        lat_start = center_lat_idx - half_size
        lat_end = center_lat_idx + half_size
        lon_start = center_lon_idx - half_size
        lon_end = center_lon_idx + half_size
        
        lats_subset = lats[lat_start:lat_end]
        lons_subset = lons[lon_start:lon_end]
        
        print(f"HYSPLIT grid: {len(lats)} x {len(lons)} -> {len(lats_subset)} x {len(lons_subset)}")
        print(f"Lat range: {lats_subset.min():.3f} to {lats_subset.max():.3f}")
        print(f"Lon range: {lons_subset.min():.3f} to {lons_subset.max():.3f}")
        
        for month in hysplit_data.keys():
            for i, lat in enumerate(lats_subset):
                for j, lon in enumerate(lons_subset):
                    coords_list.append({
                        'month': month,
                        'grid_i': i,
                        'grid_j': j, 
                        'lat': float(lat),
                        'lon': float(lon)
                    })
        
        return pd.DataFrame(coords_list)
    
    def extract_elevation_features(self, elevation_data, target_coords: pd.DataFrame) -> pd.DataFrame:
        """Extract elevation-based features at 3km grid resolution"""
        features_list = []
        
        if elevation_data is None:
            print("No elevation data available, creating dummy features")
            for _, row in target_coords.iterrows():
                features_list.append({
                    'month': row['month'],
                    'grid_i': row['grid_i'],
                    'grid_j': row['grid_j'],
                    'elevation': 1500,  # Reasonable default for Colorado
                    'elevation_diff_from_facility': 0,
                    'terrain_slope': 0,
                    'terrain_roughness': 0
                })
            return pd.DataFrame(features_list)
        
        # Calculate facility elevation once
        facility_elevation = self.sample_raster_at_point(elevation_data, self.facility_lat, self.facility_lon)
        
        print("Extracting elevation features at 3km grid resolution...")
        
        for _, row in target_coords.iterrows():
            # Sample elevation at point (center of 3km grid cell)
            elevation = self.sample_raster_at_point(elevation_data, row['lat'], row['lon'])
            
            # Calculate elevation difference from facility
            elevation_diff = elevation - facility_elevation if (elevation is not None and facility_elevation is not None) else None
            
            # Calculate slope and terrain roughness using 3km window
            slope, roughness = self.calculate_terrain_metrics(elevation_data, row['lat'], row['lon'], window_size=9)
            
            features_list.append({
                'month': row['month'],
                'grid_i': row['grid_i'],
                'grid_j': row['grid_j'],
                'elevation': elevation if elevation is not None else 1500,
                'elevation_diff_from_facility': elevation_diff if elevation_diff is not None else 0,
                'terrain_slope': slope if slope is not None else 0,
                'terrain_roughness': roughness if roughness is not None else 0
            })
        
        return pd.DataFrame(features_list)
    
    def extract_landcover_features(self, landcover_data, target_coords: pd.DataFrame) -> pd.DataFrame:
        """Extract land cover features aggregated from 10m to 3km resolution"""
        features_list = []
        
        if landcover_data is None:
            print("No land cover data available, creating dummy features")
            for _, row in target_coords.iterrows():
                features_list.append({
                    'month': row['month'],
                    'grid_i': row['grid_i'],
                    'grid_j': row['grid_j'],
                    'landcover_dominant_class': 0,
                    'landcover_diversity': 0,
                    'landcover_urban_percent': 0,
                    'landcover_forest_percent': 0,
                    'landcover_agriculture_percent': 0
                })
            return pd.DataFrame(features_list)
        
        print("Extracting land cover features (10m -> 3km aggregation)...")
        success_count = 0
        
        # Calculate proper cell size for 3km HYSPLIT grid
        cell_size_deg = 0.03  # Approximately 3km at this latitude
        
        for _, row in target_coords.iterrows():
            # Calculate 3km grid cell bounds
            bounds = (row['lon'] - cell_size_deg/2, row['lat'] - cell_size_deg/2,
                     row['lon'] + cell_size_deg/2, row['lat'] + cell_size_deg/2)
            
            # Extract and aggregate land cover classes within 3km cell
            landcover_stats = self.aggregate_landcover_to_3km(landcover_data, bounds)
            
            if landcover_stats:
                dominant_class = landcover_stats['dominant_class']
                diversity = landcover_stats['diversity']
                class_percentages = landcover_stats['class_percentages']
                success_count += 1
            else:
                dominant_class = 0
                diversity = 0
                class_percentages = {}
            
            # Common land cover classes for Sentinel-2 ESA WorldCover:
            urban_percent = class_percentages.get(50, 0)  # Built-up
            forest_percent = class_percentages.get(10, 0)  # Tree cover
            agriculture_percent = class_percentages.get(40, 0)  # Cropland
            
            features_list.append({
                'month': row['month'],
                'grid_i': row['grid_i'],
                'grid_j': row['grid_j'],
                'landcover_dominant_class': dominant_class,
                'landcover_diversity': diversity,
                'landcover_urban_percent': urban_percent,
                'landcover_forest_percent': forest_percent,
                'landcover_agriculture_percent': agriculture_percent
            })
        
        print(f"Land cover aggregation successful for {success_count}/{len(target_coords)} grid cells")
        return pd.DataFrame(features_list)
    
    def aggregate_landcover_to_3km(self, landcover_data, bounds: Tuple[float, float, float, float]) -> Dict:
        """Aggregate 10m land cover data to 3km grid cell"""
        try:
            # Create geometry for masking
            bbox_geom = [box(*bounds)]
            
            # Mask raster to bounds
            masked_array, masked_transform = mask(landcover_data, bbox_geom, crop=True, filled=False)
            
            if masked_array.size == 0:
                return None
                
            masked_array = masked_array[0]  # Take first band
            
            # Handle masked arrays properly
            if hasattr(masked_array, 'mask'):
                valid_data = masked_array.compressed()
            else:
                if hasattr(landcover_data, 'nodata') and landcover_data.nodata is not None:
                    valid_data = masked_array[masked_array != landcover_data.nodata]
                else:
                    valid_data = masked_array[~np.isnan(masked_array)]
            
            if len(valid_data) == 0:
                return None
            
            # Count pixels for each land cover class
            from collections import Counter
            class_counts = Counter(valid_data.flatten())
            
            # Remove invalid classes
            valid_counts = {}
            total_pixels = 0
            for cls, count in class_counts.items():
                if not np.isnan(cls) and np.isfinite(cls) and cls > 0:
                    cls_int = int(cls)
                    valid_counts[cls_int] = count
                    total_pixels += count
            
            if total_pixels == 0:
                return None
            
            # Calculate percentages
            class_percentages = {}
            for cls, count in valid_counts.items():
                class_percentages[cls] = (count / total_pixels) * 100
            
            # Get dominant class
            dominant_class = max(valid_counts.items(), key=lambda x: x[1])[0]
            
            # Calculate diversity
            diversity = sum(1 for pct in class_percentages.values() if pct > 1.0)
            
            return {
                'dominant_class': dominant_class,
                'diversity': diversity,
                'class_percentages': class_percentages,
                'total_pixels': total_pixels
            }
            
        except Exception as e:
            if np.random.random() < 0.01:  # Print 1% of errors
                print(f"Land cover aggregation error: {e}")
            return None
    
    def extract_roads_features(self, roads_data: gpd.GeoDataFrame, target_coords: pd.DataFrame) -> pd.DataFrame:
        """Extract road-based features aggregated to 3km grid cells"""
        features_list = []
        
        if len(roads_data) == 0:
            print("No roads data available, creating dummy features")
            for _, row in target_coords.iterrows():
                features_list.append({
                    'month': row['month'],
                    'grid_i': row['grid_i'],
                    'grid_j': row['grid_j'],
                    'road_density_total': 0,
                    'road_density_interstate': 0,
                    'road_density_us_highway': 0,
                    'distance_to_nearest_road': 999.0
                })
            return pd.DataFrame(features_list)
        
        print("Extracting road features (aggregated to 3km grid)...")
        
        # Calculate proper cell size for 3km HYSPLIT grid
        cell_size_deg = 0.03  # Approximately 3km at this latitude
        
        # Group by grid cell (same for all months)
        unique_coords = target_coords[['grid_i', 'grid_j', 'lat', 'lon']].drop_duplicates()
        
        for _, row in unique_coords.iterrows():
            # Create 3km grid cell polygon
            cell_polygon = box(row['lon'] - cell_size_deg/2, 
                             row['lat'] - cell_size_deg/2,
                             row['lon'] + cell_size_deg/2, 
                             row['lat'] + cell_size_deg/2)
            
            # Find roads intersecting this 3km cell
            cell_roads = roads_data[roads_data.geometry.intersects(cell_polygon)]
            
            # Calculate road densities by type
            total_road_length = 0
            interstate_length = 0
            us_highway_length = 0
            
            if len(cell_roads) > 0:
                for _, road in cell_roads.iterrows():
                    intersected = road.geometry.intersection(cell_polygon)
                    if intersected.is_empty:
                        continue
                    
                    # Calculate length in degrees (approximate)
                    if hasattr(intersected, 'length'):
                        road_length_deg = intersected.length
                        road_length_km = road_length_deg * 85  # ~85 km per degree longitude
                        
                        total_road_length += road_length_km
                        
                        # Categorize by road type (RTTYP field)
                        road_type = road.get('RTTYP', '')
                        if road_type == 'I':  # Interstate
                            interstate_length += road_length_km
                        elif road_type in ['U', 'S']:  # US Highway, State Highway
                            us_highway_length += road_length_km
            
            # Calculate distance to nearest major road
            point = Point(row['lon'], row['lat'])
            if len(roads_data) > 0:
                major_roads = roads_data[roads_data['RTTYP'].isin(['I', 'U', 'S'])]
                if len(major_roads) > 0:
                    distances = major_roads.geometry.distance(point)
                    min_distance = distances.min() * 111.32  # Convert degrees to km
                else:
                    min_distance = 999.0
            else:
                min_distance = 999.0
            
            features_list.append({
                'grid_i': row['grid_i'],
                'grid_j': row['grid_j'],
                'road_density_total': total_road_length,
                'road_density_interstate': interstate_length,
                'road_density_us_highway': us_highway_length,
                'distance_to_nearest_road': min_distance
            })
        
        # Convert to DataFrame and expand to all months
        roads_df = pd.DataFrame(features_list)
        
        result_list = []
        for month in range(1, 13):
            month_df = roads_df.copy()
            month_df['month'] = month
            result_list.append(month_df)
        
        print(f"Road features calculated for {len(unique_coords)} grid cells")
        return pd.concat(result_list, ignore_index=True)
    
    def extract_satellite_features_grid(self, satellite_data: Dict[int, xr.Dataset], 
                                       target_coords: pd.DataFrame,
                                       meteorology_data: xr.Dataset) -> pd.DataFrame:
        """Extract satellite PM2.5 features optimized for smaller grid"""
        features_list = []
        
        print("Processing satellite features...")
        
        for _, row in target_coords.iterrows():
            month = row['month']
            target_lat, target_lon = row['lat'], row['lon']
            
            if month not in satellite_data:
                print(f"Warning: No satellite data for month {month}")
                continue
                
            # Get satellite data for this month
            sat_ds = satellite_data[month]
            
            # Try to identify the PM2.5 variable name
            pm25_var = None
            for var_name in sat_ds.data_vars.keys():
                if 'pm' in var_name.lower() or 'PM' in var_name:
                    pm25_var = var_name
                    break
            
            if pm25_var is None:
                print(f"Warning: Could not find PM2.5 variable in satellite data")
                continue
                
            # Get satellite grid coordinates
            sat_lats = sat_ds.coords['latitude'].values if 'latitude' in sat_ds.coords else sat_ds.coords['lat'].values
            sat_lons = sat_ds.coords['longitude'].values if 'longitude' in sat_ds.coords else sat_ds.coords['lon'].values
            sat_pm25 = sat_ds[pm25_var].values
            
            # Handle 3D data (squeeze out time dimension if present)
            if sat_pm25.ndim == 3:
                sat_pm25 = sat_pm25[0]  # Take first time step
            
            # Create coordinate grids
            sat_lon_grid, sat_lat_grid = np.meshgrid(sat_lons, sat_lats)
            
            # Get wind direction for upwind/downwind features
            wind_direction = self.get_monthly_wind_direction_grid(meteorology_data, month, target_lat, target_lon)
            # Extract features
            sat_at_target = self.get_nearest_grid_value(sat_lat_grid, sat_lon_grid, sat_pm25, target_lat, target_lon)

            sat_decay_1_3km = self.calc_distance_decay(sat_lat_grid, sat_lon_grid, sat_pm25, 
                                                      self.facility_lat, self.facility_lon, 
                                                      target_lat, target_lon)

            sat_directional_asymmetry = self.calc_directional_asymmetry(sat_lat_grid, sat_lon_grid, sat_pm25,
                                                                      self.facility_lat, self.facility_lon, 
                                                                      wind_direction)

      
            
            features_list.append({
                'month': month,
                'grid_i': row['grid_i'],
                'grid_j': row['grid_j'],
                'sat_at_target': sat_at_target,
                'sat_decay_1_3km': sat_decay_1_3km,
                'sat_directional_asymmetry': sat_directional_asymmetry
            })
                    
        return pd.DataFrame(features_list)
    def calc_distance_decay(self, lat_grid, lon_grid, values, facility_lat, facility_lon, target_lat, target_lon):
        """Calculate distance-weighted decay from facility to target"""
        try:
            # Get distance from facility to target
            _, _, distance_m = self.geodesic.inv(facility_lon, facility_lat, target_lon, target_lat)
            distance_km = distance_m / 1000.0
            
            # Get PM2.5 at facility and target
            pm25_facility = self.get_nearest_grid_value(lat_grid, lon_grid, values, facility_lat, facility_lon)
            pm25_target = self.get_nearest_grid_value(lat_grid, lon_grid, values, target_lat, target_lon)
            
            if pm25_facility is None or pm25_target is None or distance_km == 0:
                return None
            
            # Calculate decay rate: (facility - target) / distance
            decay_rate = (pm25_facility - pm25_target) / distance_km
            return float(decay_rate)
            
        except Exception:
            return None
    def calc_directional_asymmetry(self, lat_grid, lon_grid, values, facility_lat, facility_lon, wind_direction):
        """Calculate directional asymmetry in PM2.5 pattern around facility"""
        try:
            if wind_direction is None:
                return None

            # 8 directional sectors around facility
            sectors = []
            radius_km = 8
            sector_width = 45

            for direction in range(0, 360, sector_width):
                sector_mean = self.get_directional_mean(
                    lat_grid, lon_grid, values, facility_lat, facility_lon,
                    direction, radius_km, sector_width
                )
                sectors.append(sector_mean if sector_mean is not None else 0)

            sectors = np.array(sectors)
            if len(sectors) > 0 and not np.all(sectors == 0):
                # Coefficient of variation as asymmetry measure
                mean_conc = np.mean(sectors)
                std_conc = np.std(sectors)
                asymmetry = std_conc / (mean_conc + 1e-6)
                return float(asymmetry)
            else:
                return None

        except Exception:
            return None
    def get_directional_mean(self, lat_grid, lon_grid, values, center_lat, center_lon, 
                        direction_deg, radius_km, sector_width):
        """Get mean value in directional sector (alias for existing method)"""
        return self.extract_directional_mean_grid(lat_grid, lon_grid, values, center_lat, center_lon, 
                                                direction_deg, radius_km, sector_width)
    
    def extract_meteorology_features_grid(self, met_data: xr.Dataset, 
                                        target_coords: pd.DataFrame) -> pd.DataFrame:
        """Extract meteorology features using grid-based approach"""
        features_list = []
        
        # Get available meteorology variables
        met_vars = {
            'd2m': 'dewpoint_temp_2m',
            't2m': 'temp_2m', 
            'u10': 'u_wind_10m',
            'v10': 'v_wind_10m',
            'sp': 'surface_pressure',
            'tp': 'total_precipitation'
        }
        
        # Get meteorology grid coordinates
        met_lats = met_data.coords['latitude'].values
        met_lons = met_data.coords['longitude'].values
        met_lon_grid, met_lat_grid = np.meshgrid(met_lons, met_lats)
        
        for _, row in target_coords.iterrows():
            month = row['month']
            target_lat, target_lon = row['lat'], row['lon']
            
            # Extract meteorology at target location
            met_features = {
                'month': month,
                'grid_i': row['grid_i'], 
                'grid_j': row['grid_j']
            }
            
            # Time index for this month (month-1 because 0-indexed)
            time_idx = int(month) - 1
            
            if time_idx >= len(met_data.coords['valid_time']):
                print(f"Warning: Month {month} not available in meteorology data")
                continue
            
            u_wind_val = None
            v_wind_val = None
            
            for var_name, feature_name in met_vars.items():
                if var_name in met_data.data_vars:
                    var_data = met_data[var_name].isel(valid_time=time_idx).values
                    
                    # Get nearest grid cell value
                    grid_val = self.get_nearest_grid_value(met_lat_grid, met_lon_grid, var_data, target_lat, target_lon)
                    met_features[feature_name] = grid_val
                    
                    # Store wind components for derived features
                    if var_name == 'u10':
                        u_wind_val = grid_val
                    elif var_name == 'v10':
                        v_wind_val = grid_val
            
            # Calculate derived features
            if u_wind_val is not None and v_wind_val is not None:
                wind_speed = np.sqrt(u_wind_val**2 + v_wind_val**2)
                wind_direction = np.arctan2(v_wind_val, u_wind_val) * 180 / np.pi
                wind_direction = (wind_direction + 360) % 360  # Convert to 0-360°
                
                met_features.update({
                    'wind_speed': wind_speed,
                    'wind_direction': wind_direction
                })
            
            features_list.append(met_features)
        
        return pd.DataFrame(features_list)
    
    # Utility functions for grid-based processing
    def sample_raster_at_point(self, raster_data, lat: float, lon: float) -> Optional[float]:
        """Sample raster value at specific point"""
        try:
            row, col = raster_data.index(lon, lat)
            
            if 0 <= row < raster_data.height and 0 <= col < raster_data.width:
                value = raster_data.read(1)[row, col]
                return float(value) if not np.isnan(value) and value != raster_data.nodata else None
            else:
                return None
        except Exception:
            return None
    
    def calculate_terrain_metrics(self, elevation_data, lat: float, lon: float, 
                                window_size: int = 3) -> Tuple[Optional[float], Optional[float]]:
        """Calculate slope and terrain roughness around point"""
        try:
            row, col = elevation_data.index(lon, lat)
            
            half_window = window_size // 2
            row_start = max(0, row - half_window)
            row_end = min(elevation_data.height, row + half_window + 1)
            col_start = max(0, col - half_window)
            col_end = min(elevation_data.width, col + half_window + 1)
            
            elevation_window = elevation_data.read(1)[row_start:row_end, col_start:col_end]
            valid_elevations = elevation_window[elevation_window != elevation_data.nodata]
            
            if len(valid_elevations) > 1:
                slope = np.std(np.gradient(elevation_window.astype(float)))
                roughness = np.std(valid_elevations.astype(float))
                return float(slope), float(roughness)
            else:
                return None, None
                
        except Exception:
            return None, None
    
    def get_nearest_grid_value(self, lat_grid: np.ndarray, lon_grid: np.ndarray, 
                              values: np.ndarray, target_lat: float, target_lon: float) -> Optional[float]:
        """Find nearest grid cell value using haversine distance"""
        try:
            lat_flat = lat_grid.flatten()
            lon_flat = lon_grid.flatten()
            values_flat = values.flatten()
            
            valid_mask = ~np.isnan(values_flat)
            if not np.any(valid_mask):
                return None
                
            lat_valid = lat_flat[valid_mask]
            lon_valid = lon_flat[valid_mask]
            values_valid = values_flat[valid_mask]
            
            target_coords = np.radians([[target_lat, target_lon]])
            grid_coords = np.radians(np.column_stack([lat_valid, lon_valid]))
            
            distances = haversine_distances(target_coords, grid_coords)[0] * 6371000
            
            nearest_idx = np.argmin(distances)
            return float(values_valid[nearest_idx])
            
        except Exception as e:
            print(f"Nearest grid value extraction failed: {e}")
            return None
    
    def extract_regional_mean_grid(self, lat_grid: np.ndarray, lon_grid: np.ndarray, values: np.ndarray,
                                 center_lat: float, center_lon: float, radius_km: float) -> Optional[float]:
        """Extract mean value within radius using grid cells"""
        try:
            lat_flat = lat_grid.flatten()
            lon_flat = lon_grid.flatten()
            values_flat = values.flatten()
            
            valid_mask = ~np.isnan(values_flat)
            if not np.any(valid_mask):
                return None
                
            lat_valid = lat_flat[valid_mask]
            lon_valid = lon_flat[valid_mask]
            values_valid = values_flat[valid_mask]
            
            center_coords = np.radians([[center_lat, center_lon]])
            grid_coords = np.radians(np.column_stack([lat_valid, lon_valid]))
            
            distances_km = haversine_distances(center_coords, grid_coords)[0] * 6371
            
            mask = distances_km <= radius_km
            
            if np.any(mask):
                return float(np.mean(values_valid[mask]))
            else:
                return None
        except Exception as e:
            print(f"Regional mean extraction failed: {e}")
            return None
    
    def extract_directional_mean_grid(self, lat_grid: np.ndarray, lon_grid: np.ndarray, values: np.ndarray,
                                    center_lat: float, center_lon: float, 
                                    direction_deg: float, radius_km: float, sector_width: float = 60) -> Optional[float]:
        """Extract mean value in directional sector using grid cells"""
        try:
            lat_flat = lat_grid.flatten()
            lon_flat = lon_grid.flatten()
            values_flat = values.flatten()
            
            valid_mask = ~np.isnan(values_flat)
            if not np.any(valid_mask):
                return None
                
            lat_valid = lat_flat[valid_mask]
            lon_valid = lon_flat[valid_mask]
            values_valid = values_flat[valid_mask]
            
            center_coords = np.radians([[center_lat, center_lon]])
            grid_coords = np.radians(np.column_stack([lat_valid, lon_valid]))
            
            distances_km = haversine_distances(center_coords, grid_coords)[0] * 6371
            
            # Calculate bearings using pyproj
            bearings = []
            for lat_pt, lon_pt in zip(lat_valid, lon_valid):
                fwd_az, _, _ = self.geodesic.inv(center_lon, center_lat, lon_pt, lat_pt)
                bearings.append((fwd_az + 360) % 360)
            
            bearings = np.array(bearings)
            
            # Create sector mask
            angle_diff = np.abs(bearings - direction_deg)
            angle_diff = np.minimum(angle_diff, 360 - angle_diff)  # Handle wrap-around
            
            mask = (distances_km <= radius_km) & (angle_diff <= sector_width/2)
            
            if np.any(mask):
                return float(np.mean(values_valid[mask]))
            else:
                return None
        except Exception as e:
            print(f"Directional mean extraction failed: {e}")
            return None
    
    def get_monthly_wind_direction_grid(self, met_data: xr.Dataset, month: int, 
                                      target_lat: float, target_lon: float) -> Optional[float]:
        """Get wind direction using nearest grid cell"""
        try:
            time_idx = int(month) - 1
            
            if time_idx >= len(met_data.coords['valid_time']):
                return None
            
            if 'u10' not in met_data.data_vars or 'v10' not in met_data.data_vars:
                return None
                
            u_wind = met_data['u10'].isel(valid_time=time_idx).values
            v_wind = met_data['v10'].isel(valid_time=time_idx).values
            met_lats = met_data.coords['latitude'].values
            met_lons = met_data.coords['longitude'].values
            met_lon_grid, met_lat_grid = np.meshgrid(met_lons, met_lats)
            
            u_interp = self.get_nearest_grid_value(met_lat_grid, met_lon_grid, u_wind, target_lat, target_lon)
            v_interp = self.get_nearest_grid_value(met_lat_grid, met_lon_grid, v_wind, target_lat, target_lon)
            
            if u_interp is None or v_interp is None:
                return None
            
            wind_direction = np.arctan2(v_interp, u_interp) * 180 / np.pi
            return float((wind_direction + 360) % 360)
            
        except Exception as e:
            print(f"Wind direction calculation failed: {e}")
            return None


# Usage example and main execution
if __name__ == "__main__":
    # Initialize multi-facility processor
    processor = MultiFacilityAirQualityProcessor(grid_size=24)
    
    print("="*70)
    print("MULTI-FACILITY AIR QUALITY DATA PROCESSING WITH EMISSION RATES")
    print("="*70)
    
    # Read existing facility metadata
    metadata_file = f"{DATA_PATH}/processed_data/metadata/facility_info.csv"
    
    if not Path(metadata_file).exists():
        print(f"✗ Metadata file not found: {metadata_file}")
        print("Expected columns: name, lat, lon, height, NEI_annual_emission_t, month, activity")
        exit(1)
    
    print(f"✓ Reading facility metadata: {metadata_file}")
    
    # Process all facilities
    try:
        print("\n=== PROCESSING WITH ALL FEATURES ===")
        combined_results = processor.process_all_facilities(
            metadata_file=metadata_file,
            year=2023,
            include_topographical=True
        )
        
        if not combined_results:
            print("Failed with topographical features, trying without...")
            combined_results = processor.process_all_facilities(
                metadata_file=metadata_file,
                year=2023,
                include_topographical=False
            )
        
        if combined_results:
            print("\n=== SAVING PROCESSED DATA ===")
            processor.save_processed_data(combined_results, year=2023)
            
            print("\n=== PROCESSING COMPLETE ===")
            X_features = combined_results['X_features']
            print(f"✓ Total samples: {len(X_features):,}")
            print(f"✓ Total features: {len(X_features.columns)-1}")
            print(f"✓ Facilities: {list(X_features['facility_id'].unique())}")
            print("✓ Three target value methods generated")
            print("✓ Monthly emission rates included as features")
            
            # Show emission rate statistics
            if 'monthly_emission_rate_t_per_hr' in X_features.columns:
                emission_stats = X_features.groupby('facility_id')['monthly_emission_rate_t_per_hr'].agg(['min', 'max', 'mean'])
                print(f"\n=== MONTHLY EMISSION RATES BY FACILITY ===")
                print(emission_stats.round(4))
            
            # Show target value ranges
            print("\n=== TARGET VALUE RANGES BY METHOD ===")
            for method in ['method1', 'method2', 'method3']:
                y_data = combined_results[f'y_{method}']['pm25_concentration']
                valid_data = y_data.dropna()
                if len(valid_data) > 0:
                    print(f"{method}: {valid_data.min():.2e} to {valid_data.max():.2e}")
        
        else:
            print("\n✗ PROCESSING FAILED")
            
    except Exception as e:
        print(f"\n✗ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        
    print("\n" + "="*70)
    print("END OF MULTI-FACILITY PROCESSING")
    print("="*70)