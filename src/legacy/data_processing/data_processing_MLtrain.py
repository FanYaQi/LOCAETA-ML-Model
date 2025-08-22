import numpy as np
import pandas as pd
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
import sys
## add sys path for local functions
path_to_add = '/Users/yaqifan/Documents/Github/LOCAETA-ML/src'
if path_to_add not in sys.path:
    sys.path.insert(0, path_to_add)
from utils.path_util import DATA_PATH

uston2miug = 907184740000
RMBC_scale_factor = uston2miug
suncor_scale_factor = 5.711149751263955 * uston2miug

class AirQualityDataProcessor:
    """
    Process satellite, meteorology, HYSPLIT, and topographical data into training features
    using grid-based approach with configurable grid size
    """
    
    def __init__(self, grid_size: int = 24):
        """Initialize processor with configurable grid size"""
        self.geodesic = pyproj.Geod(ellps='WGS84')
        self.grid_size = grid_size
        print(f"Initialized with {grid_size}x{grid_size} grid")
        
    def process_training_data(self, facility_coords: Tuple[float, float, float], 
                            year: int = 2023, include_topographical: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Main pipeline to process all data into feature matrix X and target y
        
        Args:
            facility_coords: (lat, lon, height_m_AGL)
            year: Year to process
            include_topographical: Whether to include elevation/landcover/roads features
            
        Returns:
            X_features: DataFrame with features for each grid cell and month
            y_target: DataFrame with PM2.5 concentrations
        """
        self.facility_lat, self.facility_lon, self.facility_height = facility_coords
        
        print("Loading atmospheric data...")
        hysplit_data = self.load_hysplit_data(year)
        satellite_data = self.load_satellite_data(year)
        meteorology_data = self.load_meteorology_data()
        
        if include_topographical:
            print("Loading topographical data...")
            elevation_data = self.load_elevation_data()
            landcover_data = self.load_landcover_data()
            roads_data = self.load_roads_data()
        else:
            print("Skipping topographical data...")
            elevation_data = None
            landcover_data = None
            roads_data = None
        
        print("Processing features...")
        # Get target grid from HYSPLIT
        target_coords = self.get_target_grid_coordinates(hysplit_data)
        
        # Process each data source using grid-based approach
        satellite_features = self.extract_satellite_features_grid(satellite_data, target_coords, meteorology_data)
        meteorology_features = self.extract_meteorology_features_grid(meteorology_data, target_coords)
        facility_features = self.extract_facility_features(target_coords)
        
        feature_dfs = [
            ('satellite', satellite_features), 
            ('meteorology', meteorology_features), 
            ('facility', facility_features)
        ]
        
        # Add topographical features if requested
        if include_topographical:
            elevation_features = self.extract_elevation_features(elevation_data, target_coords)
            landcover_features = self.extract_landcover_features(landcover_data, target_coords)
            roads_features = self.extract_roads_features(roads_data, target_coords)
            
            feature_dfs.extend([
                ('elevation', elevation_features), 
                ('landcover', landcover_features), 
                ('roads', roads_features)
            ])
        
        # Combine all features with debugging
        # Debug each feature DataFrame
        for name, df in feature_dfs:
            print(f"\n{name.upper()} features:")
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {df.columns.tolist()}")
            if len(df) > 0:
                nan_counts = df.isnull().sum()
                print(f"  NaN counts: {nan_counts[nan_counts > 0].to_dict()}")
                print(f"  Sample values:\n{df.head(2)}")
            else:
                print("  WARNING: Empty DataFrame!")
        
        # Start with satellite features
        X_features = satellite_features.copy()
        print(f"\nStarting with satellite features: {X_features.shape}")
        
        # Merge other features one by one with debugging
        for name, df in feature_dfs[1:]:
            if len(df) == 0:
                print(f"Skipping empty {name} DataFrame")
                continue
                
            before_shape = X_features.shape
            X_features = X_features.merge(df, on=['month', 'grid_i', 'grid_j'], how='left')
            after_shape = X_features.shape
            
            print(f"After merging {name}: {before_shape} -> {after_shape}")
            
            # Check for new NaN values
            nan_counts = X_features.isnull().sum()
            new_nans = nan_counts[nan_counts > 0]
            if len(new_nans) > 0:
                print(f"  New NaN columns: {new_nans.to_dict()}")
        
        # Extract target values (NO log transformation, WITH scaling)
        y_target = self.extract_target_values(hysplit_data, log_transform=False)
        print(f"\nTarget values shape: {y_target.shape}")
        
        # Debug NaN values before removal
        print(f"\nBefore NaN removal: {len(X_features)} samples")
        
        # Check which columns have NaN values
        nan_counts = X_features.isnull().sum()
        cols_with_nans = nan_counts[nan_counts > 0]
        if len(cols_with_nans) > 0:
            print(f"Columns with NaN values:")
            for col, count in cols_with_nans.items():
                print(f"  {col}: {count} NaNs ({count/len(X_features)*100:.1f}%)")
        
        target_nans = y_target['pm25_concentration'].isnull().sum()
        print(f"Target NaN values: {target_nans} ({target_nans/len(y_target)*100:.1f}%)")
        
        # Remove rows with NaN values
        valid_indices = ~(X_features.isnull().any(axis=1) | y_target['pm25_concentration'].isnull())
        print(f"Valid indices: {valid_indices.sum()}/{len(valid_indices)}")
        
        X_features = X_features[valid_indices].reset_index(drop=True)
        y_target = y_target[valid_indices].reset_index(drop=True)
        
        print(f"After NaN removal: {len(X_features)} samples with {len(X_features.columns)} features")
        return X_features, y_target
    
    def load_hysplit_data(self, year: int) -> Dict[int, nc.Dataset]:
        """Load all HYSPLIT monthly files"""
        hysplit_data = {}
        months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 
                 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        
        for i, month in enumerate(months, 1):
            file_path = f"{DATA_PATH}/HYSPLIT_output/pm25_conc_output_{month}{year}_hrrr3km_netcdf.nc"
            try:
                hysplit_data[i] = nc.Dataset(file_path, 'r')
                print(f"Loaded HYSPLIT data for {month} {year}")
            except FileNotFoundError:
                print(f"Warning: Could not find HYSPLIT file for {month} {year}")
                
        return hysplit_data
    
    def load_satellite_data(self, year: int) -> Dict[int, xr.Dataset]:
        """Load satellite data for each month"""
        satellite_data = {}
        
        for month in range(1, 13):
            month_str = f"{month:02d}"
            file_path = f"{DATA_PATH}/GWRPM25_AOI/GWRPM25_AOI_{year}{month_str}.nc"
            try:
                satellite_data[month] = xr.open_dataset(file_path)
                print(f"Loaded satellite data for month {month}")
            except FileNotFoundError:
                print(f"Warning: Could not find satellite file for month {month}")
                
        return satellite_data
    
    def load_meteorology_data(self) -> xr.Dataset:
        """Load ERA5 meteorology data"""
        file_path = f"{DATA_PATH}/ERA5_monthly_raw/data_stream-moda.nc"
        met_data = xr.open_dataset(file_path)
        print(f"Loaded meteorology data with variables: {list(met_data.data_vars.keys())}")
        return met_data
    
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
                print(f"Loaded: {file_path}")
            else:
                print(f"Warning: Missing {file_path}")
        
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
        
        print(f"Merged elevation: {merged_array.shape[1]}x{merged_array.shape[2]} pixels")
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
                    print(f"Reprojecting {file_path} from {src_ds.crs} to EPSG:4326")
                
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
                    print(f"Loaded: {file_path} (already EPSG:4326)")
            else:
                print(f"Warning: Missing {file_path}")
        
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
        
        print(f"Merged land cover: {merged_array.shape[1]}x{merged_array.shape[2]} pixels")
        return merged_dataset
    
    def load_roads_data(self) -> gpd.GeoDataFrame:
        """Load primary roads shapefile"""
        print("Loading roads data...")
        
        roads_file = f"{DATA_PATH}/primaryroads_raw/tl_2023_us_primaryroads.shp"
        
        if not Path(roads_file).exists():
            print(f"Warning: Roads file not found: {roads_file}")
            return gpd.GeoDataFrame()
        
        roads_gdf = gpd.read_file(roads_file)
        
        # Create bounding box around facility area (larger buffer for roads)
        buffer_deg = 0.5  # Approximate degrees for road analysis
        bbox = box(self.facility_lon - buffer_deg, self.facility_lat - buffer_deg,
                  self.facility_lon + buffer_deg, self.facility_lat + buffer_deg)
        
        # Clip roads to study area
        roads_clipped = roads_gdf[roads_gdf.geometry.intersects(bbox)].copy()
        
        print(f"Clipped roads: {len(roads_clipped)} features")
        if 'RTTYP' in roads_clipped.columns:
            print(f"Road types (RTTYP): {roads_clipped['RTTYP'].value_counts().to_dict()}")
        
        return roads_clipped
    
    def get_target_grid_coordinates(self, hysplit_data: Dict[int, nc.Dataset]) -> pd.DataFrame:
        """
        Extract coordinate grid from HYSPLIT data with configurable grid size
        """
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
            # For ~1 arc-second DEM data, use larger window for 3km analysis
            slope, roughness = self.calculate_terrain_metrics(elevation_data, row['lat'], row['lon'], window_size=9)
            
            features_list.append({
                'month': row['month'],
                'grid_i': row['grid_i'],
                'grid_j': row['grid_j'],
                'elevation': elevation if elevation is not None else 1500,  # Default elevation
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
        # At ~40°N: 1° lon ≈ 85km, 1° lat ≈ 111km
        # 3km ≈ 0.035° lon, 0.027° lat, use conservative 0.03°
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
            # 10=Tree cover, 20=Shrubland, 30=Grassland, 40=Cropland, 50=Built-up, 
            # 60=Bare/sparse, 70=Snow/ice, 80=Water, 90=Wetland, 95=Mangroves
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
        print(f"Each grid cell aggregates ~90,000 pixels (10m -> 3km)")
        return pd.DataFrame(features_list)
    
    def aggregate_landcover_to_3km(self, landcover_data, bounds: Tuple[float, float, float, float]) -> Dict:
        """Aggregate 10m land cover data to 3km grid cell"""
        try:
            # Create geometry for masking
            bbox_geom = [box(*bounds)]
            
            # Mask raster to bounds - this extracts all 10m pixels within 3km cell
            masked_array, masked_transform = mask(landcover_data, bbox_geom, crop=True, filled=False)
            
            if masked_array.size == 0:
                return None
                
            masked_array = masked_array[0]  # Take first band
            
            # Handle masked arrays properly
            if hasattr(masked_array, 'mask'):
                # For masked arrays, get unmasked values
                valid_data = masked_array.compressed()
            else:
                # For regular arrays, exclude nodata values
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
            
            # Get dominant class (most pixels)
            dominant_class = max(valid_counts.items(), key=lambda x: x[1])[0]
            
            # Calculate diversity (number of classes with >1% coverage)
            diversity = sum(1 for pct in class_percentages.values() if pct > 1.0)
            
            return {
                'dominant_class': dominant_class,
                'diversity': diversity,
                'class_percentages': class_percentages,
                'total_pixels': total_pixels
            }
            
        except Exception as e:
            # Debug: Print error details occasionally
            if np.random.random() < 0.01:  # Print 1% of errors to avoid spam
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
                        # Convert to km (rough approximation at ~40°N)
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
                # Only consider major roads (Interstate, US, State highways)
                major_roads = roads_data[roads_data['RTTYP'].isin(['I', 'U', 'S'])]
                if len(major_roads) > 0:
                    distances = major_roads.geometry.distance(point)
                    min_distance = distances.min() * 111.32  # Convert degrees to km
                else:
                    min_distance = 999.0  # No major roads found
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
        """
        Extract satellite PM2.5 features optimized for smaller grid
        """
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
            
            # Extract features
            sat_at_target = self.get_nearest_grid_value(sat_lat_grid, sat_lon_grid, sat_pm25, target_lat, target_lon)
            sat_local_3km = self.extract_regional_mean_grid(sat_lat_grid, sat_lon_grid, sat_pm25, 
                                                          target_lat, target_lon, radius_km=3)
            sat_local_6km = self.extract_regional_mean_grid(sat_lat_grid, sat_lon_grid, sat_pm25, 
                                                          target_lat, target_lon, radius_km=6)
            sat_around_facility = self.extract_regional_mean_grid(sat_lat_grid, sat_lon_grid, sat_pm25, 
                                                               self.facility_lat, self.facility_lon, radius_km=2)
            
            # Gradient from facility to target
            sat_at_facility = self.get_nearest_grid_value(sat_lat_grid, sat_lon_grid, sat_pm25, 
                                                        self.facility_lat, self.facility_lon)
            sat_gradient = sat_at_target - sat_at_facility if (sat_at_target is not None and sat_at_facility is not None) else None
            
            # Get wind direction for upwind/downwind features
            wind_direction = self.get_monthly_wind_direction_grid(meteorology_data, month, target_lat, target_lon)
            
            # Directional features
            sat_upwind = None
            sat_downwind = None
            if wind_direction is not None:
                sat_upwind = self.extract_directional_mean_grid(sat_lat_grid, sat_lon_grid, sat_pm25, 
                                                              target_lat, target_lon, 
                                                              wind_direction + 180, radius_km=5, sector_width=60)
                
                sat_downwind = self.extract_directional_mean_grid(sat_lat_grid, sat_lon_grid, sat_pm25,
                                                                target_lat, target_lon,
                                                                wind_direction, radius_km=5, sector_width=60)
            
            features_list.append({
                'month': month,
                'grid_i': row['grid_i'],
                'grid_j': row['grid_j'],
                'satellite_at_target_cell': sat_at_target,
                'satellite_local_3km': sat_local_3km,
                'satellite_local_6km': sat_local_6km,
                'satellite_around_facility': sat_around_facility,
                'satellite_facility_to_cell_gradient': sat_gradient,
                'satellite_upwind_5km': sat_upwind,
                'satellite_downwind_5km': sat_downwind
            })
        
        return pd.DataFrame(features_list)
    
    def extract_meteorology_features_grid(self, met_data: xr.Dataset, 
                                        target_coords: pd.DataFrame) -> pd.DataFrame:
        """
        Extract meteorology features using grid-based approach
        """
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
    
    def extract_facility_features(self, target_coords: pd.DataFrame) -> pd.DataFrame:
        """
        Extract facility-related features using pyproj for better accuracy
        """
        features_list = []
        
        for _, row in target_coords.iterrows():
            target_lat, target_lon = row['lat'], row['lon']
            
            # Calculate distance and bearing using pyproj
            fwd_azimuth, back_azimuth, distance_m = self.geodesic.inv(
                self.facility_lon, self.facility_lat, target_lon, target_lat
            )
            
            distance_km = distance_m / 1000.0  # Convert to km
            bearing = (fwd_azimuth + 360) % 360  # Ensure 0-360 range
            
            features_list.append({
                'month': row['month'],
                'grid_i': row['grid_i'],
                'grid_j': row['grid_j'],
                'facility_lat': self.facility_lat,
                'facility_lon': self.facility_lon,
                'facility_height': self.facility_height,
                'distance_to_facility': distance_km,
                'bearing_from_facility': bearing
            })
        
        return pd.DataFrame(features_list)
    
    def extract_target_values(self, hysplit_data: Dict[int, nc.Dataset], 
                            log_transform: bool = False) -> pd.DataFrame:
        """
        Extract PM2.5 concentration target values with scaling (NO log transformation by default)
        
        Args:
            hysplit_data: Dictionary of HYSPLIT datasets by month
            log_transform: Whether to apply log10 transformation (default False)
            
        Returns:
            DataFrame with PM2.5 concentrations (scaled but not log-transformed)
        """
        targets_list = []
        
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
            
            for i, lat in enumerate(lats_subset):
                for j, lon in enumerate(lons_subset):
                    pm25_val = float(pm25_subset[i, j])
                    
                    # Handle potential masked values or invalid data
                    if np.ma.is_masked(pm25_val) or not np.isfinite(pm25_val):
                        pm25_val = np.nan
                    elif not np.isnan(pm25_val):
                        # Apply scaling factor
                        pm25_val = pm25_val * RMBC_scale_factor
                        
                        # Apply log transformation only if requested (default False)
                        if log_transform and pm25_val > 0:
                            epsilon = 1e-18
                            pm25_val = np.log10(pm25_val + epsilon)
                        
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
        print(f"\nTarget values (with scale factor {RMBC_scale_factor:.2e}):")
        print(f"  Range: {valid_values.min():.2e} to {valid_values.max():.2e}")
        print(f"  Mean: {valid_values.mean():.2e} ± {valid_values.std():.2e}")
        print(f"  Zero/negative values: {(valid_values <= 0).sum()} ({(valid_values <= 0).mean()*100:.1f}%)")
        
        return df
    
    # Utility functions for topographical processing
    def sample_raster_at_point(self, raster_data, lat: float, lon: float) -> Optional[float]:
        """Sample raster value at specific point"""
        try:
            # Convert point to raster row/col
            row, col = raster_data.index(lon, lat)
            
            # Check bounds
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
            # Get window around point
            row, col = elevation_data.index(lon, lat)
            
            # Define window bounds
            half_window = window_size // 2
            row_start = max(0, row - half_window)
            row_end = min(elevation_data.height, row + half_window + 1)
            col_start = max(0, col - half_window)
            col_end = min(elevation_data.width, col + half_window + 1)
            
            # Extract elevation window
            elevation_window = elevation_data.read(1)[row_start:row_end, col_start:col_end]
            
            # Remove nodata values
            valid_elevations = elevation_window[elevation_window != elevation_data.nodata]
            
            # Calculate slope and roughness
            if len(valid_elevations) > 1:
                slope = np.std(np.gradient(elevation_window.astype(float)))
                roughness = np.std(valid_elevations.astype(float))
                return float(slope), float(roughness)
            else:
                return None, None
                
        except Exception:
            return None, None
    
    # Grid-based utility functions (keep existing methods)
    def get_nearest_grid_value(self, lat_grid: np.ndarray, lon_grid: np.ndarray, 
                              values: np.ndarray, target_lat: float, target_lon: float) -> Optional[float]:
        """Find nearest grid cell value using haversine distance"""
        try:
            # Flatten grids for distance calculation
            lat_flat = lat_grid.flatten()
            lon_flat = lon_grid.flatten()
            values_flat = values.flatten()
            
            # Remove NaN values
            valid_mask = ~np.isnan(values_flat)
            if not np.any(valid_mask):
                return None
                
            lat_valid = lat_flat[valid_mask]
            lon_valid = lon_flat[valid_mask]
            values_valid = values_flat[valid_mask]
            
            # Calculate haversine distances using sklearn
            target_coords = np.radians([[target_lat, target_lon]])
            grid_coords = np.radians(np.column_stack([lat_valid, lon_valid]))
            
            distances = haversine_distances(target_coords, grid_coords)[0] * 6371000  # Convert to meters
            
            # Find nearest grid cell
            nearest_idx = np.argmin(distances)
            return float(values_valid[nearest_idx])
            
        except Exception as e:
            print(f"Nearest grid value extraction failed: {e}")
            return None
    
    def extract_regional_mean_grid(self, lat_grid: np.ndarray, lon_grid: np.ndarray, values: np.ndarray,
                                 center_lat: float, center_lon: float, radius_km: float) -> Optional[float]:
        """Extract mean value within radius using grid cells"""
        try:
            # Flatten grids
            lat_flat = lat_grid.flatten()
            lon_flat = lon_grid.flatten()
            values_flat = values.flatten()
            
            # Remove NaN values
            valid_mask = ~np.isnan(values_flat)
            if not np.any(valid_mask):
                return None
                
            lat_valid = lat_flat[valid_mask]
            lon_valid = lon_flat[valid_mask]
            values_valid = values_flat[valid_mask]
            
            # Calculate distances using sklearn haversine
            center_coords = np.radians([[center_lat, center_lon]])
            grid_coords = np.radians(np.column_stack([lat_valid, lon_valid]))
            
            distances_km = haversine_distances(center_coords, grid_coords)[0] * 6371  # Earth radius in km
            
            # Create mask for points within radius
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
            # Flatten grids
            lat_flat = lat_grid.flatten()
            lon_flat = lon_grid.flatten()
            values_flat = values.flatten()
            
            # Remove NaN values
            valid_mask = ~np.isnan(values_flat)
            if not np.any(valid_mask):
                return None
                
            lat_valid = lat_flat[valid_mask]
            lon_valid = lon_flat[valid_mask]
            values_valid = values_flat[valid_mask]
            
            # Calculate distances using sklearn haversine
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


# Usage example
if __name__ == "__main__":
    # Facility coordinates: Suncor (lat, lon, height_m_AGL)
    suncor_coords = (39.78741, -105.11498, 10)
    
    # Initialize processor with configurable grid size
    processor = AirQualityDataProcessor(grid_size=24)  # 24x24 grid instead of 41x41
    
    # Option 1: Try without topographical features first (recommended for debugging)
    print("=== PROCESSING WITHOUT TOPOGRAPHICAL FEATURES ===")
    try:
        X_features, y_target = processor.process_training_data(
            facility_coords=suncor_coords,
            year=2023,
            include_topographical=False  # Skip topographical features
        )
        
        if len(X_features) > 0:
            print(f"✓ SUCCESS: {len(X_features)} samples with atmospheric features only")
        else:
            print("✗ FAILED: No valid samples even without topographical features")
            
    except Exception as e:
        print(f"✗ ERROR in atmospheric processing: {e}")
    
    # Option 2: Try with topographical features (if atmospheric worked)
    if len(X_features) > 0:
        print("\n=== PROCESSING WITH TOPOGRAPHICAL FEATURES ===")
        try:
            X_features_full, y_target_full = processor.process_training_data(
                facility_coords=suncor_coords,
                year=2023,
                include_topographical=True  # Include all features
            )
            
            if len(X_features_full) > 0:
                print(f"✓ SUCCESS: {len(X_features_full)} samples with all features")
                X_features, y_target = X_features_full, y_target_full  # Use full dataset
            else:
                print("✗ FAILED: Topographical features causing issues, using atmospheric only")
                
        except Exception as e:
            print(f"✗ ERROR in topographical processing: {e}")
            print("Using atmospheric features only")
    
    # Print final results
    print("\n=== FINAL PROCESSING SUMMARY ===")
    print(f"Grid size: {processor.grid_size}x{processor.grid_size}")
    print(f"Total samples: {len(X_features)}")
    print(f"Features: {len(X_features.columns)}")
    
    if len(X_features) > 0:
        print("\n=== FEATURE CATEGORIES ===")
        feature_categories = {
            'satellite': [col for col in X_features.columns if 'satellite' in col],
            'meteorology': [col for col in X_features.columns if any(x in col for x in ['temp', 'wind', 'pressure', 'precipitation', 'dewpoint'])],
            'facility': [col for col in X_features.columns if any(x in col for x in ['facility', 'distance', 'bearing'])],
            'elevation': [col for col in X_features.columns if any(x in col for x in ['elevation', 'terrain', 'slope', 'roughness'])],
            'landcover': [col for col in X_features.columns if 'landcover' in col],
            'roads': [col for col in X_features.columns if 'road' in col]
        }
        
        for category, features in feature_categories.items():
            if features:
                print(f"{category.upper()}: {len(features)} features")
                for feature in features[:2]:  # Show first 2
                    print(f"  - {feature}")
        
        print("\nTarget statistics:")
        print(y_target['pm25_concentration'].describe())
        
        # Save processed data
        output_dir = Path(f"{DATA_PATH}/processed_data")
        output_dir.mkdir(exist_ok=True)
        
        X_features.to_csv(output_dir / f"X_features_RMBC_2023_grid{processor.grid_size}.csv", index=False)
        y_target.to_csv(output_dir / f"y_target_RMBC_2023_grid{processor.grid_size}.csv", index=False)
        
        print(f"\nData saved to {output_dir}")
        print("✓ Ready for ML model training!")
    else:
        print("\n✗ No valid data generated. Check your input files and coordinate systems.")