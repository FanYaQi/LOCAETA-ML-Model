"""
Base data processor with common functionality for air quality data processing
"""
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

from utils.path_util import DATA_PATH


class BaseAirQualityProcessor:
    """
    Base processor for satellite, meteorology, HYSPLIT, and topographical data
    """
    
    def __init__(self, grid_size: int = 24):
        """Initialize processor with configurable grid size"""
        self.geodesic = pyproj.Geod(ellps='WGS84')
        self.grid_size = grid_size
        print(f"Initialized with {grid_size}x{grid_size} grid")
    
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
    
    # Utility functions for grid-based processing
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