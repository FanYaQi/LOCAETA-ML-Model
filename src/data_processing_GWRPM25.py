import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os
import pandas as pd
from utils.path_util import DATA_PATH

def find_coord_names(ds):
    """Find latitude and longitude coordinate names in dataset"""
    lat_names = ['lat', 'latitude', 'LAT', 'LATITUDE', 'y']
    lon_names = ['lon', 'longitude', 'LON', 'LONGITUDE', 'x']
    
    lat_coord = None
    lon_coord = None
    
    for name in lat_names:
        if name in ds.coords or name in ds.dims:
            lat_coord = name
            break
            
    for name in lon_names:
        if name in ds.coords or name in ds.dims:
            lon_coord = name
            break
    
    return lat_coord, lon_coord

def crop_satellite_data(satellite_file, bbox, buffer_deg=0.1):
    """
    Crop satellite PM2.5 data to bounding box with buffer
    
    Parameters:
    -----------
    satellite_file : str
        Path to satellite PM2.5 netCDF file
    bbox : list
        Bounding box as [lat_max, lon_min, lat_min, lon_max]
    buffer_deg : float
        Buffer zone in degrees (default 0.1)
        
    Returns:
    --------
    xarray.Dataset
        Cropped satellite data
    """
    ds = xr.open_dataset(satellite_file)
    
    # Find coordinate names
    lat_coord, lon_coord = find_coord_names(ds)
    if not lat_coord or not lon_coord:
        raise ValueError(f"Could not identify coordinates in {satellite_file}")
    
    # Parse bounding box: [lat_max, lon_min, lat_min, lon_max]
    lat_max, lon_min, lat_min, lon_max = bbox
    
    # Add buffer
    lat_min_buf = lat_min - buffer_deg
    lat_max_buf = lat_max + buffer_deg
    lon_min_buf = lon_min - buffer_deg
    lon_max_buf = lon_max + buffer_deg
    
    # Crop the data
    cropped = ds.sel({
        lat_coord: slice(lat_min_buf, lat_max_buf),
        lon_coord: slice(lon_min_buf, lon_max_buf)
    })
    
    # Add metadata
    cropped.attrs['cropped_to_bbox'] = f"{lat_min}, {lon_min}, {lat_max}, {lon_max}"
    cropped.attrs['buffer_degrees'] = buffer_deg
    cropped.attrs['original_file'] = satellite_file
    
    print(f"Cropped {os.path.basename(satellite_file)}: {dict(cropped.dims)}")
    
    return cropped

def process_gwrpm25_12_months(bbox, buffer_deg=0.1, year=2023):
    """
    Process 12 months of GWRPM25 data
    
    Parameters:
    -----------
    bbox : list
        Bounding box as [lat_max, lon_min, lat_min, lon_max]
    buffer_deg : float
        Buffer zone in degrees
    year : int
        Year to process
        
    Returns:
    --------
    xarray.Dataset
        Combined annual dataset
    """
    
    # Define paths using DATA_PATH
    output_path = f"{DATA_PATH}/GWRPM25_AOI/"
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    print(f"=== Processing {year} GWRPM25 Data ===")
    print(f"Bbox: {bbox}")
    print(f"Buffer: {buffer_deg}°")
    print(f"Output: {output_path}")
    
    all_cropped_data = []
    
    # Process each month
    for month in range(1, 13):
        month_str = f"{month:02d}"
        
        # File paths using DATA_PATH
        input_file = f"{DATA_PATH}/GWRPM25_raw/V5GL0502.HybridPM25.NorthAmerica.{year}{month_str}-{year}{month_str}.nc"
        output_file = f"{DATA_PATH}/GWRPM25_AOI/GWRPM25_AOI_{year}{month_str}.nc"
        
        print(f"Processing {year}-{month_str}")
        
        # Crop and save
        cropped_ds = crop_satellite_data(input_file, bbox, buffer_deg)
        cropped_ds.attrs['month'] = month
        cropped_ds.attrs['year'] = year
        
        # Save individual month
        cropped_ds.to_netcdf(output_file)
        all_cropped_data.append(cropped_ds)
    
    # Create combined annual dataset
    print("\nCreating combined annual dataset...")
    combined_ds = xr.concat(all_cropped_data, dim='time')
    combined_ds.attrs['description'] = f'GWRPM25 AOI data for {year} (all months)'
    combined_ds.attrs['bbox'] = bbox
    combined_ds.attrs['buffer_degrees'] = buffer_deg
    
    # Save combined dataset
    annual_output = f"{DATA_PATH}/GWRPM25_AOI/GWRPM25_AOI_{year}_annual.nc"
    combined_ds.to_netcdf(annual_output)
    
    print(f"✓ Processed 12 months")
    print(f"✓ Individual files: {DATA_PATH}/GWRPM25_AOI/GWRPM25_AOI_{year}XX.nc")
    print(f"✓ Annual file: {annual_output}")
    
    return combined_ds

def process_single_month(bbox, month, year=2023, buffer_deg=0.1):
    """
    Process a single month of GWRPM25 data
    
    Parameters:
    -----------
    bbox : list
        Bounding box as [lat_max, lon_min, lat_min, lon_max]
    month : int
        Month number (1-12)
    year : int
        Year
    buffer_deg : float
        Buffer zone in degrees
        
    Returns:
    --------
    xarray.Dataset
        Cropped dataset
    """
    
    # Define paths using DATA_PATH
    output_path = f"{DATA_PATH}/GWRPM25_AOI/"
    os.makedirs(output_path, exist_ok=True)
    
    month_str = f"{month:02d}"
    input_file = f"{DATA_PATH}/GWRPM25_raw/V5GL0502.HybridPM25.NorthAmerica.{year}{month_str}-{year}{month_str}.nc"
    output_file = f"{DATA_PATH}/GWRPM25_AOI/GWRPM25_AOI_{year}{month_str}.nc"
    
    print(f"Processing {year}-{month_str}")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    
    # Process
    cropped_ds = crop_satellite_data(input_file, bbox, buffer_deg)
    cropped_ds.attrs['month'] = month
    cropped_ds.attrs['year'] = year
    
    # Save
    cropped_ds.to_netcdf(output_file)
    print(f"✓ Saved: {output_file}")
    
    return cropped_ds

def plot_monthly_data(dataset, variable_name=None):
    """Plot first month of data for quality check"""
    
    if variable_name is None:
        data_vars = list(dataset.data_vars.keys())
        variable_name = data_vars[0] if data_vars else None
    
    if variable_name is None:
        print("No data variables found")
        return
    
    # Find coordinates
    lat_coord, lon_coord = find_coord_names(dataset)
    
    # Plot first time step
    data_to_plot = dataset[variable_name]
    if 'time' in data_to_plot.dims:
        data_to_plot = data_to_plot.isel(time=0)
    
    plt.figure(figsize=(10, 8))
    data_to_plot.plot(x=lon_coord, y=lat_coord, cmap='viridis')
    plt.title(f'{variable_name} - Cropped GWRPM25 Data')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Your bounding box
    bbox = [40.32, -105.5, 39.41, -104.41]  # [lat_max, lon_min, lat_min, lon_max]
    
    # Process all 12 months
    combined_data = process_gwrpm25_12_months(bbox)
    
    # Plot first month for quality check
    plot_monthly_data(combined_data)
    
    # Or process single month
    # january_data = process_single_month(bbox, month=1)