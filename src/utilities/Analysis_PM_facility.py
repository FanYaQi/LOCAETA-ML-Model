import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.features import geometry_mask
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy import stats
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from collections import defaultdict

import warnings
warnings.filterwarnings('ignore')


class PM25FacilityAnalyzer:
    """
    Simple analyzer for PM2.5 and facility relationships
    """
    
    def __init__(self, pm25_tiff_path, facilities_geojson_path, column_config=None, max_pm25=200):
        """
        column_config : dict (required)
          Dictionary with 'required' and 'optional' keys:
          {
              'required': {
                  'name': 'actual_name_column',
                  'facility_type': 'actual_facility_column', 
                  'state': 'actual_state_column'
              },
              'optional': {
                  'capacity': 'actual_capacity_column',
                  'pm_emissions': 'actual_emissions_column',
                  'any_other_field': 'actual_column_name'
              }
          }"""
        print("Loading data...")
        
        # Load PM2.5 raster
        self.pm25_raster = rasterio.open(pm25_tiff_path)
        self.pm25_data = self.pm25_raster.read(1)
        self.pm25_transform = self.pm25_raster.transform

        # Clean PM2.5 data (remove extreme values and NaN)
        cleaned_data = self.pm25_data.copy()
        cleaned_data[(self.pm25_data > max_pm25) | (self.pm25_data <= 0) | np.isnan(self.pm25_data)] = np.nan
        self.pm25_data = cleaned_data
        
        # Store column configuration
        if column_config is None:
            raise ValueError("column_config is required! Must specify facility data column mappings.")
        
        if 'required' not in column_config:
            raise ValueError("column_config must have 'required' key")
        
        # Validate required columns
        required_fields = ['name', 'facility_type', 'state']
        required_config = column_config['required']
        missing_fields = [field for field in required_fields if field not in required_config]
        if missing_fields:
            raise ValueError(f"Missing required field mappings: {missing_fields}")
        self.column_config = column_config
        print(f"✓ Column configuration validated")
        
        # Get all valid PM2.5 values for overall distribution
        self.all_pm25_values = self.pm25_data[~np.isnan(self.pm25_data)]
        
        # Load facilities
        self.facilities = gpd.read_file(facilities_geojson_path)
        if self.facilities.crs != self.pm25_raster.crs:
            self.facilities = self.facilities.to_crs(self.pm25_raster.crs)
        
        print(f"✓ Loaded PM2.5 raster: {self.pm25_data.shape}")
        print(f"✓ Valid PM2.5 pixels: {len(self.all_pm25_values):,}")
        print(f"✓ PM2.5 range: {self.all_pm25_values.min():.1f} - {self.all_pm25_values.max():.1f} μg/m³")
        print(f"✓ Loaded {len(self.facilities)} facilities")

    def haversine_distance(self, lon1, lat1, lon2, lat2):
        """
        Calculate haversine distance between two points in kilometers
        """
        R = 6371  # Earth's radius in km
        
        # Convert to radians
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c    
    
    def get_facility_pm25(self, buffer_radius=0):
        """
        Get PM2.5 values for facilities with configurable buffer
        Uses the column_config specified during initialization
        
        Parameters:
        buffer_radius : int
            0 = point location only
            1 = 3x3 grid around facility  
            2 = 5x5 grid around facility
            etc.
        """
        buffer_desc = "point location" if buffer_radius == 0 else f"{2*buffer_radius+1}×{2*buffer_radius+1} buffer"
        print(f"Calculating PM2.5 for facilities ({buffer_desc})...")
        
        # Use stored column configuration
        required_config = self.column_config['required']
        optional_config = self.column_config.get('optional', {})
        
        facility_data = []
        height, width = self.pm25_data.shape
        
        for idx, facility in self.facilities.iterrows():
            # Get facility coordinates
            if facility.geometry.geom_type == 'Point':
                x, y = facility.geometry.x, facility.geometry.y
            else:
                centroid = facility.geometry.centroid
                x, y = centroid.x, centroid.y
            
            # Convert to raster grid position
            row, col = rasterio.transform.rowcol(self.pm25_transform, x, y)
            
            # Collect PM2.5 values within buffer
            pm25_values = []
            
            if buffer_radius == 0:
                # Point location only
                if 0 <= row < height and 0 <= col < width:
                    pm25_val = self.pm25_data[row, col]
                    if not np.isnan(pm25_val):
                        pm25_values.append(pm25_val)
            else:
                # Buffer area around facility
                for dr in range(-buffer_radius, buffer_radius + 1):
                    for dc in range(-buffer_radius, buffer_radius + 1):
                        r = row + dr
                        c = col + dc
                        if 0 <= r < height and 0 <= c < width:
                            pm25_val = self.pm25_data[r, c]
                            if not np.isnan(pm25_val):
                                pm25_values.append(pm25_val)
            
            # Calculate max PM2.5
            max_pm25 = np.max(pm25_values) if pm25_values else np.nan
            
            # Build facility data with required fields
            facility_info = {
                'facility_id': idx,
                'plant_name': facility.get(required_config['name'], 'Unknown'),
                'facility_type': facility.get(required_config['facility_type'], 'Unknown'),
                'state': facility.get(required_config['state'], 'Unknown'),
                'pm25': max_pm25,
                'sample_count': len(pm25_values),
                'buffer_radius': buffer_radius
            }
            
            # Add all optional fields dynamically
            for field_name, column_name in optional_config.items():
                # Set appropriate default values based on field name or data type
                default_value = 0 if 'capacity' in field_name.lower() else np.nan
                facility_info[field_name] = facility.get(column_name, default_value)
            
            facility_data.append(facility_info)
        
        facility_df = pd.DataFrame(facility_data)
        valid_count = (~np.isnan(facility_df['pm25'])).sum()
        
        print(f"✓ Facilities with valid PM2.5: {valid_count}")
        print(f"✓ Columns included: {list(facility_df.columns)}")
        
        return facility_df  

    # Add these methods to your existing PM25FacilityAnalyzer class

    def create_facility_presence_mask(self, buffer_radius=0):
        """
        Create a binary mask where 1 = pixel contains or is within buffer of a facility, 0 = no facility
        
        Parameters:
        buffer_radius : int
            0 = point location only
            1 = 3x3 grid around facility  
            2 = 5x5 grid around facility
            etc.
        
        Returns:
        facility_mask : numpy array (same shape as PM2.5 data)
            Binary mask: 1 where facility present, 0 otherwise
        """
        print(f"Creating facility presence mask (buffer_radius={buffer_radius})...")
        
        height, width = self.pm25_data.shape
        facility_mask = np.zeros((height, width), dtype=int)
        
        for idx, facility in self.facilities.iterrows():
            # Get facility coordinates
            if facility.geometry.geom_type == 'Point':
                x, y = facility.geometry.x, facility.geometry.y
            else:
                centroid = facility.geometry.centroid
                x, y = centroid.x, centroid.y
            
            # Convert to raster grid position
            row, col = rasterio.transform.rowcol(self.pm25_transform, x, y)
            
            # Mark pixels within buffer
            if buffer_radius == 0:
                # Point location only
                if 0 <= row < height and 0 <= col < width:
                    facility_mask[row, col] = 1
            else:
                # Buffer area around facility
                for dr in range(-buffer_radius, buffer_radius + 1):
                    for dc in range(-buffer_radius, buffer_radius + 1):
                        r = row + dr
                        c = col + dc
                        if 0 <= r < height and 0 <= c < width:
                            facility_mask[r, c] = 1
        
        facility_pixels = np.sum(facility_mask)
        total_pixels = height * width
        print(f"✓ Facility presence mask created")
        print(f"✓ Facility pixels: {facility_pixels:,} ({facility_pixels/total_pixels*100:.2f}%)")
        print(f"✓ Non-facility pixels: {total_pixels-facility_pixels:,} ({(total_pixels-facility_pixels)/total_pixels*100:.2f}%)")
        
        return facility_mask

    def analyze_pixel_correlation(self, buffer_radius=0, sample_size=None, random_seed=42):
        """
        Analyze correlation between PM2.5 concentration (continuous) and facility presence (binary)
        at the pixel level with configurable buffer radius.
        
        Parameters:
        buffer_radius : int
            Buffer radius around facilities (0 = point only, 1 = 3x3, 2 = 5x5, etc.)
        sample_size : int, optional
            Number of pixels to sample for analysis (default: all valid pixels)
        random_seed : int
            Random seed for reproducible sampling
        
        Returns:
        dict: Statistical results including correlation and t-test
        """
        print(f"\n{'='*60}")
        print(f"PIXEL-LEVEL CORRELATION ANALYSIS")
        print(f"Buffer radius: {buffer_radius} ({'point' if buffer_radius == 0 else f'{2*buffer_radius+1}x{2*buffer_radius+1} grid'})")
        print(f"{'='*60}")
        
        np.random.seed(random_seed)
        
        # Step 1: Create facility presence mask
        facility_mask = self.create_facility_presence_mask(buffer_radius=buffer_radius)
        
        # Step 2: Get valid PM2.5 pixels (remove NaN values)
        valid_pm_mask = ~np.isnan(self.pm25_data)
        
        # Step 3: Combine masks to get analyzable pixels
        analyzable_mask = valid_pm_mask
        pm25_values = self.pm25_data[analyzable_mask]
        facility_presence = facility_mask[analyzable_mask]
        
        print(f"\n📊 DATA PREPARATION:")
        print(f"   • Total pixels: {self.pm25_data.size:,}")
        print(f"   • Valid PM2.5 pixels: {np.sum(valid_pm_mask):,}")
        print(f"   • Analyzable pixels: {len(pm25_values):,}")
        
        # Step 4: Sample if requested
        if sample_size is not None and sample_size < len(pm25_values):
            sample_indices = np.random.choice(len(pm25_values), size=sample_size, replace=False)
            pm25_values = pm25_values[sample_indices]
            facility_presence = facility_presence[sample_indices]
            print(f"   • Sampled pixels for analysis: {len(pm25_values):,}")
        
        # Step 5: Separate facility vs non-facility pixels
        facility_pixels = pm25_values[facility_presence == 1]
        non_facility_pixels = pm25_values[facility_presence == 0]
        
        print(f"\n🏭 FACILITY PRESENCE BREAKDOWN:")
        print(f"   • Pixels with facilities: {len(facility_pixels):,}")
        print(f"   • Pixels without facilities: {len(non_facility_pixels):,}")
        
        if len(facility_pixels) == 0:
            print("❌ ERROR: No facility pixels found. Cannot perform analysis.")
            return None
        
        if len(non_facility_pixels) == 0:
            print("❌ ERROR: No non-facility pixels found. Cannot perform analysis.")
            return None
        
        # Step 6: Calculate descriptive statistics
        facility_stats = {
            'mean': np.mean(facility_pixels),
            'std': np.std(facility_pixels),
            'median': np.median(facility_pixels),
            'min': np.min(facility_pixels),
            'max': np.max(facility_pixels)
        }
        
        non_facility_stats = {
            'mean': np.mean(non_facility_pixels),
            'std': np.std(non_facility_pixels),
            'median': np.median(non_facility_pixels),
            'min': np.min(non_facility_pixels),
            'max': np.max(non_facility_pixels)
        }
        
        print(f"\n📈 DESCRIPTIVE STATISTICS:")
        print(f"   Facility pixels PM2.5 (μg/m³):")
        print(f"      Mean: {facility_stats['mean']:.2f} ± {facility_stats['std']:.2f}")
        print(f"      Median: {facility_stats['median']:.2f}")
        print(f"      Range: {facility_stats['min']:.2f} - {facility_stats['max']:.2f}")
        print(f"   Non-facility pixels PM2.5 (μg/m³):")
        print(f"      Mean: {non_facility_stats['mean']:.2f} ± {non_facility_stats['std']:.2f}")
        print(f"      Median: {non_facility_stats['median']:.2f}")
        print(f"      Range: {non_facility_stats['min']:.2f} - {non_facility_stats['max']:.2f}")
        
        # Step 7: Point-biserial correlation
        correlation_coef, correlation_p = stats.pointbiserialr(facility_presence, pm25_values)
        
        # Step 8: Independent samples t-test (Welch's t-test)
        t_statistic, t_p_value = stats.ttest_ind(facility_pixels, non_facility_pixels, equal_var=False)
        
        # Step 9: Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(facility_pixels) - 1) * facility_stats['std']**2 + 
                            (len(non_facility_pixels) - 1) * non_facility_stats['std']**2) / 
                            (len(facility_pixels) + len(non_facility_pixels) - 2))
        cohens_d = (facility_stats['mean'] - non_facility_stats['mean']) / pooled_std
        
        # Step 10: Additional tests
        # Mann-Whitney U test (non-parametric alternative)
        u_statistic, u_p_value = stats.mannwhitneyu(facility_pixels, non_facility_pixels, 
                                                    alternative='two-sided')
        
        # Levene's test for equal variances
        levene_stat, levene_p = stats.levene(facility_pixels, non_facility_pixels)
        
        # Step 11: Compile results
        results = {
            'buffer_radius': buffer_radius,
            'sample_info': {
                'total_pixels': len(pm25_values),
                'facility_pixels': len(facility_pixels),
                'non_facility_pixels': len(non_facility_pixels)
            },
            'descriptive_stats': {
                'facility': facility_stats,
                'non_facility': non_facility_stats
            },
            'correlation': {
                'point_biserial_r': correlation_coef,
                'p_value': correlation_p
            },
            't_test': {
                't_statistic': t_statistic,
                'p_value': t_p_value,
                'cohens_d': cohens_d
            },
            'mann_whitney': {
                'u_statistic': u_statistic,
                'p_value': u_p_value
            },
            'levene_test': {
                'statistic': levene_stat,
                'p_value': levene_p
            },
            'raw_data': {
                'pm25_values': pm25_values,
                'facility_presence': facility_presence
            }
        }
        
        # Step 12: Print results
        print(f"\n🔍 STATISTICAL ANALYSIS RESULTS:")
        print(f"   Point-biserial correlation:")
        print(f"      r = {correlation_coef:.4f}")
        print(f"      p-value = {correlation_p:.6f}")
        
        print(f"   Independent samples t-test:")
        print(f"      t-statistic = {t_statistic:.4f}")
        print(f"      p-value = {t_p_value:.6f}")
        print(f"      Cohen's d = {cohens_d:.4f}")
        
        print(f"   Mann-Whitney U test:")
        print(f"      U-statistic = {u_statistic:.0f}")
        print(f"      p-value = {u_p_value:.6f}")
        
        print(f"   Levene's test (equal variances):")
        print(f"      Statistic = {levene_stat:.4f}")
        print(f"      p-value = {levene_p:.6f}")
        
        # Step 13: Interpretation
        print(f"\n💡 INTERPRETATION:")
        
        # Correlation interpretation
        if abs(correlation_coef) < 0.1:
            corr_strength = "negligible"
        elif abs(correlation_coef) < 0.3:
            corr_strength = "small"
        elif abs(correlation_coef) < 0.5:
            corr_strength = "medium"
        else:
            corr_strength = "large"
        
        corr_direction = "positive" if correlation_coef > 0 else "negative"
        
        print(f"   • Correlation: {corr_strength} {corr_direction} correlation")
        if correlation_p < 0.05:
            print(f"   • The correlation is statistically significant (p < 0.05)")
        else:
            print(f"   • The correlation is not statistically significant (p >= 0.05)")
        
        # Effect size interpretation
        if abs(cohens_d) < 0.2:
            effect_size = "small"
        elif abs(cohens_d) < 0.5:
            effect_size = "medium"
        elif abs(cohens_d) < 0.8:
            effect_size = "large"
        else:
            effect_size = "very large"
        
        print(f"   • Effect size: {effect_size} (Cohen's d = {cohens_d:.3f})")
        
        # T-test interpretation
        if t_p_value < 0.05:
            print(f"   • There IS a statistically significant difference in PM2.5 between facility and non-facility pixels")
            if facility_stats['mean'] > non_facility_stats['mean']:
                print(f"   • Facility pixels have significantly HIGHER PM2.5 concentrations")
            else:
                print(f"   • Facility pixels have significantly LOWER PM2.5 concentrations")
        else:
            print(f"   • There is NO statistically significant difference in PM2.5 between groups")
        
        # Variance equality
        if levene_p < 0.05:
            print(f"   • The groups have significantly different variances (heteroscedastic)")
        else:
            print(f"   • The groups have similar variances (homoscedastic)")
        
        print(f"\n{'='*60}")
        
        return results

    def plot_correlation_analysis(self, results, figsize=(16, 12), 
                                facility_dist='beta', facility_params=(3, 2),
                                non_facility_dist='norm', non_facility_params=None,
                                include_nearest_neighbor=True, nn_percentile=95, nn_bin_width=0.5):
        """
        Create comprehensive visualization of correlation analysis results with configurable Q-Q plots
        
        Parameters:
        results : dict
            Results from analyze_pixel_correlation
        figsize : tuple
            Figure size
        facility_dist : str
            Distribution for facility pixels Q-Q plot ('norm', 'beta', 'lognorm', 'uniform', etc.)
        facility_params : tuple
            Parameters for facility distribution (e.g., (3, 2) for beta(3,2))
        non_facility_dist : str
            Distribution for non-facility pixels Q-Q plot
        non_facility_params : tuple
            Parameters for non-facility distribution
        include_nearest_neighbor : bool
            Whether to include nearest neighbor analysis in plots 4 & 5
        nn_percentile : float
            Percentile for defining PM2.5 hotspots in nearest neighbor analysis
        nn_bin_width : float
            Bin width for nearest neighbor distance histogram
        """
        if results is None:
            print("No results to plot")
            return None
        
        from scipy import stats as scipy_stats
        
        # Extract data
        pm25_values = results['raw_data']['pm25_values']
        facility_presence = results['raw_data']['facility_presence']
        facility_pixels = pm25_values[facility_presence == 1]
        non_facility_pixels = pm25_values[facility_presence == 0]
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle(f'Satellite PM2.5 and Facility Correlation Analysis\nBuffer Radius: {results["buffer_radius"]} km'
                    if results["buffer_radius"] > 0 else '(point locations)',
                    fontsize=16, fontweight='bold')
        
        # Plot 1: Box plot comparison
        ax1 = axes[0, 0]
        ax1.text(-0.1, 1.05, 'A)', transform=ax1.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')
        box_data = [non_facility_pixels, facility_pixels]
        box_labels = ['No Facility', 'Facility']
        
        bp = ax1.boxplot(box_data, labels=box_labels, patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('orange')
        
        ax1.set_ylabel('PM2.5 Concentration (μg/m³)')
        ax1.set_title('PM2.5 Distribution by Facility Presence')
        ax1.grid(True, alpha=0.3)
        
        # Add statistical annotation
        mean_diff = results['descriptive_stats']['facility']['mean'] - results['descriptive_stats']['non_facility']['mean']
        ax1.text(0.5, 0.95, f"Mean difference: {mean_diff:.2f} μg/m³\nt = {results['t_test']['t_statistic']:.3f}, p = {results['t_test']['p_value']:.4f}",
                transform=ax1.transAxes, ha='center', va='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
        
        # Plot 2: Histograms with distribution curves
        ax2 = axes[0, 1]
        ax2.text(-0.1, 1.05, 'B)', transform=ax2.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')
        bins = np.linspace(min(pm25_values), max(pm25_values), 50)
        
        ax2.hist(non_facility_pixels, bins=bins, alpha=0.7, label='No Facility', color='lightblue', density=True)
        ax2.hist(facility_pixels, bins=bins, alpha=0.7, label='Facility', color='orange', density=True)
        
        # Add distribution curves
        x_range = np.linspace(min(pm25_values), max(pm25_values), 200)
        
        # Non-facility distribution curve
        try:
            if non_facility_dist == 'beta' and non_facility_params:
                # Scale to [0,1] for beta, then scale back
                nf_scaled = (non_facility_pixels - non_facility_pixels.min()) / (non_facility_pixels.max() - non_facility_pixels.min() + 1e-8)
                x_scaled = np.linspace(0, 1, 200)
                y_beta = scipy_stats.beta.pdf(x_scaled, non_facility_params[0], non_facility_params[1])
                # Scale back to original range
                x_orig = x_scaled * (non_facility_pixels.max() - non_facility_pixels.min()) + non_facility_pixels.min()
                y_scaled = y_beta / (non_facility_pixels.max() - non_facility_pixels.min())
                ax2.plot(x_orig, y_scaled, 'blue', linewidth=2, label=f'Non-facility β({non_facility_params[0]},{non_facility_params[1]})')
            elif non_facility_dist == 'norm':
                mean_nf = results['descriptive_stats']['non_facility']['mean']
                std_nf = results['descriptive_stats']['non_facility']['std']
                y_norm = scipy_stats.norm.pdf(x_range, mean_nf, std_nf)
                ax2.plot(x_range, y_norm, 'blue', linewidth=2, label='Non-facility Normal')
            elif non_facility_params:
                dist_obj = getattr(scipy_stats, non_facility_dist)
                y_dist = dist_obj.pdf(x_range, *non_facility_params)
                ax2.plot(x_range, y_dist, 'blue', linewidth=2, label=f'Non-facility {non_facility_dist}')
        except Exception as e:
            print(f"Could not plot non-facility distribution curve: {e}")
        
        # Facility distribution curve  
        try:
            if facility_dist == 'beta' and facility_params:
                # Scale to [0,1] for beta, then scale back
                f_scaled = (facility_pixels - facility_pixels.min()) / (facility_pixels.max() - facility_pixels.min() + 1e-8)
                x_scaled = np.linspace(0, 1, 200)
                y_beta = scipy_stats.beta.pdf(x_scaled, facility_params[0], facility_params[1])
                # Scale back to original range
                x_orig = x_scaled * (facility_pixels.max() - facility_pixels.min()) + facility_pixels.min()
                y_scaled = y_beta / (facility_pixels.max() - facility_pixels.min())
                ax2.plot(x_orig, y_scaled, 'red', linewidth=2, label=f'Facility β({facility_params[0]},{facility_params[1]})')
            elif facility_dist == 'norm':
                mean_f = results['descriptive_stats']['facility']['mean']
                std_f = results['descriptive_stats']['facility']['std']
                y_norm = scipy_stats.norm.pdf(x_range, mean_f, std_f)
                ax2.plot(x_range, y_norm, 'red', linewidth=2, label='Facility Normal')
            elif facility_params:
                dist_obj = getattr(scipy_stats, facility_dist)
                y_dist = dist_obj.pdf(x_range, *facility_params)
                ax2.plot(x_range, y_dist, 'red', linewidth=2, label=f'Facility {facility_dist}')
        except Exception as e:
            print(f"Could not plot facility distribution curve: {e}")
        
        ax2.set_xlabel('PM2.5 Concentration (μg/m³)')
        ax2.set_ylabel('Density')
        ax2.set_title('PM2.5 Distribution Comparison with Fitted Curves')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Scatter plot with correlation
        ax3 = axes[0, 2]
        ax3.text(-0.1, 1.05, 'C)', transform=ax3.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')
        
        # Add some jitter to binary variable for better visualization
        jittered_facility = facility_presence + np.random.normal(0, 0.05, len(facility_presence))
        
        ax3.scatter(jittered_facility, pm25_values, alpha=0.3, s=1)
        
        # Add correlation line
        z = np.polyfit(facility_presence, pm25_values, 1)
        p = np.poly1d(z)
        ax3.plot([0, 1], p([0, 1]), "r--", alpha=0.8, linewidth=2)
        
        ax3.set_xlim(-0.2, 1.2)
        ax3.set_xlabel('Facility Presence (0=No, 1=Yes)')
        ax3.set_ylabel('PM2.5 Concentration (μg/m³)')
        ax3.set_title('Point-Biserial Correlation')
        ax3.set_xticks([0, 1])
        ax3.set_xticklabels(['No Facility', 'Facility'])
        ax3.grid(True, alpha=0.3)
        
        # Add correlation text
        ax3.text(0.05, 0.95, f"r = {results['correlation']['point_biserial_r']:.4f}\np = {results['correlation']['p_value']:.4f}",
                transform=ax3.transAxes, va='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
        
        # Plot 4: Nearest Neighbor Distance Histogram
        ax4 = axes[1, 0]
        ax4.text(-0.1, 1.05, 'D)', transform=ax4.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')
        
        if include_nearest_neighbor:
            try:
                print("Running nearest neighbor analysis for correlation plot...")
                # Run the existing nearest neighbor analysis
                _, nearest_distances = self.plot_nearest_neighbor_analysis(
                    percentile=nn_percentile, 
                    bin_width=nn_bin_width, 
                    figsize=(1, 1)  # Small figure since we only want the data
                )
                plt.close()  # Close the temporary figure
                
                if nearest_distances is not None:
                    # Create histogram of distances
                    max_distance = np.percentile(nearest_distances, 95)  # Use 95th percentile to avoid outliers
                    bins = np.arange(0, max_distance + nn_bin_width, nn_bin_width)
                    
                    counts, _ = np.histogram(nearest_distances[nearest_distances <= max_distance], bins=bins)
                    bin_centers = (bins[:-1] + bins[1:]) / 2
                    
                    # Create gradient colors (closer = warmer colors)
                    colors = plt.cm.YlOrRd(np.linspace(0.3, 1.0, len(bin_centers)))
                    
                    bars = ax4.bar(bin_centers, counts, width=nn_bin_width*0.8, color=colors, 
                                alpha=0.8, edgecolor='white', linewidth=0.5)
                    
                    # Add count labels on significant bars
                    for bar, count in zip(bars, counts):
                        if count > max(counts) * 0.05:  # Only label bars with >5% of max count
                            height = bar.get_height()
                            ax4.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.02,
                                f'{count}', ha='center', va='bottom', fontsize=8)
                    
                    ax4.set_xlabel('Distance to Nearest Facility (km)')
                    ax4.set_ylabel('Number of High PM2.5 Hotspots')
                    ax4.set_title(f'Nearest Neighbor Analysis\n({len(nearest_distances):,} hotspots above {nn_percentile}th percentile)')
                    ax4.grid(True, alpha=0.3, axis='y')
                    
                    # Add statistics text
                    stats_text = f"Distance Stats (km):\nMean: {nearest_distances.mean():.2f}\nMedian: {np.median(nearest_distances):.2f}"
                    ax4.text(0.95, 0.95, stats_text, transform=ax4.transAxes, va='top', ha='right',
                            fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
                    
                    # Store distances in results for plot 5
                    results['nearest_distances'] = nearest_distances
                else:
                    raise Exception("No distances returned")
                    
            except Exception as e:
                ax4.text(0.5, 0.5, f'Nearest neighbor analysis failed:\n{str(e)[:100]}...', 
                        ha='center', va='center', transform=ax4.transAxes, fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral", alpha=0.8))
                ax4.set_title('Nearest Neighbor Analysis (Failed)')
        else:
            ax4.text(0.5, 0.5, 'Nearest neighbor analysis\nnot included\n\nSet include_nearest_neighbor=True', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=11,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
            ax4.set_title('Nearest Neighbor Analysis (Disabled)')
        
        # Plot 5: Nearest Neighbor Cumulative Distribution
        ax5 = axes[1, 1]
        ax5.text(-0.1, 1.05, 'E)', transform=ax5.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')
        
        if include_nearest_neighbor and 'nearest_distances' in results:
            distances = results['nearest_distances']
            
            # Create cumulative distribution
            sorted_distances = np.sort(distances)
            cumulative_pct = np.arange(1, len(sorted_distances) + 1) / len(sorted_distances) * 100
            
            ax5.plot(sorted_distances, cumulative_pct, linewidth=2, color='darkred')
            ax5.fill_between(sorted_distances, cumulative_pct, alpha=0.3, color='red')
            
            ax5.set_xlabel('Distance to Nearest Facility (km)')
            ax5.set_ylabel('Cumulative % of Hotspots')
            ax5.set_title('Cumulative Distribution\nof Distances')
            ax5.grid(True, alpha=0.3)
            ax5.set_ylim(0, 100)
            
            # Add percentile annotations
            percentiles_to_show = [25, 50, 75, 90]
            for pct in percentiles_to_show:
                distance_at_pct = np.percentile(distances, pct)
                ax5.axhline(y=pct, color='gray', linestyle='--', alpha=0.6)
                ax5.axvline(x=distance_at_pct, color='gray', linestyle='--', alpha=0.6)
                ax5.text(distance_at_pct + max(sorted_distances)*0.01, pct + 2, 
                        f'{pct}%: {distance_at_pct:.1f}km', fontsize=8, color='darkred', fontweight='bold')
            
            # Identify concerning areas (hotspots very close to facilities)
            very_close = distances < 1.0  # Within 1km
            if np.sum(very_close) > 0:
                close_pct = np.sum(very_close)/len(distances)*100
                ax5.text(0.02, 0.98, f"Within 1km:\n{np.sum(very_close):,} hotspots\n({close_pct:.1f}%)", 
                        transform=ax5.transAxes, va='top', ha='left', fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.8))
        else:
            ax5.text(0.5, 0.5, 'Nearest neighbor analysis\nnot included or failed\n\nCheck plot 4 status', 
                    ha='center', va='center', transform=ax5.transAxes, fontsize=11,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
            ax5.set_title('Cumulative Distribution (N/A)')
        
        # Plot 6: Summary statistics table
        ax6 = axes[1, 2]
        ax6.text(-0.1, 1.05, 'F)', transform=ax6.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')
        ax6.axis('off')
        
        # Create summary table
        summary_data = [
            ['Metric', 'Facility Pixels', 'Non-Facility Pixels'],
            ['Count', f"{len(facility_pixels):,}", f"{len(non_facility_pixels):,}"],
            ['Mean ± SD', f"{results['descriptive_stats']['facility']['mean']:.2f} ± {results['descriptive_stats']['facility']['std']:.2f}",
            f"{results['descriptive_stats']['non_facility']['mean']:.2f} ± {results['descriptive_stats']['non_facility']['std']:.2f}"],
            ['Median', f"{results['descriptive_stats']['facility']['median']:.2f}",
            f"{results['descriptive_stats']['non_facility']['median']:.2f}"],
            ['Range', f"{results['descriptive_stats']['facility']['min']:.2f} - {results['descriptive_stats']['facility']['max']:.2f}",
            f"{results['descriptive_stats']['non_facility']['min']:.2f} - {results['descriptive_stats']['non_facility']['max']:.2f}"],
            ['', '', ''],
            ['Distribution Tests', 'Assumption', 'Parameters'],
            [f'Facility Q-Q', f'{facility_dist.title()}', f'{facility_params}' if facility_params else 'Default'],
            [f'Non-facility Q-Q', f'{non_facility_dist.title()}', f'{non_facility_params}' if non_facility_params else 'Default'],
            ['', '', ''],
            ['Statistical Tests', 'Value', 'p-value'],
            ['Point-biserial r', f"{results['correlation']['point_biserial_r']:.4f}", f"{results['correlation']['p_value']:.4f}"],
            ['t-test statistic', f"{results['t_test']['t_statistic']:.4f}", f"{results['t_test']['p_value']:.4f}"],
            ['Cohen\'s d', f"{results['t_test']['cohens_d']:.4f}", ''],
            ['Mann-Whitney U', f"{results['mann_whitney']['u_statistic']:.0f}", f"{results['mann_whitney']['p_value']:.4f}"],
            ['Levene test', f"{results['levene_test']['statistic']:.4f}", f"{results['levene_test']['p_value']:.4f}"]
        ]
        
        # Create table
        table = ax6.table(cellText=summary_data[1:], colLabels=summary_data[0],
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.2)
        
        # Style the table
        for i in range(len(summary_data)):
            for j in range(3):
                cell = table[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor('#40466e')
                    cell.set_text_props(weight='bold', color='white')
                elif i in [6, 10]:  # Sub-headers
                    cell.set_facecolor('#d4d4d4')
                    cell.set_text_props(weight='bold')
                else:
                    cell.set_facecolor('#f0f0f0')
        
        ax6.set_title('Summary Statistics & Distribution Tests')
        
        plt.tight_layout()
        return fig

    # Helper function to add nearest neighbor data to correlation results
    def add_nearest_neighbor_to_correlation(self, correlation_results, nearest_distances):
        """
        Add nearest neighbor distances to existing correlation results
        
        Parameters:
        correlation_results : dict
            Results from analyze_pixel_correlation
        nearest_distances : numpy array
            Distance array from plot_nearest_neighbor_analysis
        
        Returns:
        dict: Combined results
        """
        if correlation_results is None:
            print("No correlation results provided")
            return None
        
        if nearest_distances is None:
            print("No nearest neighbor distances provided")
            return correlation_results
        
        # Add nearest neighbor data
        correlation_results['nearest_distances'] = nearest_distances
        correlation_results['num_hotspots'] = len(nearest_distances)
        
        print(f"✓ Added nearest neighbor data: {len(nearest_distances):,} distances")
        print(f"✓ Distance stats: Mean {nearest_distances.mean():.2f}km, Median {np.median(nearest_distances):.2f}km")
        
        return correlation_results

    def run_buffer_comparison(self, buffer_radii=[0, 1, 2, 3], sample_size=None):
        """
        Compare correlation analysis across different buffer radii
        """
        print(f"\n{'='*80}")
        print(f"BUFFER RADIUS COMPARISON ANALYSIS")
        print(f"{'='*80}")
        
        comparison_results = {}
        
        for radius in buffer_radii:
            print(f"\n🔄 Analyzing buffer radius: {radius}")
            results = self.analyze_pixel_correlation(buffer_radius=radius, sample_size=sample_size)
            if results is not None:
                comparison_results[radius] = results
        
        # Create comparison summary
        if comparison_results:
            print(f"\n📊 BUFFER COMPARISON SUMMARY:")
            print(f"{'Buffer':<8} {'Grid':<8} {'Facility%':<10} {'Correlation':<12} {'p-value':<10} {'t-stat':<10} {'Effect Size':<12}")
            print(f"{'-'*80}")
            
            for radius, results in comparison_results.items():
                grid_size = f"{2*radius+1}x{2*radius+1}" if radius > 0 else "point"
                facility_pct = results['sample_info']['facility_pixels'] / results['sample_info']['total_pixels'] * 100
                corr = results['correlation']['point_biserial_r']
                p_val = results['correlation']['p_value']
                t_stat = results['t_test']['t_statistic']
                effect_size = results['t_test']['cohens_d']
                
                print(f"{radius:<8} {grid_size:<8} {facility_pct:<10.2f} {corr:<12.4f} {p_val:<10.4f} {t_stat:<10.3f} {effect_size:<12.3f}")
        
        return comparison_results  

    def load_eis_categories(self, eis_csv_path):
        """
        Loads EIS facility categories from a CSV, merges them with the facility data,
        and updates the internal configuration to use the new category column.
        
        Parameters:
        eis_csv_path (str): Path to the NEI_Category_by_eis.csv file.
        """
        print("\n--- Loading and Merging EIS Facility Categories ---")
        
        try:
            eis_categories = pd.read_csv(eis_csv_path)
            print(f"✓ Loaded {len(eis_categories)} EIS facility categories from CSV.")
        except FileNotFoundError:
            print(f"❌ ERROR: EIS categories file not found at: {eis_csv_path}")
            return

        # Get the column name for the EIS ID from the user's configuration
        # It's expected to be in the 'optional' section of the config.
        eis_col_name_in_facilities = self.column_config.get('optional', {}).get('eis')
        
        if eis_col_name_in_facilities is None:
            print("❌ ERROR: EIS facility ID column not defined in `column_config['optional']['eis']`.")
            return
        
        if eis_col_name_in_facilities not in self.facilities.columns:
            print(f"❌ ERROR: Column '{eis_col_name_in_facilities}' not found in the facilities GeoDataFrame.")
            print(f"Available columns: {list(self.facilities.columns)}")
            return
            
        # Standardize the merge key in the EIS categories dataframe
        eis_merge_key = 'eis facility id'
        if eis_merge_key not in eis_categories.columns:
            print(f"❌ ERROR: Merge key '{eis_merge_key}' not found in the EIS categories CSV.")
            return

        # Perform the merge
        original_facility_count = len(self.facilities)
        self.facilities = self.facilities.merge(
            eis_categories[['eis facility id', 'Category']], 
            left_on=eis_col_name_in_facilities, 
            right_on=eis_merge_key, 
            how='left'
        )
        
        # --- CRITICAL STEP: Update the configuration to use the new category ---
        # This makes all subsequent analysis automatically use the 'Category' column.
        print("✓ Updating 'facility_type' to use the new 'Category' column for all subsequent analysis.")
        self.column_config['required']['facility_type'] = 'Category'
        
        # Fill missing categories with 'Unknown' for robustness
        self.facilities['Category'].fillna('Unknown', inplace=True)

        # Print merge statistics
        facilities_with_category = self.facilities['Category'].ne('Unknown').sum()
        print(f"✓ Merge complete:")
        print(f"   • Total facilities: {original_facility_count}")
        print(f"   • Facilities matched with an EIS category: {facilities_with_category}")
        print(f"   • Match rate: {facilities_with_category / original_facility_count * 100:.1f}%")
        
        if facilities_with_category > 0:
            print("✓ Top 5 new facility categories:")
            print(self.facilities['Category'].value_counts().head(5).to_string())
        
        return self.facilities

    def create_facility_type_masks(self, buffer_radius=0, category_col='Category'):
        """
        Create separate binary masks for each facility type/category
        
        Parameters:
        buffer_radius : int
            Buffer radius around facilities
        category_col : str
            Column name containing facility categories
        
        Returns:
        dict: Dictionary of masks for each facility type
        """
        print(f"Creating facility type masks (buffer_radius={buffer_radius})...")
        
        if category_col not in self.facilities.columns:
            print(f"❌ ERROR: Column '{category_col}' not found in facility data")
            return None
        
        # Get unique categories (excluding NaN)
        categories = self.facilities[category_col].dropna().unique()
        print(f"✓ Found {len(categories)} facility categories")
        
        height, width = self.pm25_data.shape
        facility_type_masks = {}
        
        # Create mask for each category
        for category in categories:
            category_mask = np.zeros((height, width), dtype=int)
            category_facilities = self.facilities[self.facilities[category_col] == category]
            
            for idx, facility in category_facilities.iterrows():
                # Get facility coordinates
                if facility.geometry.geom_type == 'Point':
                    x, y = facility.geometry.x, facility.geometry.y
                else:
                    centroid = facility.geometry.centroid
                    x, y = centroid.x, centroid.y
                
                # Convert to raster grid position
                row, col = rasterio.transform.rowcol(self.pm25_transform, x, y)
                
                # Mark pixels within buffer
                if buffer_radius == 0:
                    if 0 <= row < height and 0 <= col < width:
                        category_mask[row, col] = 1
                else:
                    for dr in range(-buffer_radius, buffer_radius + 1):
                        for dc in range(-buffer_radius, buffer_radius + 1):
                            r = row + dr
                            c = col + dc
                            if 0 <= r < height and 0 <= c < width:
                                category_mask[r, c] = 1
            
            facility_type_masks[category] = category_mask
            facility_pixels = np.sum(category_mask)
            print(f"   • {category}: {len(category_facilities)} facilities, {facility_pixels} pixels")
        
        return facility_type_masks

    def analyze_facility_types_correlation(self, buffer_radius=0, category_col='Category', 
                                        sample_size=None, min_facilities=5, random_seed=42):
        """
        Analyze correlation between PM2.5 and different facility types
        
        Parameters:
        buffer_radius : int
            Buffer radius around facilities
        category_col : str
            Column name containing facility categories
        sample_size : int, optional
            Number of pixels to sample for analysis
        min_facilities : int
            Minimum number of facilities required for a category to be analyzed
        random_seed : int
            Random seed for reproducible sampling
        
        Returns:
        dict: Results for each facility type
        """
        print(f"\n{'='*80}")
        print(f"FACILITY TYPE CORRELATION ANALYSIS")
        print(f"Buffer radius: {buffer_radius}, Category column: {category_col}")
        print(f"{'='*80}")
        
        np.random.seed(random_seed)
        
        # Create facility type masks
        facility_type_masks = self.create_facility_type_masks(buffer_radius, category_col)
        if facility_type_masks is None:
            return None
        
        # Get valid PM2.5 pixels
        valid_pm_mask = ~np.isnan(self.pm25_data)
        pm25_values_all = self.pm25_data[valid_pm_mask]
        
        # Sample if requested
        if sample_size is not None and sample_size < len(pm25_values_all):
            sample_indices = np.random.choice(len(pm25_values_all), size=sample_size, replace=False)
            pm25_sampled = pm25_values_all[sample_indices]
            # Convert back to coordinates for masking
            valid_coords = np.where(valid_pm_mask)
            sampled_coords = [(valid_coords[0][i], valid_coords[1][i]) for i in sample_indices]
        else:
            pm25_sampled = pm25_values_all
            sampled_coords = list(zip(*np.where(valid_pm_mask)))
        
        print(f"✓ Analyzing {len(pm25_sampled):,} pixels")
        
        # Filter categories by minimum facility count
        category_counts = self.facilities[category_col].value_counts()
        valid_categories = category_counts[category_counts >= min_facilities].index.tolist()
        
        print(f"✓ Categories with ≥{min_facilities} facilities: {len(valid_categories)}")
        
        # Create baseline (no facility) mask
        all_facility_mask = np.zeros_like(self.pm25_data, dtype=int)
        for mask in facility_type_masks.values():
            all_facility_mask = np.logical_or(all_facility_mask, mask)
        
        # Get baseline PM2.5 values (non-facility pixels)
        if sample_size is not None:
            baseline_presence = np.array([all_facility_mask[coord] for coord in sampled_coords])
            baseline_pm25 = pm25_sampled[baseline_presence == 0]
        else:
            baseline_pm25 = pm25_values_all[all_facility_mask[valid_pm_mask] == 0]
        
        baseline_stats = {
            'mean': np.mean(baseline_pm25),
            'std': np.std(baseline_pm25),
            'count': len(baseline_pm25)
        }
        
        print(f"✓ Baseline (no facilities): {baseline_stats['count']:,} pixels, "
            f"mean PM2.5 = {baseline_stats['mean']:.2f} ± {baseline_stats['std']:.2f} μg/m³")
        
        # Analyze each facility type
        type_results = {}
        
        for category in valid_categories:
            if category not in facility_type_masks:
                continue
                
            # Get facility type presence for sampled pixels
            if sample_size is not None:
                type_presence = np.array([facility_type_masks[category][coord] for coord in sampled_coords])
                type_pm25 = pm25_sampled[type_presence == 1]
            else:
                type_pm25 = pm25_values_all[facility_type_masks[category][valid_pm_mask] == 1]
            
            if len(type_pm25) == 0:
                continue
            
            # Calculate statistics
            type_stats = {
                'mean': np.mean(type_pm25),
                'std': np.std(type_pm25),
                'count': len(type_pm25),
                'facility_count': category_counts[category]
            }
            
            # Point-biserial correlation
            if sample_size is not None:
                corr_coef, corr_p = stats.pointbiserialr(type_presence, pm25_sampled)
            else:
                type_presence_full = facility_type_masks[category][valid_pm_mask].astype(int)
                corr_coef, corr_p = stats.pointbiserialr(type_presence_full, pm25_values_all)
            
            # T-test vs baseline
            t_stat, t_p = stats.ttest_ind(type_pm25, baseline_pm25, equal_var=False)
            
            # Effect size vs baseline
            pooled_std = np.sqrt(((len(type_pm25) - 1) * type_stats['std']**2 + 
                                (len(baseline_pm25) - 1) * baseline_stats['std']**2) / 
                                (len(type_pm25) + len(baseline_pm25) - 2))
            cohens_d = (type_stats['mean'] - baseline_stats['mean']) / pooled_std
            
            # Store results
            type_results[category] = {
                'stats': type_stats,
                'correlation': {'r': corr_coef, 'p': corr_p},
                't_test_vs_baseline': {'t': t_stat, 'p': t_p},
                'effect_size': cohens_d,
                'mean_difference': type_stats['mean'] - baseline_stats['mean']
            }
        
        # Sort results by effect size (descending)
        sorted_results = dict(sorted(type_results.items(), 
                                    key=lambda x: abs(x[1]['effect_size']), reverse=True))
        
        # Print summary
        print(f"\n📊 FACILITY TYPE ANALYSIS RESULTS:")
        print(f"{'Category':<25} {'Facilities':<10} {'Pixels':<8} {'Mean PM2.5':<12} {'vs Baseline':<12} {'Effect Size':<12} {'p-value':<10}")
        print(f"{'-'*100}")
        
        for category, results in sorted_results.items():
            stats_data = results['stats']
            diff = results['mean_difference']
            effect = results['effect_size']
            p_val = results['t_test_vs_baseline']['p']
            
            print(f"{category[:24]:<25} {stats_data['facility_count']:<10} {stats_data['count']:<8} "
                f"{stats_data['mean']:<12.2f} {diff:+12.2f} {effect:<12.3f} {p_val:<10.4f}")
        
        # Add baseline to results
        results_with_baseline = {
            'baseline': {
                'stats': baseline_stats,
                'correlation': {'r': 0, 'p': 1.0},
                't_test_vs_baseline': {'t': 0, 'p': 1.0},
                'effect_size': 0,
                'mean_difference': 0
            }
        }
        results_with_baseline.update(sorted_results)
        
        return results_with_baseline

    def plot_facility_types_analysis(self, type_results, figsize=(16, 12), top_n=10):
        """
        Create visualization comparing different facility types
        """
        if type_results is None or len(type_results) <= 1:
            print("No results to plot")
            return None
        
        # Remove baseline from plotting data but keep for reference
        baseline_mean = type_results['baseline']['stats']['mean']
        plot_results = {k: v for k, v in type_results.items() if k != 'baseline'}
        
        # Limit to top N categories by absolute effect size
        sorted_categories = list(plot_results.keys())[:top_n]
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'Facility Type Analysis: PM2.5 Correlation\n(Top {len(sorted_categories)} categories by effect size)', 
                    fontsize=16, fontweight='bold')
        
        # Plot 1: Mean PM2.5 by facility type
        ax1 = axes[0, 0]
        categories = sorted_categories
        means = [plot_results[cat]['stats']['mean'] for cat in categories]
        stds = [plot_results[cat]['stats']['std'] for cat in categories]
        colors = ['red' if plot_results[cat]['mean_difference'] > 0 else 'blue' for cat in categories]
        
        bars = ax1.bar(range(len(categories)), means, yerr=stds, capsize=3, color=colors, alpha=0.7)
        ax1.axhline(y=baseline_mean, color='black', linestyle='--', linewidth=2, label=f'Baseline: {baseline_mean:.2f}')
        
        ax1.set_xlabel('Facility Type', fontweight='bold')
        ax1.set_ylabel('Mean PM2.5 (μg/m³)', fontweight='bold')
        ax1.set_title('Mean PM2.5 by Facility Type', fontweight='bold')
        ax1.set_xticks(range(len(categories)))
        ax1.set_xticklabels([cat[:15] + '...' if len(cat) > 15 else cat for cat in categories], 
                        rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Effect size comparison
        ax2 = axes[0, 1]
        effect_sizes = [plot_results[cat]['effect_size'] for cat in categories]
        colors_effect = ['red' if es > 0 else 'blue' for es in effect_sizes]
        
        bars2 = ax2.bar(range(len(categories)), effect_sizes, color=colors_effect, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax2.axhline(y=0.2, color='gray', linestyle=':', alpha=0.7, label='Small effect')
        ax2.axhline(y=0.5, color='gray', linestyle=':', alpha=0.7, label='Medium effect')
        ax2.axhline(y=0.8, color='gray', linestyle=':', alpha=0.7, label='Large effect')
        
        ax2.set_xlabel('Facility Type', fontweight='bold')
        ax2.set_ylabel("Cohen's d (vs Baseline)", fontweight='bold')
        ax2.set_title('Effect Size vs Baseline', fontweight='bold')
        ax2.set_xticks(range(len(categories)))
        ax2.set_xticklabels([cat[:15] + '...' if len(cat) > 15 else cat for cat in categories], 
                        rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Correlation coefficients
        ax3 = axes[1, 0]
        correlations = [plot_results[cat]['correlation']['r'] for cat in categories]
        p_values = [plot_results[cat]['correlation']['p'] for cat in categories]
        colors_corr = ['red' if p < 0.05 else 'lightgray' for p in p_values]
        
        bars3 = ax3.bar(range(len(categories)), correlations, color=colors_corr, alpha=0.7)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        ax3.set_xlabel('Facility Type', fontweight='bold')
        ax3.set_ylabel('Point-Biserial Correlation', fontweight='bold')
        ax3.set_title('Correlation with PM2.5 (Red = p<0.05)', fontweight='bold')
        ax3.set_xticks(range(len(categories)))
        ax3.set_xticklabels([cat[:15] + '...' if len(cat) > 15 else cat for cat in categories], 
                        rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Summary table
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Create summary table data
        table_data = [['Facility Type', 'Facilities', 'Effect Size', 'p-value']]
        for cat in categories[:8]:  # Show top 8 in table
            facilities = plot_results[cat]['stats']['facility_count']
            effect = plot_results[cat]['effect_size']
            p_val = plot_results[cat]['t_test_vs_baseline']['p']
            
            table_data.append([
                cat[:20] + ('...' if len(cat) > 20 else ''),
                str(facilities),
                f"{effect:.3f}",
                f"{p_val:.4f}"
            ])
        
        # Create table
        table = ax4.table(cellText=table_data[1:], colLabels=table_data[0],
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.8)
        
        # Style the table
        for i in range(len(table_data)):
            for j in range(4):
                cell = table[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor('#40466e')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#f0f0f0')
        
        ax4.set_title('Top Categories Summary', fontweight='bold', pad=20)
        
        plt.tight_layout()
        return fig
    
    def plot_overall_pm25_distribution(self, percentile=95, bin_width=0.1, figsize=(10, 6)):
        """
        Plot overall PM2.5 distribution from the entire raster
        """
        threshold = np.percentile(self.all_pm25_values, percentile)
        
        # Create fine bins
        pm25_min = self.all_pm25_values.min()
        pm25_max = self.all_pm25_values.max()
        n_bins = int(np.ceil((pm25_max - pm25_min) / bin_width))
        
        # Create histogram
        counts, bins = np.histogram(self.all_pm25_values, bins=n_bins)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # Color bars based on threshold
        color_below = '#87CEEB'  # Light blue
        color_above = '#FF6B35'  # Orange
        colors = [color_above if center >= threshold else color_below for center in bin_centers]
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        bars = ax.bar(bin_centers, counts, width=bin_width*0.8, color=colors, 
                     alpha=0.7, edgecolor='white', linewidth=0.5)
        
        # Add threshold line
        ax.axvline(x=threshold, color='red', linestyle='--', linewidth=2)
        ax.text(threshold, max(counts)*0.9, f'{percentile}th percentile\n{threshold:.2f} μg/m³',
               ha='center', va='top', fontsize=11, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
        
        # Count pixels above/below threshold
        pixels_above = np.sum(self.all_pm25_values >= threshold)
        pixels_below = np.sum(self.all_pm25_values < threshold)
        
        # Add summary text
        summary_text = f"Above threshold: {pixels_above:,} pixels\n"
        summary_text += f"Below threshold: {pixels_below:,} pixels"
        ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, va='top', ha='left',
               fontsize=10, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        ax.set_xlabel('PM2.5 Concentration (μg/m³)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Pixels', fontsize=12, fontweight='bold')
        ax.set_title(f'Overall PM2.5 Distribution\n({len(self.all_pm25_values):,} valid pixels)', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        print(f"✓ Overall PM2.5 distribution plot created")
        print(f"✓ Threshold ({percentile}th percentile): {threshold:.2f} μg/m³")
        print(f"✓ Pixels above threshold: {pixels_above:,} ({pixels_above/len(self.all_pm25_values)*100:.1f}%)")
        
        return fig, threshold
    
    def plot_facility_pm25_distribution(self, buffer_radius=0, percentile=95, bin_width=0.1, figsize=(10, 6)):
        """
        Plot facility PM2.5 distribution with configurable buffer
        
        Parameters:
        buffer_radius : int
            0 = point location, 1 = 3x3 buffer, 2 = 5x5 buffer, etc.
        percentile : float
            Percentile threshold calculated from ALL PM2.5 pixels (not just facilities)
        bin_width : float
            Width of histogram bins in μg/m³
        """
        print(f"Creating facility PM2.5 distribution plot...")
        
        # Calculate threshold from ALL PM2.5 pixels (not facilities)
        overall_threshold = np.percentile(self.all_pm25_values, percentile)
        print(f"✓ Using overall PM2.5 threshold ({percentile}th percentile): {overall_threshold:.2f} μg/m³")
        
        # Calculate facility PM2.5 with specified buffer
        facility_data = self.get_facility_pm25(buffer_radius=buffer_radius)
        valid_facilities = facility_data.dropna(subset=['pm25'])
        
        if len(valid_facilities) == 0:
            print("No valid facility data to plot")
            return None, None
        
        # Classify facilities based on overall threshold
        facilities_above_threshold = valid_facilities[valid_facilities['pm25'] >= overall_threshold]
        facilities_below_threshold = valid_facilities[valid_facilities['pm25'] < overall_threshold]
        
        print(f"✓ Facilities above overall threshold: {len(facilities_above_threshold)}")
        print(f"✓ Facilities below overall threshold: {len(facilities_below_threshold)}")
        
        # Create fine bins for facilities
        facility_min = valid_facilities['pm25'].min()
        facility_max = valid_facilities['pm25'].max()
        
        # Ensure bins align nicely
        bin_start = np.floor(facility_min / bin_width) * bin_width
        bin_end = np.ceil(facility_max / bin_width) * bin_width
        bin_edges = np.arange(bin_start, bin_end + bin_width, bin_width)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Create histogram
        counts, _ = np.histogram(valid_facilities['pm25'], bins=bin_edges)
        
        # Color bars based on overall threshold
        color_below = '#87CEEB'  # Light blue
        color_above = '#FF6B35'  # Orange
        colors = [color_above if center >= overall_threshold else color_below for center in bin_centers]
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        bars = ax.bar(bin_centers, counts, width=bin_width*0.8, color=colors,
                     alpha=0.8, edgecolor='white', linewidth=0.5)
        
        # Add count labels on bars (only if count > 0)
        for bar, count in zip(bars, counts):
            if count > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.02,
                       f'{count}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Add threshold line
        ax.axvline(x=overall_threshold, color='red', linestyle='--', linewidth=2)
        ax.text(overall_threshold, max(counts)*0.9, f'Overall {percentile}th percentile\n{overall_threshold:.2f} μg/m³',
               ha='center', va='top', fontsize=11, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
        
        # Add summary text
        summary_text = f"Above overall threshold: {len(facilities_above_threshold)}\n"
        summary_text += f"Below overall threshold: {len(facilities_below_threshold)}"
        ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, va='top', ha='left',
               fontsize=10, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        # Create title based on buffer radius
        if buffer_radius == 0:
            facility_title = "Facility Point Location"
        else:
            grid_size = 2 * buffer_radius + 1
            facility_title = f"Facility {grid_size}×{grid_size} Buffer Average"
        
        ax.set_xlabel('PM2.5 Concentration (μg/m³)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Facilities', fontsize=12, fontweight='bold')
        ax.set_title(f'{facility_title} PM2.5 Distribution\n({len(valid_facilities)} facilities)', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        print(f"✓ Facility PM2.5 distribution plot created")
        print(f"✓ Buffer configuration: {buffer_radius} ({'point' if buffer_radius == 0 else f'{2*buffer_radius+1}×{2*buffer_radius+1}'})")
        
        return fig, overall_threshold
    
    def plot_facility_type_pie_chart(self, type_col="facility_type", buffer_radius=0, percentile=95, 
                                    max_types=8, figsize=(10, 6)):
        """
        Plot pie chart of facility types for facilities above overall PM2.5 threshold
        
        Parameters:
        type_col : str
            Column name for facility type
        buffer_radius : int
            0 = point location, 1 = 3x3 buffer, 2 = 5x5 buffer, etc.
        percentile : float
            Percentile threshold calculated from ALL PM2.5 pixels
        max_types : int
            Maximum number of types to show individually (default: 8)
            Types beyond this will be grouped as "Others"
        """
        print(f"Creating facility type pie chart...")
        
        # Calculate threshold from ALL PM2.5 pixels (not facilities)
        overall_threshold = np.percentile(self.all_pm25_values, percentile)
        print(f"✓ Using overall PM2.5 threshold ({percentile}th percentile): {overall_threshold:.2f} μg/m³")
        
        # Calculate facility PM2.5 with specified buffer
        facility_data = self.get_facility_pm25(buffer_radius=buffer_radius)
        
        valid_facilities = facility_data.dropna(subset=['pm25'])
        print(f"✓ Found {len(valid_facilities)} facilities with valid PM2.5 data.")
        valid_facilities = valid_facilities[valid_facilities['facility_type'] != 'Unknown']
        print(f"✓ After removing 'Unknown' types, {len(valid_facilities)} facilities remain for plotting.")

       
        if len(valid_facilities) == 0:
            print("No valid facility data for pie chart")
            return None
        
        # Get facilities above overall threshold
        facilities_above_threshold = valid_facilities[valid_facilities['pm25'] >= overall_threshold]
        
        if len(facilities_above_threshold) == 0:
            print(f"No facilities above overall {percentile}th percentile threshold")
            return None
        
        # Get facility type counts and handle "Others" grouping
        all_facility_counts = facilities_above_threshold[type_col].value_counts()
        
        if len(all_facility_counts) <= max_types:
            # No need to group - show all types
            facility_counts = all_facility_counts
            facilities_for_stats = facilities_above_threshold
        else:
            # Group smaller types into "Others"
            top_types = all_facility_counts.head(max_types - 1)  # Leave room for "Others"
            others_types = all_facility_counts.iloc[max_types - 1:]
            others_count = others_types.sum()
            
            # Create new counts with "Others"
            facility_counts = top_types.copy()
            facility_counts['Others'] = others_count
            
            # Create modified dataframe for statistics
            facilities_for_stats = facilities_above_threshold.copy()
            others_mask = facilities_for_stats[type_col].isin(others_types.index)
            facilities_for_stats.loc[others_mask, type_col] = 'Others'
        
        # Get facility type information for statistics
        facility_stats = facilities_for_stats.groupby(type_col).agg({
            'pm25': ['count', 'mean', 'std'],
        }).round(2)
        
        facility_stats.columns = ['Count', 'Avg_PM25', 'Std_PM25']
        facility_stats = facility_stats.reset_index()
        
        # Create figure with pie chart and details
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [2, 1]})
        
        # Create pie chart
        colors_pie = plt.cm.Oranges(np.linspace(0.4, 0.9, len(facility_counts)))
        
        wedges, texts, autotexts = ax1.pie(facility_counts.values, labels=facility_counts.index,
                                        autopct='%1.0f%%', colors=colors_pie, startangle=90)
        
        # Format text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
        
        for text in texts:
            text.set_fontsize(10)
            text.set_fontweight('bold')
        
        # Create title based on buffer radius
        if buffer_radius == 0:
            location_desc = "Point Location"
        else:
            grid_size = 2 * buffer_radius + 1
            location_desc = f"{grid_size}×{grid_size} Buffer Avg"
        
        # Update title to show grouping info if applicable
        title = f'Facility Types Above Overall {percentile}th Percentile\n({location_desc}, {len(facilities_above_threshold)} facilities)'
        if len(all_facility_counts) > max_types:
            others_count = facility_counts['Others']
            others_type_count = len(all_facility_counts) - (max_types - 1)
            title += f'\n(Top {max_types-1} types + Others: {others_count} facilities from {others_type_count} types)'
        
        ax1.set_title(title, fontsize=12, fontweight='bold')
        
        # Add detailed statistics table
        ax2.axis('off')
        
        # Create detailed text
        detail_text = f"OVERALL THRESHOLD: {overall_threshold:.2f} μg/m³\n"
        detail_text += f"(Based on all {len(self.all_pm25_values):,} PM2.5 pixels)\n\n"
        detail_text += "FACILITY TYPE DETAILS:\n"
        detail_text += "=" * 25 + "\n\n"
        
        for _, row in facility_stats.iterrows():
            facility_type = row[type_col]
            count = int(row['Count'])
            avg_pm25 = row['Avg_PM25']
            std_pm25 = row['Std_PM25'] if not np.isnan(row['Std_PM25']) else 0
            
            detail_text += f"{facility_type}:\n"
            detail_text += f"  • Count: {count} facilities\n"
            detail_text += f"  • Avg PM2.5: {avg_pm25:.2f} ± {std_pm25:.2f} μg/m³\n\n"

        # Add overall statistics
        avg_pm25_all = facilities_above_threshold['pm25'].mean()
        
        detail_text += f"OVERALL STATS:\n"
        detail_text += f"  • Total facilities: {len(facilities_above_threshold)}\n"
        detail_text += f"  • Avg PM2.5: {avg_pm25_all:.2f} μg/m³\n"
        
        # # Add "Others" breakdown if applicable
        # if len(all_facility_counts) > max_types:
        #     detail_text += f"\nOTHERS BREAKDOWN:\n"
        #     others_list = all_facility_counts.iloc[max_types - 1:]
        #     for otype, ocount in others_list.items():
        #         detail_text += f"  • {otype}: {ocount}\n"
        
        ax2.text(0.05, 0.95, detail_text, transform=ax2.transAxes, va='top', ha='left',
                fontsize=9, fontweight='normal', family='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
        
        plt.tight_layout()
        
        print(f"✓ Facility type pie chart created")
        print(f"✓ Buffer configuration: {buffer_radius} ({'point' if buffer_radius == 0 else f'{2*buffer_radius+1}×{2*buffer_radius+1}'})")
        if len(all_facility_counts) > max_types:
            print(f"✓ Showing top {max_types-1} types + Others ({facility_counts['Others']} facilities)")
        print(f"✓ Facilities above overall threshold breakdown:")
        for facility_type, count in facility_counts.items():
            if facility_type == 'Others':
                print(f"   - {facility_type}: {count} facilities (grouped from {len(all_facility_counts) - (max_types - 1)} types)")
            else:
                avg_pm25 = facilities_above_threshold[facilities_above_threshold[type_col] == facility_type]['pm25'].mean()
                print(f"   - {facility_type}: {count} facilities (avg PM2.5: {avg_pm25:.2f} μg/m³)")
        
        return fig
 
    def create_combined_analysis_plot(self, facility_data, type_col ="facility_type",percentile=95, bin_width=0.1, figsize=(18, 12)):
        """
        Create a combined plot with all three analyses
        """
        valid_facilities = facility_data.dropna(subset=['pm25'])
        
        if len(valid_facilities) == 0:
            print("No valid facility data for combined plot")
            return None
        
        # Calculate thresholds
        overall_threshold = np.percentile(self.all_pm25_values, percentile)
        facility_threshold = np.percentile(valid_facilities['pm25'], percentile)
        
        # Create figure with 2x2 layout
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1])
        
        # Plot 1: Overall PM2.5 distribution
        ax1 = fig.add_subplot(gs[0, 0])
        
        # Create histogram for overall distribution
        pm25_min = self.all_pm25_values.min()
        pm25_max = self.all_pm25_values.max()
        n_bins = int(np.ceil((pm25_max - pm25_min) / bin_width))
        counts, bins = np.histogram(self.all_pm25_values, bins=n_bins)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        color_below = '#87CEEB'
        color_above = '#FF6B35'
        colors = [color_above if center >= overall_threshold else color_below for center in bin_centers]
        
        ax1.bar(bin_centers, counts, width=bin_width*0.8, color=colors, alpha=0.7, edgecolor='white', linewidth=0.5)
        ax1.axvline(x=overall_threshold, color='red', linestyle='--', linewidth=2)
        ax1.text(overall_threshold, max(counts)*0.9, f'{percentile}th percentile\n{overall_threshold:.2f}',
                ha='center', va='top', fontsize=9, bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.8))
        
        ax1.set_xlabel('PM2.5 (μg/m³)', fontsize=10, fontweight='bold')
        ax1.set_ylabel('Pixels', fontsize=10, fontweight='bold')
        ax1.set_title(f'Overall PM2.5 Distribution\n({len(self.all_pm25_values):,} pixels)', fontsize=11, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Facility PM2.5 distribution
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Create histogram for facilities
        facility_min = valid_facilities['pm25'].min()
        facility_max = valid_facilities['pm25'].max()
        bin_start = np.floor(facility_min / bin_width) * bin_width
        bin_end = np.ceil(facility_max / bin_width) * bin_width
        bin_edges = np.arange(bin_start, bin_end + bin_width, bin_width)
        bin_centers_fac = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        counts_fac, _ = np.histogram(valid_facilities['pm25'], bins=bin_edges)
        colors_fac = [color_above if center >= facility_threshold else color_below for center in bin_centers_fac]
        
        bars = ax2.bar(bin_centers_fac, counts_fac, width=bin_width*0.8, color=colors_fac, alpha=0.8, edgecolor='white', linewidth=0.5)
        
        # Add count labels
        for bar, count in zip(bars, counts_fac):
            if count > 0:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + max(counts_fac)*0.02,
                       f'{count}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        ax2.axvline(x=facility_threshold, color='red', linestyle='--', linewidth=2)
        ax2.text(facility_threshold, max(counts_fac)*0.9, f'{percentile}th percentile\n{facility_threshold:.2f}',
                ha='center', va='top', fontsize=9, bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.8))
        
        buffer_radius = facility_data['buffer_radius'].iloc[0] if 'buffer_radius' in facility_data.columns else 0
        if buffer_radius == 0:
            facility_title = "Point Location"
        else:
            grid_size = 2 * buffer_radius + 1
            facility_title = f"{grid_size}×{grid_size} Buffer"
        
        ax2.set_xlabel('PM2.5 (μg/m³)', fontsize=10, fontweight='bold')
        ax2.set_ylabel('Facilities', fontsize=10, fontweight='bold')
        ax2.set_title(f'Facility {facility_title} PM2.5\n({len(valid_facilities)} facilities)', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: facility type pie chart
        ax3 = fig.add_subplot(gs[0, 2])
        
        facilities_above_threshold = valid_facilities[valid_facilities['pm25'] >= facility_threshold]
        
        if len(facilities_above_threshold) > 0:
            facility_counts = facilities_above_threshold[type_col].value_counts()
            colors_pie = plt.cm.Oranges(np.linspace(0.4, 0.9, len(facility_counts)))
            
            wedges, texts, autotexts = ax3.pie(facility_counts.values, labels=facility_counts.index,
                                              autopct='%1.0f%%', colors=colors_pie, startangle=90)
            
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(9)
            
            for text in texts:
                text.set_fontsize(8)
        else:
            ax3.text(0.5, 0.5, 'No facilities\nabove threshold', ha='center', va='center',
                    transform=ax3.transAxes, fontsize=10)
        
        ax3.set_title(f'Facility Types Above {percentile}th Percentile\n({len(facilities_above_threshold)} facilities)', 
                     fontsize=11, fontweight='bold')
        
        # Plot 4: Summary statistics (spans bottom row)
        ax4 = fig.add_subplot(gs[1, :])
        ax4.axis('off')
        
        # Create summary text
        pixels_above_overall = np.sum(self.all_pm25_values >= overall_threshold)
        facilities_above = len(facilities_above_threshold)
        
        summary_text = f"ANALYSIS SUMMARY - {percentile}th Percentile Thresholds\n"
        summary_text += "=" * 80 + "\n\n"
        
        summary_text += f"OVERALL PM2.5 ENVIRONMENT:\n"
        summary_text += f"  • Total valid pixels: {len(self.all_pm25_values):,}\n"
        summary_text += f"  • Threshold: {overall_threshold:.2f} μg/m³\n"
        summary_text += f"  • Pixels above threshold: {pixels_above_overall:,} ({pixels_above_overall/len(self.all_pm25_values)*100:.1f}%)\n\n"
        
        summary_text += f"FACILITY ANALYSIS ({facility_title}):\n"
        summary_text += f"  • Total facilities: {len(valid_facilities)}\n"
        summary_text += f"  • Threshold: {facility_threshold:.2f} μg/m³\n"
        summary_text += f"  • Facilities above threshold: {facilities_above} ({facilities_above/len(valid_facilities)*100:.1f}%)\n\n"
        
        if len(facilities_above_threshold) > 0:
            summary_text += f"facility TYPES ABOVE THRESHOLD:\n"
            for facility_type, count in facilities_above_threshold[type_col].value_counts().items():
                max_pm25 = facilities_above_threshold[facilities_above_threshold[type_col] == facility_type]['pm25'].mean()
                pct = count / len(facilities_above_threshold) * 100
                summary_text += f"  • {facility_type}: {count} facilities ({pct:.1f}%) - Avg PM2.5: {max_pm25:.2f} μg/m³\n"
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, va='top', ha='left',
                fontsize=10, fontweight='normal', family='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.9))
        
        plt.suptitle(f'PM2.5 Facility Analysis - {percentile}th Percentile Analysis', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        
        print(f"✓ Combined analysis plot created")
        
        return fig
    
    def plot_nearest_neighbor_analysis(self, percentile=95, bin_width=0.5, figsize=(12, 8)):
        """
        Nearest neighbor analysis: For each high PM2.5 hotspot, plot distance to nearest facility
        
        Parameters:
        percentile : float
            Percentile to define high PM2.5 hotspots
        bin_width : float
            Width of distance bins in km
        """
        print(f"Creating nearest neighbor analysis...")
        
        # Get high PM2.5 hotspots (using overall threshold)
        overall_threshold = np.percentile(self.all_pm25_values, percentile)
        print(f"✓ High PM2.5 threshold ({percentile}th percentile): {overall_threshold:.2f} μg/m³")
        
        # Find high PM2.5 hotspot locations
        height, width = self.pm25_data.shape
        hotspot_coords = []
        hotspot_values = []
        
        for row in range(height):
            for col in range(width):
                pm25_val = self.pm25_data[row, col]
                if not np.isnan(pm25_val) and pm25_val >= overall_threshold:
                    x, y = rasterio.transform.xy(self.pm25_transform, row, col)
                    hotspot_coords.append((x, y))
                    hotspot_values.append(pm25_val)
        
        hotspot_coords = np.array(hotspot_coords)
        hotspot_values = np.array(hotspot_values)
        
        print(f"✓ Found {len(hotspot_coords):,} high PM2.5 hotspot pixels")
        
        if len(hotspot_coords) == 0:
            print("No high PM2.5 hotspots found!")
            return None
        
        # Get facility coordinates
        facility_coords = []
        for idx, facility in self.facilities.iterrows():
            if facility.geometry.geom_type == 'Point':
                x, y = facility.geometry.x, facility.geometry.y
            else:
                centroid = facility.geometry.centroid
                x, y = centroid.x, centroid.y
            facility_coords.append((x, y))
        
        facility_coords = np.array(facility_coords)
        print(f"✓ Analyzing {len(facility_coords)} facilities")
        
        # Calculate distance from each hotspot to nearest facility
        print("Calculating nearest neighbor distances...")
        nearest_distances = []
        
        for i, hotspot_coord in enumerate(hotspot_coords):
            # Calculate haversine distances in km
            distances = self.haversine_distance(
                facility_coords[:, 0], facility_coords[:, 1],  # facility lon, lat
                hotspot_coord[0], hotspot_coord[1]             # hotspot lon, lat
            )
            nearest_distance = np.min(distances)  # Distance to nearest facility
            nearest_distances.append(nearest_distance)
            
            if (i + 1) % 10000 == 0 or i == len(hotspot_coords) - 1:
                print(f"  Processed {i+1:,} / {len(hotspot_coords):,} hotspots")
        
        nearest_distances = np.array(nearest_distances)
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Histogram of distances
        # Create bins
        max_distance = np.percentile(nearest_distances, 95)  # Use 95th percentile to avoid outliers
        n_bins = int(np.ceil(max_distance / bin_width))
        bins = np.arange(0, max_distance + bin_width, bin_width)
        
        counts, _ = np.histogram(nearest_distances[nearest_distances <= max_distance], bins=bins)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # Create gradient colors (closer = more concerning = warmer colors)
        colors = plt.cm.YlOrRd(np.linspace(0.3, 1.0, len(bin_centers)))
        
        bars = ax1.bar(bin_centers, counts, width=bin_width*0.8, color=colors, 
                      alpha=0.8, edgecolor='white', linewidth=0.5)
        
        # Add count labels on significant bars
        for bar, count in zip(bars, counts):
            if count > max(counts) * 0.05:  # Only label bars with >5% of max count
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                       f'{count:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax1.set_xlabel('Distance to Nearest Facility (km)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Number of High PM2.5 Hotspots', fontsize=12, fontweight='bold')
        ax1.set_title(f'Nearest Neighbor Analysis\n({len(hotspot_coords):,} hotspots above {percentile}th percentile)', 
                     fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add statistics text
        stats_text = f"Distance Statistics (km):\n"
        stats_text += f"Min: {nearest_distances.min():.2f}\n"
        stats_text += f"Mean: {nearest_distances.mean():.2f}\n"
        stats_text += f"Median: {np.median(nearest_distances):.2f}\n"
        stats_text += f"Max: {nearest_distances.max():.2f}"
        
        ax1.text(0.98, 0.98, stats_text, transform=ax1.transAxes, va='top', ha='right',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        
        # Plot 2: Cumulative distribution
        sorted_distances = np.sort(nearest_distances)
        cumulative_pct = np.arange(1, len(sorted_distances) + 1) / len(sorted_distances) * 100
        
        ax2.plot(sorted_distances, cumulative_pct, linewidth=2, color='darkred')
        ax2.fill_between(sorted_distances, cumulative_pct, alpha=0.3, color='red')
        
        ax2.set_xlabel('Distance to Nearest Facility (km)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Cumulative % of Hotspots', fontsize=12, fontweight='bold')
        ax2.set_title('Cumulative Distribution of Distances', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)
        
        # Add percentile annotations
        percentiles_to_show = [25, 50, 75, 90, 95]
        for pct in percentiles_to_show:
            distance_at_pct = np.percentile(nearest_distances, pct)
            ax2.axhline(y=pct, color='gray', linestyle='--', alpha=0.6)
            ax2.axvline(x=distance_at_pct, color='gray', linestyle='--', alpha=0.6)
            ax2.text(distance_at_pct + max(sorted_distances)*0.01, pct + 1, 
                    f'{pct}%: {distance_at_pct:.1f}km', fontsize=9, color='darkred', fontweight='bold')
        
        plt.tight_layout()
        
        # Print summary
        print(f"\n✓ Nearest neighbor analysis complete!")
        print(f"✓ High PM2.5 hotspots analyzed: {len(hotspot_coords):,}")
        print(f"✓ Distance statistics (km):")
        print(f"   - Mean distance to nearest facility: {nearest_distances.mean():.2f}")
        print(f"   - Median distance: {np.median(nearest_distances):.2f}")
        print(f"   - 25th percentile: {np.percentile(nearest_distances, 25):.2f}")
        print(f"   - 75th percentile: {np.percentile(nearest_distances, 75):.2f}")
        print(f"   - 95th percentile: {np.percentile(nearest_distances, 95):.2f}")
        
        # Identify concerning areas (hotspots very close to facilities)
        very_close = nearest_distances < 1.0  # Within 1km
        if np.sum(very_close) > 0:
            print(f"✓ Hotspots within 1km of facilities: {np.sum(very_close):,} ({np.sum(very_close)/len(nearest_distances)*100:.1f}%)")
        
        return fig, nearest_distances
    
    def save_facility_data(self, facility_data, buffer_radius=0, filename='facility_pm25_data'):
        """Save facility data to CSV"""
        buffer_desc = "point" if buffer_radius == 0 else f"buffer{buffer_radius}"
        csv_file = f'{filename}_{buffer_desc}.csv'
        facility_data.to_csv(csv_file, index=False)
        print(f"✓ Facility data saved to {csv_file}")
        return csv_file

