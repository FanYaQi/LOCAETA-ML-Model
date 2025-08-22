"""
Enhanced configurable ML trainer that extends base functionality
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import r2_score, mean_squared_error
from pathlib import Path
from scipy.interpolate import griddata

from core.base_trainer import BaseMLTrainer
from utils.path_util import DATA_PATH, MODELS_PATH


class ConfigurableMLTrainer(BaseMLTrainer):
    """
    Enhanced configurable ML trainer for air quality prediction with heatmap capabilities
    """
    
    def __init__(self, config):
        """Initialize with enhanced configuration options"""
        super().__init__(config)
    
    def split_data(self, X, y):
        """Split data based on validation strategy with enhanced control"""
        validation_type = self.config['validation_type']
        test_size = self.config['test_size']
        random_state = self.config['random_state']
        
        if validation_type == 'sample_based':
            # Random sample-based split
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            print(f"Sample-based split: {len(self.X_train)} train, {len(self.X_test)} test")
            
        elif validation_type == 'site_based':
            # Site-based split with user control
            if 'facility_id' not in X.columns:
                raise ValueError("facility_id column required for site-based validation")
            
            facilities = sorted(X['facility_id'].unique())
            print(f"Available facilities: {facilities}")
            
            if self.config.get('test_facilities'):
                # User-specified test facilities
                test_facilities = self.config['test_facilities']
                
                # Validate that specified facilities exist
                missing_facilities = [f for f in test_facilities if f not in facilities]
                if missing_facilities:
                    raise ValueError(f"Specified test facilities not found: {missing_facilities}")
                
                test_mask = X['facility_id'].isin(test_facilities)
                train_mask = ~test_mask
                
                self.X_train = X[train_mask].reset_index(drop=True)
                self.X_test = X[test_mask].reset_index(drop=True)
                self.y_train = y[train_mask].reset_index(drop=True)
                self.y_test = y[test_mask].reset_index(drop=True)
                
                print(f"User-specified test facilities: {test_facilities}")
                
            else:
                # Auto-selection using GroupShuffleSplit
                gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
                train_idx, test_idx = next(gss.split(X, y, groups=X['facility_id']))
                
                self.X_train = X.iloc[train_idx].reset_index(drop=True)
                self.X_test = X.iloc[test_idx].reset_index(drop=True)
                self.y_train = y.iloc[train_idx].reset_index(drop=True)
                self.y_test = y.iloc[test_idx].reset_index(drop=True)
                
                test_facilities = sorted(self.X_test['facility_id'].unique())
                print(f"Auto-selected test facilities: {test_facilities}")
            
            train_facilities = sorted(self.X_train['facility_id'].unique())
            test_facilities = sorted(self.X_test['facility_id'].unique())
            
            print(f"Site-based split:")
            print(f"  Train facilities: {train_facilities}")
            print(f"  Test facilities: {test_facilities}")
            print(f"  Train samples: {len(self.X_train)}, Test samples: {len(self.X_test)}")
            
            # Set default heatmap facilities if not specified
            if (self.config.get('generate_heatmaps', False) and 
                self.config.get('heatmap_facilities') == 'auto'):
                self.config['heatmap_facilities'] = test_facilities
                print(f"Heatmap facilities set to: {test_facilities}")
        
        else:
            raise ValueError(f"Unknown validation type: {validation_type}")
    
    def create_plots(self):
        """Create separate plots and save individually"""
        output_dir = Path(self.config['output_dir'])
        base_name = self._get_base_name()
        
        plot_paths = []
        
        # Plot 1: Scatter plot
        fig1, ax1 = plt.subplots(1, 1, figsize=(8, 6))
        self._plot_scatter_with_intensity(ax1)
        scatter_path = output_dir / f"{base_name}_scatter.png"
        plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths.append(scatter_path)
        
        # Plot 2: Feature importance  
        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))
        self._plot_feature_importance(ax2)
        importance_path = output_dir / f"{base_name}_importance.png"
        plt.savefig(importance_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths.append(importance_path)
        
        # Plot 3: Learning curve
        fig3, ax3 = plt.subplots(1, 1, figsize=(8, 6))
        self._plot_learning_curve(ax3)
        learning_path = output_dir / f"{base_name}_learning.png"
        plt.savefig(learning_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths.append(learning_path)
        
        # Plot 4: Heatmaps (if enabled and site-based validation)
        if (self.config.get('generate_heatmaps', False) and 
            self.config['validation_type'] == 'site_based'):
            heatmap_paths = self._create_heatmaps()
            plot_paths.extend(heatmap_paths)
        
        print(f"Plots saved:")
        for path in plot_paths:
            print(f"  {path.name}")
        
        return plot_paths
    
    def _get_base_name(self):
        """Generate base name for files with optional suffix"""
        base_name = f"{self.config['model_type']}_{self.config['method']}_{self.config['validation_type']}"
        
        if self.config.get('figure_suffix'):
            base_name += f"_{self.config['figure_suffix']}"
            
        return base_name
    
    def _create_heatmaps(self):
        """Create spatial heatmaps for specified facilities and months"""
        output_dir = Path(self.config['output_dir'])
        base_name = self._get_base_name()
        heatmap_paths = []
        
        # Get facilities to visualize
        heatmap_facilities = self.config.get('heatmap_facilities', 'auto')
        if heatmap_facilities == 'auto':
            facilities_to_plot = sorted(self.X_test['facility_id'].unique())
        else:
            facilities_to_plot = heatmap_facilities
        
        months_to_plot = self.config.get('heatmap_months', [3, 6, 9])
        extent_km = self.config.get('heatmap_extent_km', 50)
        style = self.config.get('heatmap_style', 'side_by_side')
        
        print(f"Generating heatmaps for facilities: {facilities_to_plot}")
        print(f"Months: {months_to_plot}, Style: {style}")
        
        for facility_id in facilities_to_plot:
            for month in months_to_plot:
                try:
                    heatmap_path = self._create_single_heatmap(
                        facility_id, month, extent_km, style, base_name, output_dir
                    )
                    if heatmap_path:
                        heatmap_paths.append(heatmap_path)
                except Exception as e:
                    print(f"Warning: Failed to create heatmap for {facility_id}, month {month}: {e}")
        
        return heatmap_paths
    
    def _create_single_heatmap(self, facility_id, month, extent_km, style, base_name, output_dir):
        """Create a single heatmap for specified facility and month"""
        # Filter data for this facility and month
        facility_mask = (self.X_test['facility_id'] == facility_id) & (self.X_test['month'] == month)
        
        if not facility_mask.any():
            print(f"No data found for facility {facility_id}, month {month}")
            return None
        
        # Get facility data
        facility_data = self.X_test[facility_mask].copy()
        facility_measured = self.y_test[facility_mask].copy()
        facility_predicted = self.y_pred[facility_mask]
        
        if len(facility_data) < 3:
            print(f"Insufficient data points ({len(facility_data)}) for facility {facility_id}, month {month}")
            return None
        
        # Get grid coordinates
        grid_i = facility_data['grid_i'].values
        grid_j = facility_data['grid_j'].values
        
        # Create figure based on style
        if style == 'side_by_side':
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            axes = [ax1, ax2]
            titles = ['Measured PM₂.₅', 'Predicted PM₂.₅']
            data_arrays = [facility_measured.values, facility_predicted]
        elif style == 'difference':
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            axes = [ax]
            titles = ['Prediction Error (Predicted - Measured)']
            data_arrays = [facility_predicted - facility_measured.values]
        elif style == 'both':
            fig, ((ax1, ax2), (ax3, ax_empty)) = plt.subplots(2, 2, figsize=(15, 12))
            axes = [ax1, ax2, ax3]
            titles = ['Measured PM₂.₅', 'Predicted PM₂.₅', 'Prediction Error']
            data_arrays = [facility_measured.values, facility_predicted, facility_predicted - facility_measured.values]
            ax_empty.axis('off')  # Hide the fourth subplot
        
        # Create heatmaps
        for ax, title, data in zip(axes, titles, data_arrays):
            # Create scatter plot with color mapping
            if 'Error' in title:
                # Use diverging colormap for error
                vmax = max(abs(data.min()), abs(data.max()))
                scatter = ax.scatter(grid_i, grid_j, c=data, cmap='RdBu_r', 
                                   vmin=-vmax, vmax=vmax, s=60, alpha=0.8)
            else:
                # Use sequential colormap for concentrations
                scatter = ax.scatter(grid_i, grid_j, c=data, cmap='turbo', 
                                   s=60, alpha=0.8)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            unit = ' (log)' if self.config['target_log_transform'] else ' (μg/m³)'
            cbar.set_label(f'PM₂.₅{unit}' if 'Error' not in title else f'Error{unit}')
            
            # Set labels and title
            ax.set_xlabel('Grid I')
            ax.set_ylabel('Grid J')
            ax.set_title(f'{title}\n{facility_id}, Month {month}')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
        
        # Add overall title with metrics for this facility/month subset
        if len(facility_measured) > 1:
            r2_subset = r2_score(facility_measured, facility_predicted)
            rmse_subset = np.sqrt(mean_squared_error(facility_measured, facility_predicted))
            fig.suptitle(f'Spatial Distribution - {facility_id} (Month {month})\n'
                        f'R² = {r2_subset:.3f}, RMSE = {rmse_subset:.2f}, N = {len(facility_data)}',
                        fontsize=14)
        
        # Save figure
        heatmap_filename = f"{base_name}_heatmap_{facility_id}_month{month}_{style}.png"
        heatmap_path = output_dir / heatmap_filename
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return heatmap_path