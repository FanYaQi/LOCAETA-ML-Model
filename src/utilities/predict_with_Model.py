import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

from utils.path_util import DATA_PATH, MODELS_PATH, FIGURES_PATH

class FacilityTransferEvaluator:
    """
    Evaluate transfer learning performance using pre-processed facility data
    """
    
    def __init__(self, target_facility: str, source_model: str, year: int = 2023, grid_size: int = 24):
        """
        Initialize transfer evaluator
        
        Args:
            target_facility: Name of target facility (e.g., 'RMBC')
            source_model: Name of source model (e.g., 'suncor_weighted')
            year: Year of data
            grid_size: Grid size used
        """
        self.target_facility = target_facility
        self.source_model = source_model
        self.year = year
        self.grid_size = grid_size
        
        # Load trained model
        self.load_trained_model()
        
        print(f"Transfer Learning Evaluator initialized:")
        print(f"  Target facility: {target_facility}")
        print(f"  Source model: {source_model} ({year})")
        print(f"  Grid size: {grid_size}x{grid_size}")
    
    def load_trained_model(self):
        """Load trained model components"""
        try:
            # Load model, scaler, and metadata
            self.model = joblib.load(f"{MODELS_PATH}/rf_model_{self.source_model}_{self.year}.joblib")
            self.scaler = joblib.load(f"{MODELS_PATH}/scaler_{self.source_model}_{self.year}.joblib") 
            self.metadata = joblib.load(f"{MODELS_PATH}/metadata_{self.source_model}_{self.year}.joblib")
            
            print(f"\nLoaded trained model:")
            print(f"  Training R²: {self.metadata['results']['r2']:.4f}")
            print(f"  Training RMSE: {self.metadata['results']['rmse']:.2e}")
            print(f"  Features: {len(self.metadata['feature_names'])}")
            
            # Load weighting parameters if available
            try:
                base_model_name = self.source_model.split('_')[0]  # e.g., 'suncor' from 'suncor_weighted'
                self.weighting_params = joblib.load(f"{MODELS_PATH}/weighting_params_{base_model_name}_{self.year}.joblib")
                print(f"  Weighting: {self.weighting_params['method']} (power={self.weighting_params.get('percentile_power', 'N/A')})")
            except:
                self.weighting_params = None
                print("  Weighting: Standard (no weighting params found)")
                
        except FileNotFoundError as e:
            print(f"Error loading trained model: {e}")
            print("Please ensure you have trained the source model first!")
            raise
    
    def load_processed_data(self):
        """Load pre-processed target facility data"""
        try:
            # Load features and targets
            X_path = f"{DATA_PATH}/processed_data/X_features_{self.target_facility}_{self.year}_grid{self.grid_size}.csv"
            y_path = f"{DATA_PATH}/processed_data/y_target_{self.target_facility}_{self.year}_grid{self.grid_size}.csv"
            
            self.X_features = pd.read_csv(X_path)
            self.y_target = pd.read_csv(y_path)
            
            print(f"\nLoaded processed {self.target_facility} data:")
            print(f"  Features: {self.X_features.shape}")
            print(f"  Targets: {self.y_target.shape}")
            print(f"  Target range: {self.y_target['pm25_concentration'].min():.2e} to {self.y_target['pm25_concentration'].max():.2e}")
            
            return self.X_features, self.y_target
            
        except FileNotFoundError as e:
            print(f"Error loading processed data: {e}")
            print(f"Please ensure you have processed {self.target_facility} data first!")
            raise
    
    def prepare_features_for_prediction(self):
        """
        Prepare features to match training model format
        
        Returns:
            Scaled feature array ready for prediction
        """
        # Remove non-feature columns (same as training)
        excluded_cols = [
            'month', 'grid_i', 'grid_j',
            'landcover_urban_percent',      
            'landcover_forest_percent',     
            'landcover_agriculture_percent' 
        ]
        
        available_features = [col for col in self.X_features.columns if col not in excluded_cols]
        training_features = self.metadata['feature_names']
        
        # Check feature compatibility
        missing_features = set(training_features) - set(available_features)
        extra_features = set(available_features) - set(training_features)
        
        if missing_features:
            print(f"Warning: Missing features in target data: {missing_features}")
            
        if extra_features:
            print(f"Info: Extra features in target data (will be ignored): {extra_features}")
        
        # Select and order features to match training
        try:
            X_selected = self.X_features[training_features].values
        except KeyError as e:
            print(f"Error: Cannot find required training features in target data: {e}")
            print(f"Available features: {available_features}")
            print(f"Required features: {training_features}")
            raise
        
        # Scale features using trained scaler
        X_scaled = self.scaler.transform(X_selected)
        
        print(f"\nFeatures prepared for prediction:")
        print(f"  Selected features: {len(training_features)}")
        print(f"  Scaled shape: {X_scaled.shape}")
        
        return X_scaled
    
    def make_predictions(self, X_scaled):
        """Make predictions using trained model"""
        predictions = self.model.predict(X_scaled)
        
        print(f"\nPredictions generated:")
        print(f"  Range: {predictions.min():.2e} to {predictions.max():.2e}")
        print(f"  Mean: {predictions.mean():.2e} ± {predictions.std():.2e}")
        
        return predictions
    
    def evaluate_performance(self, y_true, y_pred):
        """
        Evaluate transfer learning performance
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            
        Returns:
            Dictionary of performance metrics
        """
        # Overall metrics
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-18))) * 100
        
        results = {
            'overall': {
                'r2': r2,
                'rmse': rmse,
                'mae': mae,
                'mape': mape,
                'n_samples': len(y_true)
            }
        }
        
        # Performance by concentration percentiles
        percentiles = [50, 75, 90, 95, 99]
        results['by_percentile'] = {}
        
        for p in percentiles:
            threshold = np.percentile(y_true, p)
            high_mask = y_true > threshold
            
            if np.any(high_mask):
                y_true_high = y_true[high_mask]
                y_pred_high = y_pred[high_mask]
                
                if len(y_true_high) > 1:  # Need at least 2 points for R²
                    high_r2 = r2_score(y_true_high, y_pred_high)
                    high_rmse = np.sqrt(mean_squared_error(y_true_high, y_pred_high))
                    
                    results['by_percentile'][f'top_{100-p}%'] = {
                        'r2': high_r2,
                        'rmse': high_rmse,
                        'threshold': threshold,
                        'n_samples': len(y_true_high)
                    }
        
        # Performance degradation compared to training
        training_r2 = self.metadata['results']['r2']
        training_rmse = self.metadata['results']['rmse']
        r2_drop = training_r2 - r2
        
        results['transfer'] = {
            'training_r2': training_r2,
            'training_rmse': training_rmse,
            'transfer_r2': r2,
            'transfer_rmse': rmse,
            'r2_drop': r2_drop,
            'relative_drop': r2_drop / training_r2 * 100 if training_r2 > 0 else 0
        }
        
        return results
    
    def print_evaluation_results(self, results):
        """Print formatted evaluation results"""
        print("\n" + "="*60)
        print(f"TRANSFER LEARNING EVALUATION")
        print(f"Source: {self.source_model} → Target: {self.target_facility}")
        print("="*60)
        
        # Overall performance
        overall = results['overall']
        print(f"\nOverall Performance ({overall['n_samples']} samples):")
        print(f"  R²: {overall['r2']:.4f}")
        print(f"  RMSE: {overall['rmse']:.2e}")
        print(f"  MAE: {overall['mae']:.2e}")
        print(f"  MAPE: {overall['mape']:.2f}%")
        
        # Transfer performance
        transfer = results['transfer']
        print(f"\nTransfer Learning Analysis:")
        print(f"  Training R² (source): {transfer['training_r2']:.4f}")
        print(f"  Transfer R² (target): {transfer['transfer_r2']:.4f}")
        print(f"  Performance drop: {transfer['r2_drop']:.4f} ({transfer['relative_drop']:.1f}%)")
        print(f"  Training RMSE: {transfer['training_rmse']:.2e}")
        print(f"  Transfer RMSE: {transfer['transfer_rmse']:.2e}")
        
        # Performance by concentration ranges
        if results['by_percentile']:
            print(f"\nPerformance by Concentration Range:")
            for range_name, metrics in results['by_percentile'].items():
                print(f"  {range_name:>8}: R²={metrics['r2']:.3f}, RMSE={metrics['rmse']:.2e}, n={metrics['n_samples']}")
    
    def create_monthly_heatmaps(self, predictions, y_true=None, save_plots=True):
        """
        Create monthly prediction heatmaps
        
        Args:
            predictions: Predicted values
            y_true: True values (optional, for comparison)
            save_plots: Whether to save plots
        """
        print(f"\nCreating monthly heatmaps for {self.target_facility}...")
        
        # Create results DataFrame
        results_df = self.X_features[['month', 'grid_i', 'grid_j', 'lat', 'lon']].copy()
        results_df['predicted'] = predictions
        if y_true is not None:
            results_df['actual'] = y_true
        
        # Monthly names
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # Create predictions heatmap
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        fig.suptitle(f'{self.target_facility} - Predicted PM2.5 Concentrations (μg/m³)', fontsize=16)
        
        vmin_pred = predictions.min()
        vmax_pred = predictions.max()
        
        for month in range(1, 13):
            row = (month - 1) // 4
            col = (month - 1) % 4
            ax = axes[row, col]
            
            month_data = results_df[results_df['month'] == month]
            
            if len(month_data) > 0:
                # Create pivot table for heatmap
                pivot_pred = month_data.pivot(index='grid_i', columns='grid_j', values='predicted')
                
                # Plot heatmap
                im = ax.imshow(pivot_pred.values, cmap='YlOrRd', vmin=vmin_pred, vmax=vmax_pred, origin='lower')
                ax.set_title(f'{months[month-1]}')
                ax.set_xlabel('Grid J')
                ax.set_ylabel('Grid I')
                
                # Add facility location (center of grid)
                center_i, center_j = self.grid_size//2, self.grid_size//2
                ax.scatter(center_j, center_i, c='blue', s=100, marker='x', linewidth=3, label='Facility')
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{months[month-1]} (No Data)')
        
        plt.tight_layout()
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=axes, shrink=0.6, aspect=20)
        cbar.set_label('PM2.5 Concentration (μg/m³)')
        
        if save_plots:
            plt.savefig(f"{FIGURES_PATH}/{self.target_facility}_predictions_monthly.png", 
                       dpi=300, bbox_inches='tight')
            print(f"Saved: {FIGURES_PATH}/{self.target_facility}_predictions_monthly.png")
        
        plt.show()
        
        # If ground truth available, create comparison heatmaps
        if y_true is not None:
            self.create_comparison_heatmaps(results_df, save_plots)
    
    def create_comparison_heatmaps(self, results_df, save_plots=True):
        """Create side-by-side comparison of predictions vs actual"""
        
        # Sample a few months for comparison
        sample_months = [1, 4, 7, 10]  # Jan, Apr, Jul, Oct
        month_names = ['January', 'April', 'July', 'October']
        
        fig, axes = plt.subplots(len(sample_months), 2, figsize=(12, 16))
        fig.suptitle(f'{self.target_facility} - Predictions vs Actual Comparison', fontsize=16)
        
        vmin = min(results_df['predicted'].min(), results_df['actual'].min())
        vmax = max(results_df['predicted'].max(), results_df['actual'].max())
        
        for i, (month, month_name) in enumerate(zip(sample_months, month_names)):
            month_data = results_df[results_df['month'] == month]
            
            if len(month_data) > 0:
                # Predictions
                pivot_pred = month_data.pivot(index='grid_i', columns='grid_j', values='predicted')
                im1 = axes[i, 0].imshow(pivot_pred.values, cmap='YlOrRd', vmin=vmin, vmax=vmax, origin='lower')
                axes[i, 0].set_title(f'{month_name} - Predicted')
                axes[i, 0].set_ylabel('Grid I')
                
                # Actual
                pivot_actual = month_data.pivot(index='grid_i', columns='grid_j', values='actual')
                im2 = axes[i, 1].imshow(pivot_actual.values, cmap='YlOrRd', vmin=vmin, vmax=vmax, origin='lower')
                axes[i, 1].set_title(f'{month_name} - Actual')
                
                # Add facility markers
                center_i, center_j = self.grid_size//2, self.grid_size//2
                axes[i, 0].scatter(center_j, center_i, c='blue', s=100, marker='x', linewidth=3)
                axes[i, 1].scatter(center_j, center_i, c='blue', s=100, marker='x', linewidth=3)
                
                if i == len(sample_months) - 1:  # Last row
                    axes[i, 0].set_xlabel('Grid J')
                    axes[i, 1].set_xlabel('Grid J')
        
        plt.tight_layout()
        
        # Add colorbar
        cbar = plt.colorbar(im2, ax=axes, shrink=0.8, aspect=30)
        cbar.set_label('PM2.5 Concentration (μg/m³)')
        
        if save_plots:
            plt.savefig(f"{FIGURES_PATH}/{self.target_facility}_comparison_heatmaps.png", 
                       dpi=300, bbox_inches='tight')
            print(f"Saved: {FIGURES_PATH}/{self.target_facility}_comparison_heatmaps.png")
        
        plt.show()
    
    def create_performance_plots(self, y_true, y_pred, save_plots=True):
        """Create performance analysis plots"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Scatter plot: Predicted vs Actual
        axes[0, 0].scatter(y_true, y_pred, alpha=0.6, s=20)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual PM2.5 (μg/m³)')
        axes[0, 0].set_ylabel('Predicted PM2.5 (μg/m³)')
        
        r2 = r2_score(y_true, y_pred)
        axes[0, 0].set_title(f'Predicted vs Actual (R² = {r2:.3f})')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Residuals plot
        residuals = y_true - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6, s=20)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted PM2.5 (μg/m³)')
        axes[0, 1].set_ylabel('Residuals (μg/m³)')
        axes[0, 1].set_title('Residuals vs Predicted')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Monthly performance
        results_df = self.X_features[['month']].copy()
        results_df['actual'] = y_true
        results_df['predicted'] = y_pred
        
        monthly_r2 = []
        monthly_rmse = []
        months = []
        
        for month in range(1, 13):
            month_data = results_df[results_df['month'] == month]
            if len(month_data) > 10:  # Minimum samples for reliable metrics
                month_r2 = r2_score(month_data['actual'], month_data['predicted'])
                month_rmse = np.sqrt(mean_squared_error(month_data['actual'], month_data['predicted']))
                monthly_r2.append(month_r2)
                monthly_rmse.append(month_rmse)
                months.append(month)
        
        if monthly_r2:
            ax3 = axes[1, 0]
            ax3_twin = ax3.twinx()
            
            bars1 = ax3.bar([m-0.2 for m in months], monthly_r2, width=0.4, label='R²', alpha=0.7)
            bars2 = ax3_twin.bar([m+0.2 for m in months], monthly_rmse, width=0.4, label='RMSE', alpha=0.7, color='orange')
            
            ax3.set_xlabel('Month')
            ax3.set_ylabel('R²', color='blue')
            ax3_twin.set_ylabel('RMSE', color='orange')
            ax3.set_title('Monthly Transfer Performance')
            ax3.set_xticks(months)
            ax3.grid(True, alpha=0.3)
        
        # 4. Error distribution
        axes[1, 1].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(residuals.mean(), color='r', linestyle='--', 
                          label=f'Mean: {residuals.mean():.2e}')
        axes[1, 1].axvline(0, color='black', linestyle='-', alpha=0.5)
        axes[1, 1].set_xlabel('Residuals (μg/m³)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Distribution of Residuals')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'{self.target_facility} - Transfer Learning Performance Analysis', fontsize=14)
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f"{FIGURES_PATH}/{self.target_facility}_performance_analysis.png", 
                       dpi=300, bbox_inches='tight')
            print(f"Saved: {FIGURES_PATH}/{self.target_facility}_performance_analysis.png")
        
        plt.show()
    
    def save_predictions(self, predictions, y_true=None):
        """Save predictions to CSV file"""
        
        # Create results DataFrame
        results_df = self.X_features[['month', 'grid_i', 'grid_j', 'lat', 'lon']].copy()
        results_df['predicted_pm25'] = predictions
        
        if y_true is not None:
            results_df['actual_pm25'] = y_true
            results_df['residual'] = y_true - predictions
            results_df['absolute_error'] = np.abs(y_true - predictions)
            results_df['relative_error'] = np.abs((y_true - predictions) / (y_true + 1e-18)) * 100
        
        # Add metadata
        results_df['target_facility'] = self.target_facility
        results_df['source_model'] = self.source_model
        results_df['prediction_year'] = self.year
        
        # Save to file
        output_dir = Path(f"{DATA_PATH}/predictions")
        output_dir.mkdir(exist_ok=True)
        
        output_path = output_dir / f"predictions_{self.target_facility}_{self.year}_from_{self.source_model}.csv"
        results_df.to_csv(output_path, index=False)
        
        print(f"Predictions saved to: {output_path}")
        
        return results_df

def main():
    """
    Main function for transfer learning evaluation
    """
    
    # ===== CONFIGURATION =====
    target_facility = 'RMBC'           # Target facility name
    source_model = 'suncor_weighted'   # Source model name
    year = 2023                        # Year
    grid_size = 24                     # Grid size
    # ==========================
    
    print("="*70)
    print("FACILITY TRANSFER LEARNING EVALUATION")
    print("="*70)
    
    # Initialize evaluator
    evaluator = FacilityTransferEvaluator(
        target_facility=target_facility,
        source_model=source_model,
        year=year,
        grid_size=grid_size
    )
    
    # Load processed data
    X_features, y_target = evaluator.load_processed_data()
    
    # Prepare features for prediction
    X_scaled = evaluator.prepare_features_for_prediction()
    
    # Make predictions
    predictions = evaluator.make_predictions(X_scaled)
    
    # Get ground truth
    y_true = y_target['pm25_concentration'].values
    
    # Evaluate performance
    results = evaluator.evaluate_performance(y_true, predictions)
    evaluator.print_evaluation_results(results)
    
    # Create visualizations
    evaluator.create_monthly_heatmaps(predictions, y_true, save_plots=True)
    evaluator.create_performance_plots(y_true, predictions, save_plots=True)
    
    # Save predictions
    predictions_df = evaluator.save_predictions(predictions, y_true)
    
    print("\n" + "="*50)
    print("EVALUATION COMPLETE")
    print("="*50)
    print(f"Target: {target_facility}")
    print(f"Source: {source_model}")
    print(f"Final R²: {results['overall']['r2']:.4f}")
    print(f"Performance drop: {results['transfer']['relative_drop']:.1f}%")

if __name__ == "__main__":
    main()