# LOCAETA-ML Project
Industrial facilities contribute 29% of US PM2.5 emissions, yet traditional source attribution methods require weeks of computation time for one facility alone, preventing real-time air quality management. This study presents a machine learning approach that accelerates PM2.5 facility source attribution by 2,000,000 times while maintaining 98% accuracy of predicting the concentration pattern.

### Key Achievements:
- Conducted preliminary statistical analysis validating facility impact signals in satellite data (r = 0.52, 37.8% of hotspots within 1km of facilities)
- Developed Random Forest surrogate models using satellite-derived PM2.5 data (1km resolution) for Denver Area:
- Demonstrated 98% accuracy in predicting facility impact patterns as traditional air dispersion model – HYSPLIT’s output
- Achieved great temporal validation performance (R² > 0.9) for future predictions
- Enabled facility impact assessment for LOCAETA integration

### Impact: 
The denver study case demonstrates potential for transforming facility source attribution from a research tool into an operational capability, with promising applications for permit reviews, emergency response, and environmental justice assessments pending broader validation.

## LOCAETA-ML Code Structure

### Core Modules
- `core/base_trainer.py` - Base ML trainer with common functionality
- `utils/path_util.py` - Path utilities

### Training Modules
- `trainers/configurable_trainer.py` - Enhanced configurable trainer with heatmap support
- `trainers/multi_facility_trainer.py` - Multi-facility spatial validation trainer

### Data Processing Modules
- `data_processing/base_processor.py` - Base data processor with common functionality
- `data_processing/multi_facility_processor.py` - Multi-facility data processor

### Main Entry Points
- `main_trainer.py` - Unified training script (USE THIS)

### Legacy Files (Still Available)
- `trainModel.py` - Original configurable trainer
- `multi_trainModel.py` - Original multi-facility trainer  
- `data_processing_MLtrain.py` - Original single facility processor
- `data_processing_MLtrain_multifacility.py` - Original multi-facility processor

### Analysis and Utilities
- `Analysis_PM_facility.py` - Facility analysis utilities
- `combine_data.py` - Data combination utilities
- `predict_with_Model.py` - Model prediction utilities
- `multiple_model_comparison.py` - Model comparison utilities
- `site_comprehensive.py` - Site analysis utilities
- `tri_validation_ML_experiment.py` - Triple validation experiments

## Quick Start

### Option 1: Use New Unified System (Recommended)
```python
from main_trainer import main
main()  # Runs both configurable and multi-facility training
```

### Option 2: Use Individual Trainers
```python
from trainers.configurable_trainer import ConfigurableMLTrainer
from trainers.multi_facility_trainer import MultiFacilityModelTrainer

# training code here
```