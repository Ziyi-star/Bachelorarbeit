# Bachelorarbeit - Cyclist Curb Detection under Varying Road Roughness in the Real World Using Contrastive Learning

A machine learning project for classifying road surfaces and detecting curbs using accelerometer(x,y,z) data collected from bicycle handlebar sensors during cycling.

## Overview

This project uses deep learning and the SimCLR (Simple Framework for Contrastive Learning of Visual Representations) approach to classify different road surface types and detect curb crossings based on 3-axis accelerometer data. The system supports both laboratory-controlled environments and real-world cycling scenarios.

## Project Structure

```
├── data/                           # Raw and processed data
│   ├── Curb/                     # Curb detection data from participants (P3, P6, P11, P12, P18, P21)
│   ├── Field_validation/         # Field validation datasets at various sampling rates
│   ├── Real_world_cycling/       # Real-world cycling datasets
│   ├── RoadRoughness/            # Laboratory road surface data
│   └── Training/                 # Processed training datasets
│       ├── 1s_100hz/             # 1 second windows at 100Hz
│       └── 1s_30hz/              # 1 second windows at 30Hz
│       
│
├── models/                        # Trained models
│   ├── 1s_100hz_unbalanced/     # Models for 100Hz unbalanced data
│   └── 1s_30hz/                 # Models for 30Hz data
│       ├── 2class_unbalanced_lab/     
│       ├── 2class_unbalanced_real_world_a/
│       ├── 2class_unbalanced_real_world_a_with_pseudo/
│       └── 7class_balanced/
│
├── notebooks/                     # Jupyter notebooks for running code
│   ├── field_validation/         # Field validation and preprocessing
│   ├── preprocessing/            # Data preprocessing pipelines
│   │   ├── data_processing/      # Individual surface type processing
│   │   └── combine_train_test_spilt/  # Dataset combination and train split
│   ├── training/                 # Model training workflows
│   ├── training_data/            # Training data uploaded for Colab access 
│   └── visualization/            # Data visualization
│
└── utils/                         # Utility modules
    ├── analyse.py                # Analysis functions
    ├── automation.py             # Automation scripts
    ├── preprocessing.py          # Data preprocessing utilities
    ├── segmentation.py           # Time-series segmentation
    └── visualization.py          # Plotting and visualization
```

## Classification Tasks

### Road Surface Types (7-class)
- **Asphalt** - Smooth paved roads
- **Cobblestone** - Traditional stone pavement
- **Compact Gravel** - Compressed gravel paths
- **Dirt** - Unpaved dirt roads
- **Paving Stone** - Brick or stone paving
- **Curb Scene 0** - Paving Stone (pattern b)
- **Curb Scene 1** - Curb crossing event

### Curb Detection (2-class)
- **No-curb** (Class 0) - Normal cycling
- **Curb** (Class 1) - Curb crossing


### Data Processing Pipeline

1. **Preprocessing & Segmentation** ([`notebooks/preprocessing/data_processing])
   - Missing value imputation
   - Downsampling to target frequencies (30Hz or 100Hz)
   - Standard normalization (zero mean, unit variance)
   - Sliding window approach with configurable overlap
   - Window sizes: 0.5s (50 samples) or 1.0s (100 samples) at 100Hz
   - Window sizes: 1.0s (30 samples) at 30Hz
   - 50% overlap between consecutive windows
   - Label assignment based on majority voting
2. **Dataset Combination & Train-Test Split** ([`notebooks/preprocessing/combine_train_test_spilt/`](notebooks/preprocessing/combine_train_test_spilt/))
   - Combine preprocessed segments from all participants
   - Stratified train-test split (80/20)
   - leave one man out strategy
   - Class balancing (oversampling/undersampling)
   - Generate dataset statistics and participant metadata

3. **Training using SimCLR** ([`notebooks/training/`](notebooks/training/))
   
   **Pre-training Phase** (Self-supervised Learning):
   - Train on unlabeled accelerometer data to learn general features
   - Apply data augmentations: noise injection, scaling, rotation, time warping, channel shuffling
   - Use contrastive loss (NT-Xent) to maximize agreement between augmented views
   - Base encoder: CNN architecture for time-series feature extraction
   - Projection head: MLP for mapping to contrastive learning space
   
   **Fine-tuning Phase** (Supervised Learning):
   - Initialize with pre-trained encoder weights
   - Train on labeled data for specific classification tasks
   - Two training scenarios:
     - **2-class**: Curb vs. non-curb detection
     - **7-class**: Multi-surface classification
   - Freezing/unfreezing strategies for transfer learning
   
   **Training Configurations**:
   - Batch size: 128
   - Learning rate: 0.001 with decay
   - Optimizer: Adam
   - Epochs: 50-100 (with early stopping)
   - Data augmentation probability: 50%
   
   **Model Variants**:
   - [`1s_100hz_unbalanced/`](models/1s_100hz_unbalanced/): High-frequency models with natural class distribution
   - [`1s_30hz/`](models/1s_30hz/): Lower-frequency models for resource-constrained scenarios
   - Training notebooks available for both balanced and unbalanced datasets

4. **Field Validation Analyse** ([`notebooks/field_validation/`](notebooks/field_validation)
   - Real-world cycling data
   - Ground truth annotation using video timestamps
   - False negative analysis with video matching


## Data Format

### Raw Accelerometer Data
CSV files with the following columns:
- `NTP` - Network Time Protocol timestamp
- `Acc-X` - X-axis acceleration (m/s²)
- `Acc-Y` - Y-axis acceleration (m/s²)
- `Acc-Z` - Z-axis acceleration (m/s²)

### Processed Segments
NumPy arrays with shape:
- `(n_samples, window_size, 3)` - 3D array where:
  - `n_samples`: number of segments
  - `window_size`: 30, 50, or 100 (depending on configuration)
  - `3`: X, Y, Z acceleration channels

### Labels
- **2-class**: Binary labels (0=non-curb, 1=curb)
- **7-class**: Categorical labels (0-6 for each surface type)


### Training Models

Navigate to [`notebooks/training/`](notebooks/training/) and use the Jupyter notebooks:

- **2-class (Curb Detection)**:
  - [`train_1s_100hz_2class_balanced.ipynb`](notebooks/training/train_1s_100hz_2class_balanced.ipynb)
  - [`train_1s_100hz_2class_unbalanced.ipynb`](notebooks/training/train_1s_100hz_2class_unbalanced.ipynb)
  - [`train_1s_30hz_2class_unbalanced.ipynb`](notebooks/training/train_1s_30hz_2class_unbalanced.ipynb)

- **7-class (Surface Classification)**:
  - [`train_1s_100hz_7class.ipynb`](notebooks/training/train_1s_100hz_7class.ipynb)
  - [`train_1s_30hz_7_class.ipynb`](notebooks/training/train_1s_30hz_7_class.ipynb)

- **Real world Evaluation**:
- **Real-World Training & Evaluation**:
  - [`train_1s_30hz_2class_real_world_phase_1_to_3_version_a.ipynb`](notebooks/training/train_1s_30hz_2class_real_world_phase_1_to_3_version_a.ipynb): Subject A for test
  - [`train_1s_30hz_2class_real_world_phase_1_to_3_version_b.ipynb`](notebooks/training/train_1s_30hz_2class_real_world_phase_1_to_3_version_b.ipynb): Subject B for test
  - [`train_1s_30hz_2class_real_world_phase_1_to_3_version_c.ipynb`](notebooks/training/train_1s_30hz_2class_real_world_phase_1_to_3_version_c.ipynb): Subject C for test
  - [`train_1s_30hz_2class_real_world_phase_1_to_3_version_d.ipynb`](notebooks/training/train_1s_30hz_2class_real_world_phase_1_to_3_version_d.ipynb): Subject D for test
  - [`train_1s_30hz_2class_real_world_phase_4_to_5_version_a.ipynb`](notebooks/training/train_1s_30hz_2class_real_world_phase_4_to_5_version_a.ipynb): Extended phase using pseudo labels

  **Multi-phase Training Strategy**:
  - Phase 1-3: Initial pre-training and fine-tuning on real-world data
  - Phase 4-5: Advanced training with expanded datasets
  - Cross-subject validation for generalization assessment

- **Field Evaluation**:
  - [`field_evaluation_1s_100hz_analyse.ipynb`](notebooks/training/field_evaluation_1s_100hz_analyse.ipynb)
  - [`field_evaluation_1s_30hz_analyse.ipynb`](notebooks/training/field_evaluation_1s_30hz_analyse.ipynb)

## Experimental Configurations

### Configuration 1: 1s Window at 100Hz (Unbalanced)
- **Window Size**: 100 samples (1.0 second)
- **Sampling Rate**: 100Hz
- **Overlap**: 50%
- **Data Balance**: Natural distribution
- **Model Location**: [`models/1s_100hz_unbalanced/`](models/1s_100hz_unbalanced/)

### Configuration 2: 1s Window at 30Hz
- **Window Size**: 30 samples (1.0 second)
- **Sampling Rate**: 30Hz
- **Overlap**: 50%
- **Configurations**: Suject (a, b, c, d) for real-world scenarios
- **Model Location**: [`models/1s_30hz/`](models/1s_30hz/)

## Contributing

This is a Bachelor's thesis project. For questions or collaboration:
1. Review the notebooks in [`notebooks/`](notebooks/)
2. Check utility functions in [`utils/`](utils/)
3. Examine trained models in [`models/`](models/)

## License

This project uses code adapted from SimCLR implementations:
- Licensed under GNU General Public License v3.0
- Based on Tang et al. SimCLR work (https://arxiv.org/abs/2011.11542)

---

*Last Updated: December 2025*
