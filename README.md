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

## Key Features

### Data Processing Pipeline

1. **Raw Data Import**
   - Laboratory-controlled surface recordings
   - Real-world cycling sessions with GPS tracking
   - Multiple participant data collection

2. **Preprocessing** ([`utils/preprocessing.py`](utils/preprocessing.py))
   - Timestamp normalization
   - Missing value imputation
   - Data filtering and cleaning
   - Downsampling to target frequencies (30Hz or 100Hz)
   - Standard normalization (zero mean, unit variance)

3. **Segmentation** ([`utils/segmentation.py`](utils/segmentation.py))
   - Sliding window approach with configurable overlap
   - Window sizes: 0.5s (50 samples) or 1.0s (100 samples) at 100Hz
   - Window sizes: 1.0s (30 samples) at 30Hz
   - 50% overlap between consecutive windows
   - Label assignment based on majority voting

4. **Data Augmentation** ([`notebooks/training/transformations.py`](notebooks/training/transformations.py))
   - Gaussian noise injection
   - Amplitude scaling
   - Axis rotation
   - Time warping
   - Channel shuffling

### Model Architecture

The project implements SimCLR-based contrastive learning:

- **Pre-training Phase**: Self-supervised learning on unlabeled data
- **Fine-tuning Phase**: Supervised learning on labeled data
- **Architecture**: CNN-based feature extractor with projection head
- **Loss Function**: Contrastive loss (NT-Xent)

**Key Files:**
- [`notebooks/training/simclr_models.py`](notebooks/training/simclr_models.py) - Model architecture
- [`notebooks/training/simclr_utitlities.py`](notebooks/training/simclr_utitlities.py) - Training utilities
- [`notebooks/training/data_pre_processing.py`](notebooks/training/data_pre_processing.py) - Data preprocessing

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

## Usage

### Visualization

```python
import sys
sys.path.append('utils/')
from visualization import *
import pandas as pd

# Load accelerometer data
df = pd.read_csv('data/RoadRoughness/Raw/Asphalt/P1/Accelerometer_filtered.csv')

# Check sampling frequency
print_sampling_frequency(df)

# Interactive visualization with Plotly
plot_accelerometer_data(df, "Asphalt Surface")

# Publication-quality plot with Matplotlib
plot_accelerometer_data_bachelorarbeit(df)
```

### Data Preprocessing

```python
from preprocessing import *

# Downsample to 100Hz
df_100hz = downsample_to_frequency(df, target_hz=100, timestamp_col='NTP')

# Normalize acceleration data
data_normalized = normalize_3d_data(df_100hz[['Acc-X', 'Acc-Y', 'Acc-Z']].values)

# Random sampling for balanced datasets
df_balanced = select_random_samples(df_100hz, n_samples=1000, random_state=42)
```

### Data Segmentation

```python
from segmentation import *

# Segment data with 50% overlap
segments = segment_acceleration_data_overlapping_numpy(
    df_100hz,
    window_size=100,      # 1 second at 100Hz
    overlap=50,           # 50% overlap
    channels=['Acc-X', 'Acc-Y', 'Acc-Z']
)

# Segment with labels
segments, labels = segment_acceleration_data_overlapping_numpy_with_curb_activity(
    df_100hz,
    window_size=100,
    overlap=50,
    channels=['Acc-X', 'Acc-Y', 'Acc-Z'],
    label_col='curb_activity'
)

print(f"Segments shape: {segments.shape}")  # (n_segments, 100, 3)
print(f"Labels shape: {labels.shape}")      # (n_segments,)
```

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

## Evaluation

The project includes comprehensive field validation:
- Real-world cycling sessions with multiple participants
- Various weather and road conditions
- Performance metrics: accuracy, precision, recall, F1-score
- Confusion matrices and classification reports

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
