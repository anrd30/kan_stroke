# Kannada Numeral Shape Descriptor (KNSD)

**Novel Loop-Based Feature Extraction for Kannada Handwritten Numeral Recognition**

## Overview

This project implements the **Kannada Numeral Shape Descriptor (KNSD)** - a novel 29-dimensional feature vector specifically designed for recognizing Kannada handwritten numerals (೦-೯).

## Key Innovation

KNSD exploits structural properties unique to Kannada numerals:

| Feature | Description | Discriminative For |
|---------|-------------|-------------------|
| **Loop Detection** | Counts closed holes | ೦ (1 loop), ೮ (2 loops) |
| **Endpoints** | Stroke termination points | ೧ (2 endpoints) |
| **Junctions** | Intersection points | Complex characters |
| **Circularity** | Shape roundness | ೦ (highly circular) |

## Results

**Dataset**: Kannada-MNIST (4,800 train / 1,200 test)

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **90.67%** |
| **Features** | 29 |
| **Model Size** | 312 KB |
| **Inference** | 0.19 ms/image |

## Installation

```bash
pip install numpy opencv-python scikit-learn scikit-image pandas matplotlib seaborn tqdm
```

## Usage

### Extract Features
```python
from src.knsd_features import preprocess_image, extract_knsd_features

# Load and preprocess image
img = cv2.imread('digit.png')
binary = preprocess_image(img)

# Extract 29-dimensional KNSD features
features = extract_knsd_features(binary)
print(f"Features: {features.shape}")  # (29,)
```

### Train Classifier
```bash
cd experiments
python train_evaluate.py
```

## Project Structure

```
novel_knsd/
├── src/                    # Core KNSD implementation
│   ├── __init__.py
│   └── knsd_features.py    # Feature extraction
├── experiments/            # Training scripts
│   └── train_evaluate.py
├── data/                   # Datasets
│   ├── numerics/           # Original images
│   └── kannada_mnist/      # Kannada-MNIST CSVs
├── models/                 # Saved models
├── results/                # Output figures
└── README.md
```

## Citation

If you use this work, please cite:
```
[Your paper citation here]
```

## License

MIT License
