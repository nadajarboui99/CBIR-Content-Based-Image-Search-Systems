# Content-Based Image Retrieval (CBIR) System

A comprehensive implementation of a Content-Based Image Retrieval system that searches and retrieves similar images based on visual features including color, texture, and shape.

## Overview

This project implements a CBIR system that allows users to query a database of images and retrieve visually similar images. The system extracts multiple visual features from images and uses distance metrics to find the most similar matches.

## Features

### Image Feature Extraction

- Color Moments: Statistical features (mean and variance) across RGB channels
- HSV Histogram: Color distribution in HSV color space for better lighting invariance
- Texture Features: GLCM (Gray-Level Co-occurrence Matrix) properties
- Shape Features: HOG (Histogram of Oriented Gradients) descriptors

### Search Capabilities

- Whole image comparison using Euclidean distance
- Multi-descriptor feature matching
- Top-K retrieval of most similar images
- Transformation robustness testing

## Requirements
```
numpy
opencv-python
matplotlib
scikit-image
scipy
```

## Installation

Install the required packages using pip:
```bash
pip install numpy opencv-python matplotlib scikit-image scipy
```

For Google Colab users:
```python
!pip install scikit-image opencv-python
```

## Usage

### Basic Setup
```python
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops, hog
from scipy.spatial.distance import euclidean

# Load images from folders
database_images, database_names = load_images_from_folder(base_path)
query_images, query_names = load_images_from_folder(requete_path)
```

### Indexation Phase
```python
# Extract features from all database images
features_matrix, indexed_names = CBIR_Indexation(database_images, database_names)
```

### Search Phase
```python
# Search for similar images
results = CBIR_Recherche(query_image, features_matrix, indexed_names, top_k=5)
```

## Implementation Details

### Feature Extraction Functions

**Color Moments**
```python
color_Moments(img)
```

Returns a 6-dimensional vector: [mean_R, mean_G, mean_B, var_R, var_G, var_B]

**HSV Histogram**
```python
hsvHistogramFeatures(img, bins=(8, 8, 8))
```

Returns a normalized histogram of the HSV color space

**Texture Features**
```python
textureFeatures(img)
```

Returns a 4-dimensional vector: [contrast, correlation, energy, homogeneity]

**HOG Features**
```python
shapeFeaturesHOG(img)
```

Returns HOG descriptors computed on grayscale images

### Distance Metric

The system uses Euclidean distance to measure similarity between feature vectors:
```
distance = sqrt(sum((query_features - db_features)^2))
```

## Experimental Results

### Search Methods Comparison

**1. Whole Image Search**

Direct pixel-level comparison - simple but sensitive to transformations with limited semantic understanding.

**2. Color-based Search**

Using color moments - better semantic relevance and captures color distribution.

**3. Color + Histogram**

Combined approach - more robust to lighting variations with improved discrimination.

**4. Color + Histogram + Texture**

Multi-feature approach - captures both color and pattern information, better for textured images.

**5. Complete Descriptor**

All features combined - best overall performance capturing color, texture, and shape.

### Robustness to Transformations

The system is tested against various geometric transformations:

- **Translation**: Good invariance due to global descriptors
- **Rotation**: Moderate sensitivity (HOG and GLCM are orientation-dependent)
- **Scaling**: Handled through image resizing before feature extraction
- **Combined transformations**: Performance degrades with multiple simultaneous transformations

## Limitations

- Sensitive to significant rotations due to HOG and GLCM orientation dependency
- Performance decreases with extreme scale variations
- Computational cost increases with descriptor complexity
- No invariance to perspective transformations

## Potential Improvements

- Implement rotation-invariant descriptors (SIFT, SURF, ORB)
- Add deep learning-based features (CNN embeddings)
- Implement distance metric learning
- Add data augmentation to the indexing phase
- Optimize feature extraction with parallel processing
- Implement approximate nearest neighbor search for large databases
