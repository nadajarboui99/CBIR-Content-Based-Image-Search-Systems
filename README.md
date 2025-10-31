Content-Based Image Retrieval (CBIR) System
A comprehensive implementation of a Content-Based Image Retrieval system that searches and retrieves similar images based on visual features including color, texture, and shape.

Overview

This project implements a CBIR system that allows users to query a database of images and retrieve visually similar images. The system extracts multiple visual features from images and uses distance metrics to find the most similar matches.

Features
Image Feature Extraction
Color Moments: Statistical features (mean and variance) across RGB channels
HSV Histogram: Color distribution in HSV color space for better lighting invariance
Texture Features: GLCM (Gray-Level Co-occurrence Matrix) properties including contrast, correlation, energy, and homogeneity
Shape Features: HOG (Histogram of Oriented Gradients) descriptors for capturing edge and contour information

Search Capabilities

Whole image comparison using Euclidean distance
Multi-descriptor feature matching
Top-K retrieval of most similar images
Transformation robustness testing

Requirements
numpy
opencv-python
matplotlib
scikit-image
scipy
Installation
Install the required packages using pip:
bashpip install numpy opencv-python matplotlib scikit-image scipy
For Google Colab users:
python!pip install scikit-image opencv-python


Usage
Basic Setup
python# Import required libraries
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops, hog
from scipy.spatial.distance import euclidean

# Load images from folders
database_images, database_names = load_images_from_folder(base_path)
query_images, query_names = load_images_from_folder(requete_path)

Indexation Phase
python# Extract features from all database images
features_matrix, indexed_names = CBIR_Indexation(database_images, database_names)
Search Phase
python# Search for similar images
results = CBIR_Recherche(query_image, features_matrix, indexed_names, top_k=5)
Implementation Details
Feature Extraction Functions
Color Moments
pythoncolor_Moments(img)
Returns a 6-dimensional vector: [mean_R, mean_G, mean_B, var_R, var_G, var_B]
HSV Histogram
pythonhsvHistogramFeatures(img, bins=(8, 8, 8))
Returns a normalized histogram of the HSV color space
Texture Features
pythontextureFeatures(img)
Returns a 4-dimensional vector: [contrast, correlation, energy, homogeneity]
HOG Features
pythonshapeFeaturesHOG(img)
Returns HOG descriptors computed on grayscale images
Distance Metric
The system uses Euclidean distance to measure similarity between feature vectors:
distance = sqrt(sum((query_features - db_features)^2))
Experimental Results
Search Methods Comparison

Whole Image Search: Direct pixel-level comparison

Simple but sensitive to transformations
Limited semantic understanding


Color-based Search: Using color moments

Better semantic relevance
Captures color distribution


Color + Histogram: Combined approach

More robust to lighting variations
Improved discrimination


Color + Histogram + Texture: Multi-feature approach

Captures both color and pattern information
Better for textured images


Complete Descriptor: All features combined

Best overall performance
Captures color, texture, and shape



Robustness to Transformations
The system is tested against various geometric transformations:

Translation: Good invariance due to global descriptors
Rotation: Moderate sensitivity (HOG and GLCM are orientation-dependent)
Scaling: Handled through image resizing before feature extraction
Combined transformations: Performance degrades with multiple simultaneous transformations

Limitations

Sensitive to significant rotations due to HOG and GLCM orientation dependency
Performance decreases with extreme scale variations
Computational cost increases with descriptor complexity
No invariance to perspective transformations

Potential Improvements

Implement rotation-invariant descriptors (SIFT, SURF, ORB)
Add deep learning-based features (CNN embeddings)
Implement distance metric learning
Add data augmentation to the indexing phase
Optimize feature extraction with parallel processing
Implement approximate nearest neighbor search for large databases