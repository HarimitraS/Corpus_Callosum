# High-Precision Corpus Callosum Extraction and Morphometric Analysis from Infant Brain MRI

## Overview

This repository contains an ongoing research project focused on the automatic extraction, segmentation, and quantitative morphometric analysis of the **Corpus Callosum (CC)** from infant brain MRI. The project combines classical medical image processing techniques with deep learning to build a fully automated pipeline capable of accurately segmenting the corpus callosum and extracting clinically meaningful anatomical measurements.

The long-term objective of this research is to investigate the relationship between corpus callosum morphology, myelin maturation, and neurological abnormalities, providing a foundation for future AI-assisted diagnostic systems.

---

## Motivation

The corpus callosum is the largest white matter structure in the human brain and plays a critical role in communication between the two cerebral hemispheres. Structural abnormalities of the corpus callosum have been associated with various neurological and developmental disorders.

Manual segmentation is labor-intensive, time-consuming, and highly dependent on expert radiologists. This project aims to automate the complete workflow, producing accurate, reproducible, and clinically useful measurements that can support medical research and future clinical applications.

---

# Research Pipeline

```
Brain MRI
      │
      ▼
Mid-Sagittal Slice Extraction
      │
      ▼
Intensity Normalization
      │
      ▼
Non-Local Means Denoising
      │
      ▼
CLAHE Contrast Enhancement
      │
      ▼
Atlas Construction
      │
      ▼
ECC Affine Registration
      │
      ▼
Atlas Prior Propagation
      │
      ▼
ROI Localization
      │
      ▼
Manual Annotation (Labelme)
      │
      ▼
Ground Truth Mask Generation
      │
      ▼
U-Net (ResNet34 Encoder)
      │
      ▼
Automatic Corpus Callosum Segmentation
      │
      ▼
Segmentation Refinement
      │
      ▼
Morphological Analysis
      │
      ▼
Skeleton Extraction
      │
      ▼
Thickness Estimation
      │
      ▼
Feature Extraction
      │
      ▼
Segmentation Evaluation
      │
      ▼
Statistical Analysis (In Progress)
      │
      ▼
Clinical Correlation & Classification (Future Work)
```

---

# Techniques Implemented

### Image Preprocessing

- Mid-Sagittal Slice Extraction
- Intensity Normalization
- Non-Local Means Denoising
- CLAHE Contrast Enhancement

### ROI Localization

- Atlas Construction
- ECC Affine Registration
- Atlas Prior Propagation
- Automatic ROI Cropping

### Ground Truth Generation

- Manual Annotation using Labelme
- Binary Mask Generation

### Deep Learning

- U-Net Architecture
- ResNet34 Encoder
- GPU Accelerated Training using PyTorch

### Segmentation Refinement

- Binary Thresholding
- Morphological Opening
- Morphological Closing
- Connected Component Analysis
- Hole Filling
- Contour Smoothing

### Morphological Analysis

- Skeletonization
- Euclidean Distance Transform
- Thickness Map Generation
- Area Measurement
- Perimeter Measurement
- Bounding Box Extraction
- Skeleton Length Measurement
- Mean Thickness
- Maximum Thickness
- Minimum Thickness
- Thickness Standard Deviation

### Segmentation Evaluation

- Dice Coefficient
- Intersection over Union (IoU)
- Precision
- Recall
- Accuracy
- Specificity

---

# Current Project Status

## ✅ Completed

- MRI preprocessing pipeline
- Atlas-based ROI localization
- Manual annotation and ground truth generation
- U-Net based corpus callosum segmentation
- Segmentation refinement
- Morphological feature extraction
- Skeleton extraction
- Thickness estimation
- Segmentation evaluation using standard medical image segmentation metrics

---

## 🚧 Currently In Progress

The current focus of the research is on statistical analysis of the extracted corpus callosum morphometric features.

The study aims to investigate:

- Relationship between corpus callosum morphology and myelin maturation status.
- Relationship between corpus callosum thickness and radiologist-predicted brain age.
- Statistical comparison of morphometric measurements across neurological conditions.

---

# Dataset

This project utilizes the publicly available **Infant Brain MRI Dataset** from Zenodo.

The dataset includes:

- Infant brain MRI scans
- Myelin maturation status (Delayed / Normal / Accelerated)
- Chronological age
- Corrected age
- Radiologist-predicted brain age
- Neurological diagnosis

Dataset:
https://zenodo.org/records/8055666

---

# Current Research Objective

The primary objective of the current phase is to validate whether automatically extracted corpus callosum morphometric features provide clinically meaningful information.

Specifically, the project investigates:

- Association between corpus callosum morphology and delayed myelination.
- Correlation between corpus callosum thickness and radiologist-predicted brain age.
- Quantitative analysis of corpus callosum morphology across different neurological conditions.

---

# Future Scope

The current work establishes a robust automated segmentation pipeline that serves as the foundation for future research.

Future work includes:

- Statistical significance testing across different myelination groups.
- Correlation analysis with clinical metadata.
- Automated prediction of myelin maturation status using machine learning.
- Integration of additional morphometric and texture-based features.
- Development of AI-based clinical decision support systems.
- Extension from 2D slice analysis to complete 3D MRI segmentation.
- Validation on larger multi-center datasets.
- Investigation of corpus callosum abnormalities associated with neurological disorders.

---

# Repository Structure

```
Corpus_Callosum/
│
├── MRI Preprocessing
├── ROI Localization
├── Ground Truth Generation
├── U-Net Segmentation
├── Segmentation Refinement
├── Morphological Analysis
├── Segmentation Evaluation
├── Statistical Analysis
├── Clinical Validation
└── Documentation
```

---

# Technologies Used

- Python
- PyTorch
- U-Net
- ResNet34
- OpenCV
- NumPy
- SciPy
- scikit-image
- scikit-learn
- Pandas
- Matplotlib
- Labelme

---

# Current Development Stage

**Project Status:** Active Research Project

**Completed:** Automated Segmentation Pipeline, Morphological Analysis & Segmentation Evaluation

**Current Stage:** Statistical Analysis and Clinical Correlation

**Next Stage:** Machine Learning-based Clinical Prediction and Validation

---

# Disclaimer

This repository represents an ongoing academic research project. The current implementation is intended for research and educational purposes and should not be used for clinical diagnosis without appropriate medical validation.
