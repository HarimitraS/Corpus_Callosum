# REPRODUCIBILITY CHECKLIST AND TECHNICAL ENVIRONMENT
## Corpus Callosum Infant Brain MRI Research Project

This document outlines the software dependencies, hardware environment, random seeds, and exact command sequences required to reproduce the latest synthetic-data pipeline (`33_synthetic_delayed_150`) without modifying existing project files.

---

## 1. Verified Software & Hardware Environment

- **Operating System:** Windows 11 Enterprise / Pro (64-bit)
- **Python Version:** 3.11.0 (verified: `Python 3.11.0 [MSC v.1933 64 bit (AMD64)]`)
- **PyTorch Version:** 2.11.0+cu128
- **Torchvision Version:** 0.26.0+cu128
- **CUDA Availability:** Enabled (`True`, CUDA 12.8 compatible driver)
- **Primary GPU:** NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM)
- **Core Scientific Libraries:**
  - `numpy`: Numerical tensor and array operations
  - `scipy`: Morphological distance transforms and statistical calculations
  - `scikit-image` (`skimage`): Medial axis skeletonization, contours, moments
  - `opencv-python` (`cv2`): CLAHE, image resizing, bilateral filtering
  - `scikit-learn`: Scalers, StratifiedKFold, logistic regression, classifiers, ROC-AUC
  - `pyradiomics`: High-dimensional IBSI-compliant radiomic extraction
  - `pandas`: Tabular feature management and merge pipelines

---

## 2. Model Checkpoints and Verification Hashes

All pretrained neural network checkpoints are archived locally and must match the following SHA-256 signatures:

1. **WGAN-GP Generator Checkpoint:**
   - Path: `24_gan_generation/models_wgan_gp/generator_wgan_gp_128.pth`
   - File Size: `14,446,412 bytes` (~13.78 MB)
   - SHA-256 Hash: `e0665ea5aebbcb8f8ad8f4c55a9ba0335d721a4b5aaaead63bfd791dd8b94b07`
   - Architecture: 5-layer transposed convolutional generator taking 100-dimensional Gaussian noise latent vector $z \sim \mathcal{N}(0, 1)$ to produce $128 \times 128$ grayscale images.

2. **Corpus Callosum U-Net Checkpoint:**
   - Path: `unet_cc/best_model.pth`
   - File Size: `97,895,143 bytes` (~93.36 MB)
   - SHA-256 Hash: `3ab59016cc2c17149fa410c8a0464eb7e4efcfc61f86546913062bde12ea8124`
   - Architecture: 2D U-Net encoder-decoder with skip connections, trained on paired sagittal MRI patches and manual CC annotations.

---

## 3. Authoritative Dataset Hashes

1. **Cleaned Real Clinical Master Dataset:**
   - Path: `22_statistical_analysis/cleaned_master_dataset.csv`
   - Dimensions: 192 rows $\times$ 388 columns
   - SHA-256 Hash: `42302fc8afc589c292ccc7e3805365ba1e9ca2a159f97d32e6d6fb4a3822ebb8`

2. **Historical Augmented Master Dataset:**
   - Path: `22_statistical_analysis/final_statistical_dataset.csv`
   - Dimensions: 202 rows $\times$ 414 columns
   - SHA-256 Hash: `1c84db777a9b15c940a9437de020a58d9dec9dafbe28ff73274780913da26b78`

3. **Latest Merged Synthetic Feature Dataset:**
   - Path: `33_synthetic_delayed_150/08_merged_features/synthetic_delayed_150_master_features.csv`
   - Dimensions: 133 rows $\times$ 408 columns
   - SHA-256 Hash: `254d6231de773d0701d6c47f76693c3b57aaa7c86363ff496cec99851ef30364`

---

## 4. End-to-End Reproduction Instructions

To reproduce the entire `33_synthetic_delayed_150` pipeline sequentially, execute the following commands from the project root (`Corpus_Callosum`):

### Step 1: Candidate Generation & Hash Verification
Generates 200 candidate synthetic $128 \times 128$ images from seeds 1000–1199:
```powershell
python 33_synthetic_delayed_150/step01_generate_verified.py
```
*Expected Output:* 200 PNG files in `33_synthetic_delayed_150/candidates/`.

### Step 2: MRI Preprocessing
Applies intensity normalization, bilateral filtering, and CLAHE:
```powershell
python 33_synthetic_delayed_150/step02_preprocess_runner.py
```
*Expected Output:* 200 preprocessed PNGs in `33_synthetic_delayed_150/02_preprocessed/`.

### Step 3: U-Net Mid-Sagittal Callosal Localization
Runs U-Net segmentation model with probability threshold $\tau = 0.5$:
```powershell
python 33_synthetic_delayed_150/step03b_localization_runner.py
```
*Expected Output:* 151 non-empty and 49 empty probability maps in `33_synthetic_delayed_150/03b_unet_localization/`.

### Step 4: Morphological Mask Refinement
Removes isolated artifact components $<6$ pixels and smooths boundaries:
```powershell
python 33_synthetic_delayed_150/step04_refinement_runner.py
```
*Expected Output:* 133 valid non-empty masks and 67 cumulative empty masks in `33_synthetic_delayed_150/04_refined_masks/`.

### Step 5: Morphological & Thickness Feature Extraction
Extracts geometric area, perimeter, skeleton length, and distance-transform thickness:
```powershell
python 33_synthetic_delayed_150/step05_morphology_runner.py
```
*Expected Output:* 133 rows $\times$ 12 columns in `33_synthetic_delayed_150/05_morphology/measurements/features.csv`.

### Step 6: Texture Feature Extraction
Computes intensity histogram metrics and GLCM gray-level co-occurrence descriptors:
```powershell
python 33_synthetic_delayed_150/step06_texture_runner.py
```
*Expected Output:* 133 rows $\times$ 10 columns in `33_synthetic_delayed_150/06_texture/texture_features.csv`.

### Step 7: Radiomics Extraction
Extracts IBSI-compliant shape, first-order, GLRLM, GLSZM, and wavelet features via PyRadiomics:
```powershell
python 33_synthetic_delayed_150/step07_radiomics_runner.py
```
*Expected Output:* 133 rows $\times$ 388 columns in `33_synthetic_delayed_150/07_radiomics/radiomic_features.csv`.

### Step 8: Master Feature Merging
Merges morphological, textural, and radiomic tables, strictly dropping candidate preview contact sheets:
```powershell
python 33_synthetic_delayed_150/step08_merge_runner.py
```
*Expected Output:* 133 rows $\times$ 408 columns in `33_synthetic_delayed_150/08_merged_features/synthetic_delayed_150_master_features.csv`.

### Step 9: Rigorous Clinical Evaluation
Runs 2-Fold Stratified Cross-Validation (10 repeats) comparing Real Baseline vs. Real+Synthetic Augmented:
```powershell
python 33_synthetic_delayed_150/step09_statistical_evaluation.py
```
*Expected Output:* Evaluation metrics table in `33_synthetic_delayed_150/09_statistical_evaluation/rigorous_clinical_evaluation.csv`.

### Step 10: Export Audit Artifacts
Exports verified confusion matrices and dataset provenance tables:
```powershell
python 33_synthetic_delayed_150/step10_export_audit_artifacts.py
```
*Expected Output:* CSV tables in `33_synthetic_delayed_150/audit_results/`.

---

## 5. Potential Pitfalls and Non-Reproducibility Factors
- **CPU SIMD Rounding Variance:** 45 of 200 generated candidate images differ by $\le 1$ intensity unit between different CPU hardware architectures (e.g., AVX2 vs AVX-512 BLAS implementations). This causes minor bitwise SHA-256 mismatches but does not alter downstream extracted feature statistics.
- **Path Portability:** All runner scripts in `33_synthetic_delayed_150` utilize relative directory resolution via `pathlib.Path(__file__).resolve()`, preventing broken absolute path dependencies.
