# FINAL RESEARCH VALIDATION AND SCIENTIFIC AUDIT REPORT
**Corpus Callosum Infant Brain MRI Segmentation & Myelination Analysis**  
**Directory:** `33_synthetic_delayed_150`  
**Date:** October 13, 2026  
**Auditor:** Antigravity Advanced Agentic System  
**Document Status:** Verified, Scientifically Defensible, Audit Complete  

---

## 1. Audit Scope & Files Inspected

This audit was conducted strictly non-destructively to resolve empirical discrepancies, audit statistical claims, evaluate potential data leakages, establish dataset provenance, and produce a scientifically rigorous validation report without modifying existing historical repositories (`31_gan_delayed_pipeline`, `22_statistical_analysis`, `dataset`, `unet_cc`).

### Primary Files Inspected & Verified:
- **Statistical Pipeline:** `33_synthetic_delayed_150/step09_statistical_evaluation.py`
- **Statistical Results CSV:** `33_synthetic_delayed_150/09_statistical_evaluation/rigorous_clinical_evaluation.csv`
- **Clinical Dataset Tables:**
  - `22_statistical_analysis/cleaned_master_dataset.csv` (192 rows $\times$ 388 cols)
  - `22_statistical_analysis/master_dataset.csv` (192 rows $\times$ 414 cols)
  - `22_statistical_analysis/final_statistical_dataset.csv` (202 rows $\times$ 414 cols)
  - `31_gan_delayed_pipeline/10_final_dataset/final_dataset_with_gan.csv` (202 rows $\times$ 414 cols)
- **Feature Selection & Classification Baselines:**
  - `22_statistical_analysis/feature_selection/selected_feature_dataset.csv`
  - `22_statistical_analysis/classification_results/gan_augmentation_comparison.csv`
  - `22_statistical_analysis/final_results/research_conclusions.txt`
- **GAN Training & Image Extraction:**
  - `24_gan_generation/train_wgan_gp.py`
  - `24_gan_generation/dataset_loader.py`
  - `24_gan_generation/extract_delayed_images.py`
  - `24_gan_generation/models_wgan_gp/generator_wgan_gp_128.pth`
- **Synthetic Candidate Metadata & Hashes:**
  - `33_synthetic_delayed_150/metadata/generation_metadata.csv` (200 rows)
  - `33_synthetic_delayed_150/candidates/` (200 candidate images)
- **Segmentation & Feature Tables:**
  - `33_synthetic_delayed_150/02_preprocessed/` (200 images)
  - `33_synthetic_delayed_150/03b_unet_localization/unet_localization_results.csv` (200 rows)
  - `33_synthetic_delayed_150/04_refined_masks/refinement_summary.csv` (200 rows)
  - `33_synthetic_delayed_150/05_morphology/measurements/features.csv` (133 rows $\times$ 12 cols)
  - `33_synthetic_delayed_150/06_texture/texture_features.csv` (133 rows $\times$ 10 cols)
  - `33_synthetic_delayed_150/07_radiomics/radiomic_features.csv` (133 rows $\times$ 388 cols)
  - `33_synthetic_delayed_150/08_merged_features/synthetic_delayed_150_master_features.csv` (133 rows $\times$ 408 cols)

---

## 2. Pipeline Completion & Stage Count Verification

Every processing count was empirically re-verified directly against the disk assets:

| Pipeline Stage | Expected Count | Verified Actual Count | Non-Empty / Valid | Failures / Skipped | Verified Output Artifact / Directory |
|---|---|---|---|---|---|
| **1. Candidate Generation** | 200 candidates | 200 PNG files | 200 valid | 0 failed | `33_synthetic_delayed_150/candidates/` |
| **2. Preprocessing** | 200 images | 200 PNG files | 200 valid | 0 failed | `33_synthetic_delayed_150/02_preprocessed/` |
| **3. U-Net Localization** | 200 images | 200 probability maps | **151** non-empty | **49** empty ($<1\text{ px at }\ge 0.5$) | `33_synthetic_delayed_150/03b_unet_localization/` |
| **4. Mask Refinement** | 200 masks | 200 refined masks | **133** valid non-empty | **67** total empty (49 initial + 18 post-refine) | `33_synthetic_delayed_150/04_refined_masks/` |
| **5. Morphology** | 133 valid masks | 133 rows | 133 valid | 67 empty skipped | `05_morphology/measurements/features.csv` |
| **6. Texture Analysis** | 133 valid masks | 133 rows | 133 valid | 67 empty skipped | `06_texture/texture_features.csv` |
| **7. Radiomics** | 133 valid masks | 133 rows | 133 valid | 67 empty skipped | `07_radiomics/radiomic_features.csv` |
| **8. Feature Merging** | 133 subjects | 133 rows $\times$ 408 cols | 133 valid | 0 duplicate / 0 NaNs | `08_merged_features/synthetic_delayed_150_master_features.csv` |

All subject identifiers match identically (`candidate_000001` to `candidate_000200`) across morphology, texture, and radiomics tables. No preview grids or contact sheets were included.

---

## 3. Dataset Provenance & Resolution of Dataset-Size Discrepancy

### The Discrepancy
- `cleaned_master_dataset.csv` contains **192 rows** (190 normal, **2 delayed**).
- `final_statistical_dataset.csv` contains **202 rows** (190 normal, **12 delayed**).

### Provenance Audit Findings (Definitive Resolution)
Inspection of subject identifiers, diagnostic logs, and script pipelines demonstrates conclusively:
1. **The True Real Clinical Cohort Contains 192 Subjects:**
   - 190 subjects diagnosed as `normal` myelinisation.
   - **Only 2 subjects** diagnosed as `delayed` myelinisation: `s0147` (diagnosis: *Delayed myelinisation, cerebral atrophy*) and `s0160` (diagnosis: *Intracerebral hemorrhage*).
2. **The 10 "Extra" Delayed Subjects Are Synthetic Artifacts from an Earlier Run:**
   - In `final_statistical_dataset.csv`, the 10 additional delayed records are:
     `synthetic_mri_002`, `synthetic_mri_003`, `synthetic_mri_004`, `synthetic_mri_005`, `synthetic_mri_006`, `synthetic_mri_007`, `synthetic_mri_012`, `synthetic_mri_013`, `synthetic_mri_015`, `synthetic_mri_016`.
   - These 10 records were generated during the execution of `31_gan_delayed_pipeline/10_create_final_dataset.py`, where 10 synthetic delayed feature rows from `31_gan_delayed_pipeline/09_merged_features/gan_delayed_master_features.csv` were concatenated directly onto `cleaned_master_dataset.csv` ($192 + 10 = 202$).
   - In `final_statistical_dataset.csv`, these 10 records have missing metadata (`age`, `diagnosis`, `group` are all `NaN`).
3. **Scientific Implication:**
   - Any historical report describing `final_statistical_dataset.csv` as having "12 clinical delayed subjects" was inaccurate.
   - The true clinical ground truth in the repository contains **only 2 real delayed subjects** ($1.04\%$ prevalence).
   - The verified dataset provenance table is archived at:  
     [`33_synthetic_delayed_150/audit_results/dataset_provenance_audit.csv`](file:///c:/Users/kanishk/OneDrive/Desktop/infant/Corpus_Callosum/33_synthetic_delayed_150/audit_results/dataset_provenance_audit.csv).

---

## 4. Audit of Cross-Validation & Data Leakage Prevention

### Cross-Validation Protocol
- **Evaluation Design:** 2-Fold Stratified Cross-Validation on the 192 real clinical subjects (`cleaned_master_dataset.csv`), with 10 repetitions.
- **Split Breakdown:**
  - Fold 1: 95 normal real subjects, 1 delayed real subject (96 total).
  - Fold 2: 95 normal real subjects, 1 delayed real subject (96 total).
- **Synthetic Allocation:**
  - In the "Augmented" setting, the 133 validated synthetic delayed records (`synthetic_delayed_150_master_features.csv`) were injected **strictly into the training split**.
  - **The held-out test split contained 100% real clinical subjects** (zero synthetic test leakage).

### Critical Leakage Audit Findings
While synthetic sample leakage into test folds was strictly prevented in `step09_statistical_evaluation.py`, our scientific audit revealed two subtle systemic leakage channels in the broader project pipeline:

1. **Feature Selection Leakage (Upstream):**
   - The 20 features used in the evaluation were selected in `22_statistical_analysis/feature_selection.py` across the **entire 192/202 dataset** prior to cross-validation splitting. While `StandardScaler` was fit strictly on the training fold, feature selection was performed globally upstream.
2. **GAN Training Data Contamination (Severe Provable Leakage):**
   - Inspection of `24_gan_generation/dataset_loader.py` reveals that the WGAN-GP generator was trained on `4_sagittal_slices`, which comprises **all 205 real infant sagittal MRI slices**.
   - Because the GAN was trained on the entire real clinical cohort without fold partitioning, **the GAN learned representations from the exact same real subjects that later appear in the held-out test folds**.
   - Therefore, the synthetic delayed images generated by this GAN cannot be claimed as an independent external cohort, and evaluating on real held-out folds is technically subject to **generative representation leakage**.

---

## 5. Metric Reproduction, Confusion Matrices & The Majority-Class Baseline

### Majority-Class Baseline
In a real clinical cohort of 190 normal subjects and 2 delayed subjects:
- A trivial **Majority-Class Classifier** that unconditionally predicts "normal" for every subject achieves:
  - **Accuracy: 98.96%** ($\frac{190}{192}$)
  - **Precision: 0.00%**
  - **Recall / Sensitivity: 0.00%** (0 / 2 delayed detected)
  - **Specificity: 100.00%** (190 / 190 normal correctly identified)
  - **F1-Score: 0.00%**
  - **ROC-AUC: 0.5000**
  - **Confusion Matrix:** $\text{TN}=190, \text{FP}=0, \text{FN}=2, \text{TP}=0$.

### Exact Reproduction of Reported Model Metrics
We re-ran the exact evaluation protocol across all 4 models and confirmed the previously reported numbers down to four decimal places. Full confusion matrices and specificity values were extracted:

| Model | Setting | Accuracy | Recall | Specificity | F1 | ROC-AUC | Confusion Matrix ($\text{TN}, \text{FP}, \text{FN}, \text{TP}$) |
|---|---|---:|---:|---:|---:|---:|---|
| **Majority Baseline** | *Trivial Predictor* | **0.9896** | **0.0000** | **1.0000** | **0.0000** | **0.5000** | $[190, 0, 2, 0]$ |
| **Logistic Regression** | Baseline (Real $\rightarrow$ Real) | 0.9583 | 0.0000 | 0.9684 | 0.0000 | 0.8579 | $[184, 6, 2, 0]$ |
| **Logistic Regression** | Augmented (Real+Synth $\rightarrow$ Real) | 0.9844 | 0.0000 | 0.9947 | 0.0000 | **0.9316** | $[189, 1, 2, 0]$ |
| **KNN** | Baseline (Real $\rightarrow$ Real) | 0.9896 | 0.0000 | 1.0000 | 0.0000 | 0.4632 | $[190, 0, 2, 0]$ |
| **KNN** | Augmented (Real+Synth $\rightarrow$ Real) | 0.9844 | 0.0000 | 0.9947 | 0.0000 | 0.4632 | $[189, 1, 2, 0]$ |
| **Random Forest** | Baseline (Real $\rightarrow$ Real) | 0.9896 | 0.0000 | 1.0000 | 0.0000 | 0.8263 | $[190, 0, 2, 0]$ |
| **Random Forest** | Augmented (Real+Synth $\rightarrow$ Real) | 0.9688 | 0.0000 | 0.9789 | 0.0000 | 0.8316 | $[186, 4, 2, 0]$ |
| **Gradient Boosting** | Baseline (Real $\rightarrow$ Real) | 0.9896 | 0.0000 | 1.0000 | 0.0000 | 0.4895 | $[190, 0, 2, 0]$ |
| **Gradient Boosting** | Augmented (Real+Synth $\rightarrow$ Real) | 0.9635 | 0.0000 | 0.9737 | 0.0000 | 0.4763 | $[185, 5, 2, 0]$ |

Supporting CSV archived at:  
[`33_synthetic_delayed_150/audit_results/verified_evaluation_metrics_and_confusion_matrices.csv`](file:///c:/Users/kanishk/OneDrive/Desktop/infant/Corpus_Callosum/33_synthetic_delayed_150/audit_results/verified_evaluation_metrics_and_confusion_matrices.csv).

### Critical Scientific Insights on the Metrics:
1. **Zero Clinical Detection ($\text{Recall} = 0.0000$ across all models):**
   - **Not a single delayed subject was ever identified** by any classifier under default decision thresholds ($\text{TP} = 0, \text{FN} = 2$).
   - High reported accuracies (95.8% to 98.9%) are entirely artifacts of the majority class (predicting normal for almost everything).
   - In fact, adding synthetic data actually **reduced accuracy** for Random Forest (98.96% $\rightarrow$ 96.88%) and Gradient Boosting (98.96% $\rightarrow$ 96.35%) because the models generated false positives without successfully detecting either true positive.
2. **The ROC-AUC Illusion:**
   - Logistic Regression shows an apparent improvement in ROC-AUC from $0.8579$ to $0.9316$.
   - Inspection of raw predicted probabilities reveals why: the true delayed test subject was assigned a probability of $0.0096$ (0.96%), while most normal test subjects received probabilities of $0.0010$ or lower. Because $0.0096 > 0.0010$, ROC-AUC ranks the delayed subject above most normal subjects.
   - However, because $0.0096 \ll 0.5$, the subject is classified as normal, yielding zero recall.
   - **Conclusion:** It is scientifically invalid to claim that adding synthetic data improves clinical detection or classification performance.

---

## 6. Audit of the 45 Image Hash Mismatches

### Quantitative Difference Analysis
For the 45 images where the generated SHA256 did not match `generation_metadata.csv`:
- **Image Dimensions:** Exactly $128 \times 128$, single-channel grayscale (`uint8`).
- **Total Pixels per Image:** 16,384 pixels.
- **Maximum Absolute Pixel Difference:** **1 intensity unit** (e.g., pixel value 141 vs 142 on a $[0, 255]$ scale).
- **Mean Absolute Pixel Difference:** $< 0.000122$ across the image.
- **Number of Differing Pixels per Image:**
  - In 35 images: **exactly 1 pixel** out of 16,384 differed by 1 unit ($0.0061\%$ of image).
  - In 6 images: **exactly 2 pixels** out of 16,384 differed by 1 unit ($0.0122\%$ of image).
  - In 4 images: exactly identical pixel arrays where minor PNG compression metadata varied.
- **Cause:** Deterministic floating-point arithmetic on CPU SIMD / BLAS instructions (AVX-512 vs AVX2 rounding at the 0.5 boundary during tensor conversion $((fake + 1.0) / 2.0) \times 255.0$).
- **Conclusion:** The 45 images are structurally and numerically identical for all downstream morphological, texture, and radiomic purposes.

---

## 7. Audit of Segmentation Failure Rates & Anatomical Validity

### Failure Rate Accounting
- Total synthetic candidates evaluated: **200**
- Stage 3B U-Net Initial Failures ($<1\text{ px at }\ge 0.5$): **49** (24.5%)
- Stage 4 Refinement Failures (components $<6\text{ px}$ eliminated): **18** (9.0%)
- Total Empty Refined Masks: **67** (33.5% failure rate)
- Valid Non-Empty Refined Masks: **133** (66.5% yield)

### Anatomical Validity Disclaimer
- **A non-empty binary mask is NOT proof of an anatomically valid corpus callosum.**
- In synthetic WGAN-GP images, visual inspection shows that some non-empty masks represent localized hyperintense blobs rather than a true crescent-shaped corpus callosum with distinct rostrum, genu, body, and splenium.
- Without expert radiologist ground-truth annotation for the synthetic set, downstream morphology and thickness features extracted from these masks reflect geometric properties of model predictions, not validated human anatomy.

---

## 8. Defensible Scientific Conclusions & Remaining Limitations

1. **Extreme Class Imbalance is the Primary Bottleneck:**
   - With only 2 real delayed clinical subjects in the entire dataset, any supervised machine-learning model will inevitably default to majority-class prediction.
   - The effective clinical sample size for delayed myelination is $N=2$. Statistical significance testing and cross-validation under $N=2$ are inherently unstable.
2. **Synthetic Data Did Not Solve the Clinical Task:**
   - Synthetic data generation provided an operational dataset for pipeline development, feature extraction, and linear score calibration.
   - However, synthetic augmentation **did not achieve clinical delayed detection** (recall remained 0.0 across all models).
3. **Generative Model Leakage:**
   - Because the WGAN-GP generator was trained on all available sagittal MRI scans without holding out test folds, synthetic candidates cannot be considered an independent validation cohort.
4. **Distinguishability:**
   - All synthetic records are maintained strictly identifiable (`data_type = 'synthetic'`) and isolated from clinical subjects.

---

## 9. Verification & Audit Sign-Off

- **Repository Preservation:** Confirmed. Zero existing files were modified or overwritten.
- **Reproducibility:** Confirmed. All metrics, hashes, and counts were reproduced directly from script execution.
- **New Validation Directory:** `33_synthetic_delayed_150/audit_results/` contains all audit tables.
- **Primary Validation Report:** [`33_synthetic_delayed_150/FINAL_RESEARCH_VALIDATION_REPORT.md`](file:///c:/Users/kanishk/OneDrive/Desktop/infant/Corpus_Callosum/33_synthetic_delayed_150/FINAL_RESEARCH_VALIDATION_REPORT.md)
