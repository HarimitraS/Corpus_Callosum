# High-Precision Corpus Callosum Extraction, Morphometric Analysis, and Delayed-Myelination Evaluation from Infant Brain MRI

## Overview

This repository contains an advanced biomedical image-processing and machine-learning research project investigating the automatic segmentation, quantitative morphometric characterization, and clinical classification of the **Corpus Callosum (CC)** from infant brain MRI, with a specific focus on **delayed myelination**.

The project integrates classical medical image processing (adaptive enhancement, atlas propagation, distance transforms), deep learning segmentation (U-Net with ResNet encoder), generative deep learning data augmentation (WGAN-GP), high-throughput radiomics (PyRadiomics), and rigorous statistical evaluation under extreme clinical class imbalance.

---

## 🎯 Research Objectives & Problem Statement

1. **Automated Mid-Sagittal Callosal Segmentation:** Delineate the infant corpus callosum objectively from mid-sagittal 2D brain MRI slices to replace error-prone manual tracing.
2. **Quantitative Multi-Scale Phenotyping:** Extract geometric, thickness profile, GLCM textural, and IBSI-standardized radiomic descriptors (408 features) of callosal maturation.
3. **Generative Data Augmentation Investigation:** Evaluate whether synthetic delayed-myelination MRI candidates synthesized by a Wasserstein GAN with Gradient Penalty (WGAN-GP) can rebalance training cohorts and improve clinical detection on held-out real patient cohorts.
4. **Rigorous Scientific Audit:** Forensically evaluate data leakage risks, establish true dataset provenance, benchmark against majority-class baselines, and assess the feasibility of supervised classification under extreme class imbalance.

---

## 🔬 Complete Research & Pipeline Architecture

```
                       [ Infant Brain MRI Cohort ]
                                    │
               ┌────────────────────┴────────────────────┐
               ▼                                         ▼
   [ Real Clinical Cohort (N=192) ]          [ WGAN-GP Generator (128x128) ]
   (190 Normal, 2 Delayed Myelination)                   │
               │                             (200 Synthetic Delayed Candidates)
               │                                         │
               │                                         ▼
               │                           [ Stage 2: MRI Preprocessing ]
               │                           (CLAHE, Bilateral Filter, Normalization)
               │                                         │
               │                                         ▼
               │                           [ Stage 3: U-Net Localization ]
               │                           (151 Non-Empty, 49 Empty Maps Excluded)
               │                                         │
               │                                         ▼
               │                           [ Stage 4: Mask Refinement ]
               │                           (133 Valid Masks, 18 Faint Artifacts Excluded)
               │                                         │
               ▼                                         ▼
   [ Quantitative Feature Extraction: Morphology + GLCM Texture + PyRadiomics ]
   (12 Geometric / Thickness Features + 10 Texture Features + 388 Radiomic Features)
               │                                         │
               ▼                                         ▼
     [ Real Feature Dataset ]                  [ Synthetic Feature Dataset ]
     (192 rows × 388 cols)                     (133 rows × 408 cols, 0 NaNs)
               │                                         │
               └────────────────────┬────────────────────┘
                                    │
                                    ▼
       [ Stage 9: Rigorous Stratified Cross-Validation (2-Fold, 10 Repeats) ]
               ├── Baseline Setting:  Train Real -> Test Real
               └── Augmented Setting: Train (Real + Synthetic) -> Test Real
                                    │
                                    ▼
       [ Stage 10: Forensic Audit, Verification & Viva Defense Documentation ]
```

---

## 📊 Key Research Findings & Empirical Audit

All reported results have been forensically verified against on-disk artifacts in [`33_synthetic_delayed_150/11_final_research_documentation/`](33_synthetic_delayed_150/11_final_research_documentation):

1. **True Clinical Dataset Composition:**
   - The real clinical cohort contains **192 subjects**: **190 normal myelination** and **2 delayed myelination** (`s0147` and `s0160`), reflecting a **1.04%** clinical prevalence.
   - The historical "202-row" dataset (`final_statistical_dataset.csv`) is an **augmented composite dataset** (192 real + 10 synthetic delayed records from earlier GAN experiments); it does not represent 202 real clinical subjects.
2. **Clinical Classification Performance:**
   - At standard decision thresholds ($\tau = 0.5$), **all evaluated classifiers achieved 0.0000 recall** ($\text{TP}=0, \text{FN}=2$) across Logistic Regression, KNN, Random Forest, and Gradient Boosting.
   - A trivial **Majority-Class Baseline** predicting "normal" for all cases achieves **98.96% accuracy**. High model accuracies (95.8% to 98.9%) are entirely attributable to majority-class dominance.
3. **Synthetic Augmentation Impact:**
   - Logistic Regression exhibited an apparent ROC-AUC increase from 0.8579 to 0.9316, but inspection shows this reflects relative ranking (delayed case predicted at 0.96% probability vs. normal cases at $<0.1\%$), which remains far below clinical detection cutoffs.
   - In ensemble tree models (Random Forest and Gradient Boosting), synthetic data augmentation decreased overall accuracy (98.96% $\rightarrow$ 96.88% and 96.35%) due to false positive induction without recovering true positives.
4. **Methodological Constraints Identified:**
   - **Extreme Sample Scarcity:** $N=2$ real delayed cases is statistically insufficient to train robust supervised classifiers.
   - **Feature Selection Leakage:** Upstream feature selection across the combined cohort before cross-validation fold splitting.
   - **Generative Representation Leakage:** Whole-cohort training of WGAN-GP without test-fold isolation.
   - **Anatomical Validation:** Synthetic non-empty masks require independent neuroradiologist verification.

---

## 📁 Repository Structure

```
Corpus_Callosum/
├── 4_sagittal_slices/               # 205 raw infant mid-sagittal MRI slices
├── 5_roi/                           # Region of interest crops
├── 6_cc_extraction/                 # Callosal patch extractions
├── 7_prior_masks/                   # Probabilistic prior masks
├── 10_medsam_masks/                 # MedSAM initial segmentations
├── 11_unet_masks/                   # Baseline U-Net segmentation masks
├── 12_refined_masks/                # Morphologically refined masks
├── 14_skeletons/                    # Medial axis skeletons
├── 15_thickness_maps/               # Euclidean distance transform thickness maps
├── 16_measurements/                 # Extracted morphological measurement tables
├── 19_segmentation_evaluation/      # Segmentation evaluation metrics (Dice, IoU)
├── 20_texture_analysis/             # GLCM texture analysis features
├── 21_radiomic_features/            # PyRadiomics feature tables
├── 22_statistical_analysis/         # Clinical master datasets & historical evaluation
│   ├── cleaned_master_dataset.csv   # True real cohort (192 subjects: 190 normal, 2 delayed)
│   └── final_statistical_dataset.csv# Historical augmented dataset (192 real + 10 GAN)
├── 24_gan_generation/               # WGAN-GP training pipeline and generator weights
│   └── models_wgan_gp/              # Trained generator checkpoint (generator_wgan_gp_128.pth)
├── 31_gan_delayed_pipeline/         # Historical 10-candidate synthetic delayed workflow
├── 32_augmented_dataset/            # Historical dataset augmentation scripts
├── 33_synthetic_delayed_150/        # PRIMARY RESEARCH WORKFLOW (Latest 200-Candidate Evaluation)
│   ├── candidates/                  # 200 synthetic candidate images (128x128 PNG)
│   ├── 02_preprocessed/             # 200 preprocessed synthetic candidates
│   ├── 03b_unet_localization/       # 151 non-empty, 49 empty probability maps
│   ├── 04_refined_masks/            # 133 valid non-empty refined masks (67 empty excluded)
│   ├── 05_morphology/               # Morphology & thickness features (133 rows x 12 cols)
│   ├── 06_texture/                  # GLCM texture features (133 rows x 10 cols)
│   ├── 07_radiomics/                # PyRadiomics features (133 rows x 388 cols)
│   ├── 08_merged_features/          # Master feature dataset (133 rows x 408 cols, 0 NaNs)
│   ├── 09_statistical_evaluation/   # Cross-validation results and evaluation metrics
│   ├── audit_results/               # Empirical audit tables and provenance artifacts
│   └── 11_final_research_documentation/ # COMPLETE MASTER REPORT & VIVA DEFENSE GUIDE
│       ├── FINAL_RESEARCH_REPORT.md # Academic master research report (36 sections)
│       ├── EXECUTIVE_SUMMARY.md     # High-level project summary
│       ├── PROJECT_INVENTORY.csv    # Audited catalog of 65 components
│       ├── DATASET_PROVENANCE.csv   # Lineage of all clinical and synthetic datasets
│       ├── FEATURE_DICTIONARY.csv   # Definitions of all 408 extracted features
│       ├── CLAIM_VERIFICATION.csv   # 18 major empirical claims rigorously verified
│       ├── RESULTS_SUMMARY.csv      # Metrics, ROC-AUC, and confusion matrix cells
│       ├── REPRODUCIBILITY_CHECKLIST.md # Environment specifications and seeds
│       ├── LIMITATIONS_AND_FUTURE_WORK.md # Prioritized 3-tier future work roadmap
│       ├── VIVA_QUESTIONS_AND_ANSWERS.md # 63 comprehensive technical viva Q&As
│       └── REPORT_GENERATION_LOG.md # Complete audit trace
├── ground_truth_masks/              # Manual expert ground truth annotations
├── unet_cc/                         # Trained U-Net model checkpoint (best_model.pth)
└── README.md                        # Project documentation (this file)
```

---

## 🛠️ Reproduction & Execution

All steps can be reproduced from the repository root using Python 3.11 with PyTorch and CUDA:

```powershell
# 1. Candidate Generation (200 candidates from seeds 1000-1199)
python 33_synthetic_delayed_150/step01_generate_verified.py

# 2. Preprocessing (CLAHE, bilateral filtering, intensity normalization)
python 33_synthetic_delayed_150/step02_preprocess_runner.py

# 3. Mid-Sagittal U-Net Localization
python 33_synthetic_delayed_150/step03b_localization_runner.py

# 4. Morphological Mask Refinement & Component Filtering
python 33_synthetic_delayed_150/step04_refinement_runner.py

# 5. Morphology & Thickness Feature Extraction
python 33_synthetic_delayed_150/step05_morphology_runner.py

# 6. GLCM Texture Analysis
python 33_synthetic_delayed_150/step06_texture_runner.py

# 7. PyRadiomics Extraction (388 features)
python 33_synthetic_delayed_150/step07_radiomics_runner.py

# 8. Master Feature Merging (408 columns)
python 33_synthetic_delayed_150/step08_merge_runner.py

# 9. Rigorous Cross-Validation Classification
python 33_synthetic_delayed_150/step09_statistical_evaluation.py

# 10. Export Empirical Audit Tables
python 33_synthetic_delayed_150/step10_export_audit_artifacts.py
```

Refer to [`REPRODUCIBILITY_CHECKLIST.md`](33_synthetic_delayed_150/11_final_research_documentation/REPRODUCIBILITY_CHECKLIST.md) for checkpoint hashes, random seeds, and dependency specifications.

---

## 📚 Primary Documentation & Viva Defense

For detailed academic analysis, supervisor reviews, and viva defense preparation, refer to the documents in [`33_synthetic_delayed_150/11_final_research_documentation/`](33_synthetic_delayed_150/11_final_research_documentation/):
- **Master Report:** [`FINAL_RESEARCH_REPORT.md`](33_synthetic_delayed_150/11_final_research_documentation/FINAL_RESEARCH_REPORT.md)
- **Viva Defense Guide:** [`VIVA_QUESTIONS_AND_ANSWERS.md`](33_synthetic_delayed_150/11_final_research_documentation/VIVA_QUESTIONS_AND_ANSWERS.md) (63 detailed questions and answers covering anatomy, models, metrics, data leakage, and clinical limitations)
- **Limitations & Future Work:** [`LIMITATIONS_AND_FUTURE_WORK.md`](33_synthetic_delayed_150/11_final_research_documentation/LIMITATIONS_AND_FUTURE_WORK.md)

---

## ⚖️ Scientific Disclaimer

This repository represents an academic research project. The computational models, synthetic images, and extracted features are developed for research and educational investigation and are not intended for direct clinical diagnostic use without extensive external clinical validation and regulatory clearance.
