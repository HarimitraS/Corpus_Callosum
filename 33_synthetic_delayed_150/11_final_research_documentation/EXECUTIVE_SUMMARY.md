# EXECUTIVE SUMMARY
## Research Audit and Technical Assessment of the Corpus Callosum Infant MRI Pipeline

### 1. Project Background and Objective
This project investigates the automated analysis of the infant corpus callosum (CC) on 2D mid-sagittal brain MRI scans to evaluate morphological, textural, and radiomic alterations associated with delayed myelination. Due to extreme clinical class imbalance in the available hospital cohort (190 normal myelination subjects vs. only 2 delayed myelination subjects), generative adversarial networks (WGAN-GP) were introduced to synthesize delayed-myelination MRIs and assess whether synthetic data augmentation improves classifier detection performance.

### 2. Pipeline Execution & Verification Status
The latest workflow (33_synthetic_delayed_150) successfully executed an end-to-end 10-stage processing pipeline:
1. **Candidate Generation:** 200 synthetic candidates generated using a trained WGAN-GP generator (generator_wgan_gp_128.pth).
2. **Preprocessing:** 200 candidates preprocessed via intensity normalization and contrast enhancement.
3. **U-Net Localization:** 151 candidates produced non-empty callosal probability maps; 49 candidates failed initial localization ($<1$ pixel at probability $\ge 0.5$).
4. **Morphological Mask Refinement:** 133 candidates yielded valid non-empty refined masks; 18 faint artifact clusters ($<6$ pixels) were eliminated, resulting in 67 total empty masks (33.5% cumulative failure rate).
5. **Feature Extraction:** 12 morphological, 10 textural, and 388 PyRadiomics features were extracted across all 133 valid masks with zero missing values or NaN/Inf corruptions.
6. **Feature Merging:** A master feature table of 133 rows $	imes$ 408 columns was constructed without preview grid contaminations.
7. **Clinical Cross-Validation:** 2-Fold Stratified Cross-Validation (10 repeats) was conducted on the 192 real clinical subjects, with synthetic records injected strictly into training folds during augmented runs.

### 3. Core Scientific and Clinical Findings
- **Zero Real Delayed Detection:** All evaluated classifiers (Logistic Regression, KNN, Random Forest, and Gradient Boosting) exhibited **0.0000 recall** for real delayed myelination subjects at standard decision thresholds ($	ext{TP}=0, 	ext{FN}=2$).
- **The Majority-Class Baseline:** A trivial baseline predicting \'normal\' for all cases achieves **98.96% accuracy**. High model accuracies (95.8% to 98.9%) are entirely attributable to majority-class dominance rather than genuine diagnostic detection.
- **ROC-AUC vs. Decision Thresholds:** Logistic Regression exhibited an apparent ROC-AUC increase from 0.8579 to 0.9316. However, inspection of probability distributions confirms this is driven by relative ranking (delayed probability of ~0.0096 vs normal probabilities < 0.0010), which remains far below clinical detection thresholds.
- **Augmentation Degradation in Non-Linear Models:** Synthetic data augmentation decreased accuracy in Random Forest (98.96% to 96.88%) and Gradient Boosting (98.96% to 96.35%) due to false positive induction without recovering true positives.

### 4. Critical Methodological Constraints
- **Extreme Class Imbalance:** The actual real cohort contains only 2 delayed subjects (.04\%$ prevalence). The effective clinical sample size (=2$) is statistically insufficient to train robust supervised classifiers.
- **Feature Selection Leakage:** Top 20 features were selected globally across the full cohort prior to cross-validation partitioning.
- **Generative Representation Leakage:** The WGAN-GP model was trained on all 205 available sagittal MRI scans without subject-level test fold holdouts. Synthetic images are therefore not strictly independent of test subjects.
- **Absence of Expert Clinical Validation:** Non-empty synthetic masks frequently represent localized hyperintense blobs rather than true anatomical corpus callosum structures. No blinded radiologist scoring or ground-truth segmentations exist for synthetic samples.

### 5. Research Conclusions
The computational pipeline is fully operational, verified, and technically reproducible. However, synthetic data augmentation does not establish improved clinical detection of delayed myelination. Future research must prioritize acquiring multi-center clinical cohorts with verified delayed myelination and implementing nested cross-validation protocols.
