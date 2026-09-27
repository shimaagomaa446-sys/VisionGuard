# Data Quality Report — VisionGuard

**Project:** AI Safety Monitoring System for Construction Sites
**Date:** [21/9/2026]
**Author:** [ShimaaGomaa]

---

## 1. Dataset Overview

| Property | Value |
|----------|-------|
| Name | Construction Site Safety Image Dataset |
| Source | Roboflow (v28) |
| License | CC BY 4.0 |
| Total Images | 2801 |
| Format | YOLO |
| Classes | 10 (will reduce to 5) |

---

## 2. Dataset Structure

| Split | Images | Labels |
|-------|-------:|-------:|
| Train | 2605 | 2605 |
| Valid | 114 | 114 |
| Test | 82 | 82 |
| **Total** | **2801** | **2801** |

### Observations:
- Train represents ~93% of the dataset
- Valid and Test are significantly smaller (~4% and ~3%)
- Each image has exactly one label file

---

## 3. Class Distribution

| ID | Class Name | Train | Valid | Test | Train % |
|:--:|------------|------:|------:|-----:|--------:|
| 5 | Person | 9,532 | 166 | 174 | 26.5% |
| 8 | machinery | 5,247 | 55 | 44 | 14.6% |
| 4 | NO-Safety Vest | 3,962 | 106 | 90 | 11.0% |
| 6 | Safety Cone | 3,366 | 44 | 92 | 9.4% |
| 0 | Hardhat | 3,145 | 79 | 110 | 8.8% |
| 3 | NO-Mask | 3,097 | 74 | 79 | 8.6% |
| 7 | Safety Vest | 3,033 | 41 | 61 | 8.4% |
| 2 | NO-Hardhat | 2,317 | 69 | 41 | 6.4% |
| 1 | Mask | 1,651 | 21 | 28 | 4.6% |
| 9 | vehicle | 1,545 | 42 | 41 | 4.3% |
| **Total** | | **35,895** | **697** | **760** | 100% |

### Analysis:

**Class Imbalance:**
- Person is the largest class (26.5%)
- vehicle is the smallest class (4.3%)
- Ratio: ~6:1 between largest and smallest

**PPE Balance:**
- Hardhat (3,145) vs NO-Hardhat (2,317) → ratio 1.36:1
- Safety Vest (3,033) vs NO-Safety Vest (3,962) → ratio 0.77:1
- **Note:** NO-Safety Vest is more common than Safety Vest

---

## 4. Leakage Check

### Methodology:
- Extracted filename prefix (part before `.rf.`)
- Used set intersection between train and test prefixes

### Results:

| Metric | Value |
|--------|------:|
| Test prefixes | 81 |
| Train prefixes | 514 |
| Common prefixes | 2 |
| **Leakage %** | **2.47%** |

### Interpretation:
- **Low leakage** (< 5%)
- Train and test splits are safe
- No significant duplicate filenames

---

## 5. Empty Label Files

| Split | Total | Empty | % |
|-------|------:|------:|--:|
| Train | 2605 | 6 | 0.23% |
| Valid | 114 | 10 | 8.77% |
| Test | 82 | 8 | 9.76% |
| **Total** | **2801** | **24** | **0.86%** |

### Visual Inspection:
- Inspected 6 files in train
- Corresponding images were **book images** (out-of-domain)

### Decision:
- **Keep empty files as-is** (negative samples)
- **Reason:** Small percentage (0.86% total)
- **Documented** as a known issue

---

## 6. Image Sizes

| Split | Total Images | Size |
|-------|-------------:|:----:|
| Train | 2605 | 640×640 |
| Valid | 114 | 640×640 |
| Test | 82 | 640×640 |
| **All** | **2801** | **640×640** |

### Notes:
- All images were resized by Roboflow
- No size variations
- Consistent input for YOLO

---
## 7. Data Augmentation (from Roboflow)

According to the Roboflow README:
- 5 augmented versions per source image
- Augmentations applied:
  - Horizontal flip (50%)
  - Random crop (0-20%)
  - Random rotation (-12° to +12°)
  - Random shear (±2°)
  - Brightness (±25%)
  - Exposure (±20%)
  - Gaussian blur (0-0.5 px)

**Implication:** This explains why train has 514 prefixes for 2605 images.


## 8. Issues Identified

| # | Issue | Severity |
|---|-------|:--------:|
| 1 | Class imbalance (Person 6x vehicle) | Medium |
| 2 | Small valid/test splits (~4% and ~3%) | Medium |
| 3 | Empty label files (24 total, 0.86%) | Low |
| 4 | Out-of-domain images (book images) | Low |

---

## 9. Decisions Made

| # | Decision | Reason |
|---|----------|--------|
| 1 | Keep empty files | Negative samples; low percentage |
| 2 | Reduce classes to 5 | Project requirements |
| 3 | Use existing splits | Reasonable leakage check |
| 4 | Document limitations | Rubric requirement |

---

## 10. Limitations

- **Small valid/test** → less statistically significant evaluation
- **Leakage check** based on filenames only (not visual similarity)
- **No full visual inspection** of all images
- **Out-of-domain images** exist in the dataset

---

## Baseline Results (Zero-shot)

### Model: YOLOv8n (pre-trained on COCO)

### Test Results:
| Metric | Value |
|--------|------:|
| mAP50 | 0.00253 |
| mAP50-95 | 0.000709 |
| Precision | 0.0243 |
| Recall | 0.0705 |

### Interpretation:
- The pre-trained YOLOv8n performs **extremely poorly** on our dataset
- Reason: The model is trained on COCO (80 classes) and has **no knowledge of PPE**
- The model attempts to match our 5 classes with COCO classes → wrong predictions
- **This confirms the need for fine-tuning**

### Note:
- 22 out of 82 test images are backgrounds (no objects)
- Total instances: 476

**End of Report**