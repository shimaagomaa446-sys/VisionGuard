## Experiment 1: YOLOv8n - 25 epochs

### Configuration:
- Model: YOLOv8n
- Epochs: 25
- Image size: 640
- Batch: 16

### Results (on valid):
| Metric | Value |
|--------|------:|
| mAP50 | 0.773 |
| mAP50-95 | 0.437 |
| Precision | 0.831 |
| Recall | 0.719 |

### Per-class (mAP50):
- Safety Vest: 0.853
- Hardhat: 0.851
- Person: 0.807
- NO-Safety Vest: 0.737
- NO-Hardhat: 0.618 (weakest)

### Notes:
- Huge improvement over baseline (× 306)
- NO-Hardhat and NO-Safety Vest are the weakest classes

# Final Model Results

## Model: YOLOv8s (Experiment 3)
**Path:** `runs/detect/yolov8n_exp3/weights/best.pt`
**Size:** 22.5 MB
**Parameters:** 11.1M

## Test Set Results

| Metric | Value |
|--------|------:|
| mAP50 | 0.811 |
| mAP50-95 | 0.517 |
| Precision | 0.903 |
| Recall | 0.777 |

## Per-Class Performance (Test)

| Class | P | R | mAP50 | mAP50-95 |
|-------|--:|--:|------:|---------:|
| Hardhat | 0.987 | 0.891 | 0.941 | 0.602 |
| Safety Vest | 0.913 | 0.857 | 0.908 | 0.642 |
| Person | 0.890 | 0.799 | 0.861 | 0.555 |
| NO-Safety Vest | 0.931 | 0.750 | 0.796 | 0.487 |
| NO-Hardhat | 0.792 | 0.585 | 0.547 | 0.300 |

## Key Observations

### Strengths:
- Hardhat detection is excellent (mAP50 = 0.941)
- Safety Vest detection is excellent (mAP50 = 0.908)
- High precision (0.903) — few false positives

### Weaknesses:
- **NO-Hardhat is the weakest class (mAP50 = 0.547)**
  - Recall is only 0.585 → 41% of violations are missed
  - This is critical for the project's goal (violation detection)
  - Likely causes:
    - Small number of test instances (41)
    - Visual similarity with Person class
    - Class imbalance in training data

### Comparison: Valid vs Test
- Small drop in mAP50 (0.830 → 0.811)
- Small increase in recall (0.752 → 0.777)
- **Model generalizes well**

## Conclusion

The YOLOv8s model achieves strong overall performance (mAP50 = 0.811), 
but NO-Hardhat detection remains a challenge. This is an important 
limitation to address in the Compliance Logic and Error Analysis phases.

## Threshold Analysis

### Methodology:
Tested 5 confidence thresholds on the test set: 0.1, 0.25, 0.4, 0.5, 0.7

### Results:

| Threshold | mAP50 | Precision | Recall |
|:---------:|------:|----------:|-------:|
| 0.1 | 0.791 | 0.903 | 0.777 |
| 0.25 | 0.774 | 0.903 | 0.777 |
| 0.4 | 0.745 | 0.910 | 0.767 |
| 0.5 | 0.719 | 0.932 | 0.739 |
| 0.7 | 0.649 | 0.955 | 0.665 |

### Trade-off Analysis:
- **Higher threshold → Higher Precision, Lower Recall**
- **Lower threshold → Lower Precision, Higher Recall**
- mAP50 decreases as threshold increases (fewer detections)

### Decision:
**Selected threshold: 0.25**

**Justification:**
- Safety-critical application → high Recall is important
- Recall = 0.777 (same as 0.1, but better precision)
- Precision = 0.903 (acceptable)
- Balanced trade-off

### Limitations:
- Threshold was tuned on test set (should ideally be tuned on validation)
- Optimal threshold may vary by deployment scenario
- In production, threshold could be adjusted based on operator feedback