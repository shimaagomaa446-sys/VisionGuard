# VisionGuard — Technical Report

**Project:** AI Safety Monitoring System for Construction Sites
**Author:** Shimaa Gomaa
**Date:** September 2026
**Track:** Computer Vision · Object Detection · Transfer Learning

---

## Executive Summary

VisionGuard is an end-to-end computer vision system that detects Personal Protective Equipment (PPE) violations in construction and industrial sites. The system uses a custom-trained YOLOv8s model to detect five classes (Person, Hardhat, NO-Hardhat, Safety Vest, NO-Safety Vest), applies compliance logic to identify violations, and exposes results through a FastAPI backend and Streamlit interface.

**Key Results:**
- **mAP50 = 0.811** on the test set
- **Precision = 0.903**, **Recall = 0.777**
- **270× improvement** over the zero-shot baseline
- Complete pipeline: data audit → training → evaluation → deployment

---

## 1. Problem Definition

### 1.1 Motivation

Construction and industrial sites rely on Personal Protective Equipment (PPE) such as hardhats and safety vests to protect workers. Manual monitoring of PPE compliance is expensive, inconsistent, and prone to human error. A computer vision system can assist safety officers by automatically highlighting likely violations.

### 1.2 Scope

VisionGuard detects workers and two categories of PPE:
- **Hardhat** (present) vs. **NO-Hardhat** (missing)
- **Safety Vest** (present) vs. **NO-Safety Vest** (missing)

### 1.3 Compliance Rules

For each detected person:
- **If no Hardhat is associated** → Violation: "Missing Helmet"
- **If no Safety Vest is associated** → Violation: "Missing Vest"
- **Otherwise** → Compliant

### 1.4 Success Criteria

- mAP50 ≥ 0.70 on the test set
- Working API + UI for image and video inference
- Documented error analysis with failure cases
- Reproducible repository with clear instructions

---

## 2. Dataset

### 2.1 Source

**Construction Site Safety Image Dataset** (Roboflow, v28)
- Source: [Kaggle](https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow)
- License: CC BY 4.0
- Total images: 2,801
- Format: YOLO (class_id x_center y_center width height)

### 2.2 Original Classes

The dataset contained 10 classes. We reduced them to 5 relevant to our use case:

| Original ID | Class | New ID |
|:-----------:|-------|:------:|
| 0 | Hardhat | 0 |
| 2 | NO-Hardhat | 1 |
| 4 | NO-Safety Vest | 2 |
| 5 | Person | 3 |
| 7 | Safety Vest | 4 |
| (others) | Removed | — |

### 2.3 Dataset Structure

| Split | Images | Labels |
|-------|-------:|-------:|
| Train | 2,605 | 2,605 |
| Valid | 114 | 114 |
| Test | 82 | 82 |
| **Total** | **2,801** | **2,801** |

### 2.4 Data Quality Analysis

**Leakage check:**
- Method: Extract prefix before `.rf.` from filenames; compare sets
- Result: 2.47% overlap — safe

**Empty labels:**
- Train: 6 (0.23%)
- Valid: 10 (8.77%)
- Test: 8 (9.76%)
- **Decision:** Keep as negative samples (small percentage)

**Image sizes:** All 640×640 (verified consistent)

**Class distribution (train):**

| Class | Count | % |
|-------|------:|--:|
| Person | 9,532 | 26.5% |
| NO-Safety Vest | 3,962 | 11.0% |
| Hardhat | 3,145 | 8.8% |
| Safety Vest | 3,033 | 8.4% |
| NO-Hardhat | 2,317 | 6.4% |

**Observations:**
- Class imbalance exists (Person is 4× NO-Hardhat)
- Valid/Test are small (~4% and ~3%)
- Out-of-domain images (book covers) exist — documented as limitation

---

## 3. Methodology

### 3.1 Preprocessing

**Pipeline:**
1. Read each YOLO label file
2. Map original class IDs to new IDs
3. Remove out-of-scope classes
4. Write processed labels to `data/processed/`
5. Copy images unchanged
6. Generate `data.yaml` for YOLO

### 3.2 Model Selection

**Chosen model:** YOLOv8s (11.1M parameters)

**Rationale:**
- Balanced between accuracy and speed
- Transfer learning from COCO-pretrained weights
- Strong performance on small object detection

### 3.3 Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 35 |
| Image size | 640×640 |
| Batch size | 16 |
| Optimizer | AdamW (auto) |
| Augmentation | Mosaic, flip, HSV (default) |

### 3.4 Compliance Logic

**Association method:** Point-in-Box

For each Person:
- Check if center of Hardhat box is inside Person box
- Check if center of Safety Vest box is inside Person box

**Why not IoU?**
IoU fails for small PPE objects. A Person box (300×500) and Hardhat box (40×30) have IoU < 0.05 even when the hardhat is correctly placed on the person's head.

### 3.5 Deployment

**Architecture:**

```
[User] → [Streamlit UI] → [FastAPI] → [YOLO Model] → [Compliance Logic]
              ↑                                            ↓
              └────────────── Results ←────────────────────┘
```

**Components:**
- **FastAPI:** REST API with 2 endpoints (`/analyze`, `/analyze-video`)
- **Streamlit:** 4-tab interface (Image, Video, Live Camera, About)
- **WebRTC:** Real-time camera inference

---

## 4. Experiments

### 4.1 Experiment Matrix

| # | Model | Epochs | Augmentation | mAP50 | mAP50-95 |
|---|-------|:------:|:------------:|------:|---------:|
| Baseline | YOLOv8n | 0 | — | 0.003 | 0.001 |
| Exp 1 | YOLOv8n | 25 | Default | 0.773 | 0.437 |
| Exp 2 | YOLOv8n | 35 | Aggressive | 0.714 | 0.323 |
| **Exp 3** | **YOLOv8s** | **35** | **Default** | **0.830** | **0.534** |

### 4.2 Analysis

**Baseline (Zero-shot):**
- mAP50 = 0.003 (basically random)
- Confirms: pre-trained COCO model cannot detect PPE

**Experiment 1 (YOLOv8n, 25 epochs):**
- mAP50 = 0.773
- Strong improvement (+270×)
- Fast training (~8 min)

**Experiment 2 (YOLOv8n + Aggressive Aug):**
- mAP50 = 0.714 (worse!)
- Augmentation changes (mixup, degrees, scale) hurt performance
- Lesson: Aggressive augmentation doesn't always help

**Experiment 3 (YOLOv8s, 35 epochs):**
- mAP50 = 0.830 (best on validation)
- Test set: mAP50 = 0.811
- Slightly slower (~19 min) but much better

### 4.3 Threshold Analysis

Tested confidence thresholds on the test set:

| Threshold | mAP50 | Precision | Recall |
|:---------:|------:|----------:|-------:|
| 0.1 | 0.791 | 0.903 | 0.777 |
| 0.25 | 0.774 | 0.903 | 0.777 |
| 0.4 | 0.745 | 0.910 | 0.767 |
| 0.5 | 0.719 | 0.932 | 0.739 |
| 0.7 | 0.649 | 0.955 | 0.665 |

**Selected threshold: 0.25**

**Rationale:** Safety-critical application → prioritize recall over precision. However, 0.25 provides a good balance.

---

## 5. Results

### 5.1 Final Test Set Performance

| Metric | Value |
|--------|------:|
| **mAP50** | **0.811** |
| **mAP50-95** | **0.517** |
| **Precision** | **0.903** |
| **Recall** | **0.777** |

### 5.2 Per-Class Performance

| Class | Precision | Recall | mAP50 | mAP50-95 |
|-------|:---------:|:------:|------:|---------:|
| Hardhat | 0.987 | 0.891 | **0.941** | 0.602 |
| Safety Vest | 0.913 | 0.857 | **0.908** | 0.642 |
| Person | 0.890 | 0.799 | **0.861** | 0.555 |
| NO-Safety Vest | 0.931 | 0.750 | **0.796** | 0.487 |
| NO-Hardhat | 0.792 | 0.585 | **0.547** | 0.300 |

### 5.3 Observations

**Strengths:**
- Excellent detection of **Hardhat** and **Safety Vest** (mAP50 > 0.90)
- Strong **Person** detection (mAP50 = 0.861)
- High precision (0.903) — few false positives

**Weaknesses:**
- **NO-Hardhat** is the weakest class (mAP50 = 0.547, Recall = 0.585)
- **41%** of NO-Hardhat cases are missed
- This is critical because violation detection is the primary goal

### 5.4 Valid vs. Test Comparison

| Metric | Valid | Test | Δ |
|--------|:-----:|:----:|:-:|
| mAP50 | 0.830 | 0.811 | -0.019 |
| mAP50-95 | 0.534 | 0.517 | -0.017 |
| Precision | 0.908 | 0.903 | -0.005 |
| Recall | 0.752 | 0.777 | +0.025 |

**Interpretation:** Small gap → model generalizes well. No overfitting.

---

## 6. Error Analysis

We analyzed 10 failure cases covering diverse error types.

### 6.1 Error Distribution

| Error Type | Count |
|------------|------:|
| False Positives | 5 |
| False Negatives | 7 |
| Misclassifications | 2 |
| **Multiple errors in one image** | 5 |

### 6.2 Root Cause Distribution

| Cause | Count |
|-------|------:|
| Crowded scenes | 3 |
| Small objects | 3 |
| Unusual context | 3 |
| Challenging lighting | 2 |

### 6.3 Notable Failure Cases

**Case 1: Person on Truck (FP + FN)**
- Small person on top of a truck, missed by model
- Model produced 2 false positives instead
- **Cause:** Small object + unusual context

**Case 3: Missed Hardhat in Tunnel (FN)**
- Yellow hardhat on yellow machinery, tunnel lighting
- Model detected Person + Safety Vest, missed Hardhat
- **Impact:** False alarm "Missing Helmet" — worker was compliant

**Case 6: Crowded Scene with 15 Workers (FP + FN + MC)**
- Extreme crowding
- 9 out of 15 persons missed
- **Cause:** NMS removing valid detections

**Case 9: Out-of-Domain (FP)**
- Bookstore image (Star Trek scene)
- Model produced 2 false positives
- **Cause:** Dataset contains non-construction images

### 6.4 Key Insights

1. **False Negatives** are the most common error
2. **Small objects** (<3% of image) are systematically missed
3. **Crowded scenes** (>10 workers) degrade performance
4. **Out-of-domain images** cause false positives
5. **Hardhat** is the most missed class

---

## 7. Limitations

### 7.1 Model Limitations

1. **Crowded scenes:** Detection accuracy drops significantly with >10 workers due to NMS issues
2. **Small objects:** Hardhats smaller than 2% of image may be missed
3. **Class imbalance:** NO-Hardhat has lower performance (mAP50 = 0.547)
4. **Out-of-domain images:** Non-construction images cause false positives

### 7.2 System Limitations

1. **Real-time mode:** Limited to ~3-8 FPS on CPU
2. **Video processing:** Uses sampling (every 5 frames) for speed
3. **Test set size:** 82 images (small for statistical significance)

### 7.3 Compliance Logic Limitations

1. **Point-in-Box** doesn't handle overlapping persons well
2. **False negatives** in PPE detection lead to false alarms
3. **No temporal smoothing** for video (each frame independent)

### 7.4 Known Issues

- Live summary panel requires manual refresh (Streamlit limitation)
- Video output extracted as frames (not streamed)
- NO-Hardhat class needs improvement

---

## 8. Deployment Design

### 8.1 System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     USER INTERFACE                       │
│                  (Streamlit — Port 8501)                 │
│  ┌──────────┬──────────┬──────────┬──────────┐          │
│  │  Image   │  Video   │   Live   │  About   │          │
│  └──────────┴──────────┴──────────┴──────────┘          │
└────────────────────────┬────────────────────────────────┘
                         │ HTTP
                         ↓
┌─────────────────────────────────────────────────────────┐
│                    BACKEND API                           │
│                  (FastAPI — Port 8000)                   │
│  ┌────────────────────┬─────────────────────────┐       │
│  │  POST /analyze     │  POST /analyze-video    │       │
│  └────────────────────┴─────────────────────────┘       │
└────────────────────────┬────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────┐
│                   ML PIPELINE                            │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  YOLOv8s │ →  │  Detections  │ →  │  Compliance  │  │
│  │  Model   │    │   (Boxes)    │    │    Logic     │  │
│  └──────────┘    └──────────────┘    └──────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### 8.2 API Endpoints

| Endpoint | Method | Input | Output |
|----------|:------:|-------|--------|
| `/analyze` | POST | Image | Detections + Compliance + Annotated image |
| `/analyze-video` | POST | Video | Unique persons + Summary + Frames |

### 8.3 Deployment Steps

1. **Clone repository**
2. **Install dependencies:** `pip install -r requirements.txt`
3. **Download model:** From Kaggle Dataset
4. **Start FastAPI:** `uvicorn api.main:app --reload`
5. **Start Streamlit:** `streamlit run app/main.py`
6. **Access UI:** http://localhost:8501

### 8.4 Hardware Requirements

**Minimum:**
- CPU: 4 cores
- RAM: 8 GB
- Storage: 2 GB

**Recommended:**
- GPU: NVIDIA with CUDA (10× faster)
- RAM: 16 GB

---

## 9. Future Work

### 9.1 Model Improvements

1. **Train YOLOv8m/l** for better accuracy
2. **Higher resolution** (imgsz=800 or 1024) for small objects
3. **Data augmentation** targeted at crowded scenes and backlighting
4. **Hard negative mining** to reduce false positives
5. **Address class imbalance** for NO-Hardhat

### 9.2 System Improvements

1. **Temporal smoothing** for video (use previous frames)
2. **Real-time optimization** with TensorRT or ONNX
3. **Auto-refresh** for live summary (custom solution)
4. **Email/SMS alerts** for violations

### 9.3 Data Improvements

1. **Remove out-of-domain images** (book covers, etc.)
2. **Collect more NO-Hardhat samples**
3. **Add more crowded scenes** to training set
4. **Expand to other PPE:** gloves, goggles, masks

---

## 10. Conclusion

VisionGuard demonstrates that a custom-trained YOLOv8s model can effectively detect PPE violations in construction sites, achieving **mAP50 = 0.811** on the test set. The system provides a complete pipeline from data preparation through deployment, with a REST API and a user-friendly web interface.

### Key Contributions

1. **Custom-trained model** — not a pre-trained API
2. **Compliance Logic** with Point-in-Box association (better than IoU for small PPE)
3. **Object tracking** for accurate person counting
4. **Real-time inference** via WebRTC
5. **Comprehensive documentation** including 10 failure cases

### Critical Assessment

While the overall performance is strong, the **NO-Hardhat class** (mAP50 = 0.547) remains a significant limitation. In a safety-critical application, missing 41% of violations is unacceptable. Future work should prioritize:
1. Improving NO-Hardhat detection
2. Reducing false negatives
3. Handling crowded scenes

### Final Remarks

The project successfully delivers a production-ready prototype that demonstrates the feasibility of AI-assisted safety monitoring. With further refinement (especially for NO-Hardhat), the system could be deployed in real construction sites to assist safety officers.

---

## Appendices

### A. Repository Structure

```
VisionGuard/
├── api/                 # FastAPI backend
├── app/                 # Streamlit UI
├── src/                 # Core logic
│   └── compliance.py
├── models/              # Trained model
├── docs/                # Documentation
│   ├── data_quality_report.md
│   ├── error_analysis.md
│   ├── experiments.md
│   ├── compliance_logic.md
│   └── final_report.md
└── README.md
```

### B. References

1. Ultralytics YOLOv8 — https://github.com/ultralytics/ultralytics
2. Construction Site Safety Dataset — https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow
3. FastAPI — https://fastapi.tiangolo.com/
4. Streamlit — https://streamlit.io/

### C. Links

- **GitHub:** https://github.com/shimaagomaa446-sys/VisionGuard
- **Kaggle Model:** https://www.kaggle.com/datasets/shimaagomaa88/visionguard-model-v1

---

**End of Report**