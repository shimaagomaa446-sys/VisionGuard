<div align="center">

# 🦺 VisionGuard

### AI-Powered Safety Monitoring System for Construction Sites

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=for-the-badge)](https://github.com/ultralytics/ultralytics)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

**Computer Vision system that detects PPE (Personal Protective Equipment) violations in construction sites.**

[Features](#-features) · [Installation](#-installation) · [Usage](#-usage) · [Results](#-results)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Model Performance](#-model-performance)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Endpoints](#-api-endpoints)
- [Results](#-results)
- [Limitations](#-limitations)
- [Author](#-author)

---

## 🎯 Overview

**VisionGuard** is an end-to-end computer vision system designed to assist safety officers in construction and industrial sites. It automatically detects workers and Personal Protective Equipment (PPE), identifies safety violations, and provides actionable insights through a modern web interface.

### What it does:

- 🎯 **Detects** workers and PPE in real-time
- ⚠️ **Identifies** safety violations (missing helmet / missing vest)
- 📊 **Provides** comprehensive compliance reports
- 🖥️ **Exposes** results via REST API and web UI

### Key Highlights:

- **Custom-trained YOLOv8s** model (not just a pre-trained API)
- **mAP50 = 0.811** on the test set
- **Real-time** inference (image, video, and live camera)
- **Production-ready** FastAPI backend + Streamlit frontend

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📸 **Image Analysis** | Upload and analyze single images |
| 🎥 **Video Analysis** | Process full videos with object tracking |
| 🎬 **Live Camera** | Real-time detection via WebRTC |
| 🔍 **5-Class Detection** | Person, Hardhat, NO-Hardhat, Safety Vest, NO-Safety Vest |
| ⚠️ **Compliance Logic** | Converts detections into violation reports |
| 📊 **Modern UI** | Clean, professional Streamlit interface |
| 🚀 **REST API** | FastAPI backend with documented endpoints |

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| **Deep Learning** | YOLOv8s (Ultralytics) |
| **Backend** | FastAPI + Uvicorn |
| **Frontend** | Streamlit |
| **Real-time** | streamlit-webrtc |
| **Computer Vision** | OpenCV, PIL |
| **Tracking** | ByteTrack (via Ultralytics) |
| **Language** | Python 3.11+ |

---

## 📊 Model Performance

### Test Set Results

| Metric | Value |
|--------|------:|
| **mAP50** | **0.811** |
| **mAP50-95** | **0.517** |
| **Precision** | **0.903** |
| **Recall** | **0.777** |

### Per-Class Performance (Test)

| Class | Precision | Recall | mAP50 |
|-------|:---------:|:------:|------:|
| Hardhat | 0.987 | 0.891 | **0.941** |
| Safety Vest | 0.913 | 0.857 | **0.908** |
| Person | 0.890 | 0.799 | **0.861** |
| NO-Safety Vest | 0.931 | 0.750 | **0.796** |
| NO-Hardhat | 0.792 | 0.585 | **0.547** |

### Experiments Summary

| Experiment | Model | Epochs | mAP50 | mAP50-95 |
|-----------|-------|:------:|------:|---------:|
| Baseline (Zero-shot) | YOLOv8n | 0 | 0.003 | 0.001 |
| Exp 1 | YOLOv8n | 25 | 0.773 | 0.437 |
| Exp 2 | YOLOv8n + Aug | 35 | 0.714 | 0.323 |
| **Exp 3 (Best)** | **YOLOv8s** | **35** | **0.830** | **0.534** |

---

## 📁 Project Structure

```
VisionGuard/
├── api/
│   ├── __init__.py
│   └── main.py                 # FastAPI endpoints
│
├── app/
│   ├── __init__.py
│   └── main.py                 # Streamlit UI
│
├── src/
│   ├── __init__.py
│   └── compliance.py           # Compliance logic
│
├── models/
│   └── best.pt                 # Trained YOLOv8s model
│
├── docs/
│   ├── data_quality_report.md
│   ├── error_analysis.md
│   ├── experiments.md
│   └── compliance_logic.md
│
├── .streamlit/
│   └── config.toml             # Streamlit theme config
│
├── requirements.txt            # Local dependencies
├── requirements-train.txt      # Training dependencies (Kaggle)
└── README.md
```

---

## 🚀 Installation

### Prerequisites

- Python 3.11+
- pip
- Git

### Steps

**1. Clone the repository**

```bash
git clone https://github.com/shimaagomaa446-sys/VisionGuard.git
cd VisionGuard
```

**2. Create virtual environment**

```bash
python -m venv venv

# Windows
source venv/Scripts/activate

# macOS / Linux
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Download the trained model**

Download `best.pt` from [Kaggle Dataset](https://www.kaggle.com/datasets/shimaagomaa88/visionguard-model-v1) and place it in:

```
models/best.pt
```

---

## 🎮 Usage

### Option 1: Full System (API + UI)

**Terminal 1 — Start FastAPI backend:**

```bash
uvicorn api.main:app --reload
```

**Terminal 2 — Start Streamlit frontend:**

```bash
streamlit run app/main.py
```

**Open in browser:**

- **Streamlit UI:** http://localhost:8501
- **API Docs:** http://localhost:8000/docs

### Option 2: API Only

```bash
uvicorn api.main:app --reload
```

Access the interactive API docs at: http://localhost:8000/docs

---

## 🔌 API Endpoints

### `POST /analyze`

Analyze a single image.

**Request:**
```bash
curl -X POST "http://127.0.0.1:8000/analyze" \
     -F "file=@image.jpg"
```

**Response:**
```json
{
  "annotated_image": "base64...",
  "results": [
    {
      "person_id": 1,
      "conf": 0.89,
      "has_hardhat": true,
      "has_vest": false,
      "violations": ["Missing Vest"],
      "status": "NON-COMPLIANT"
    }
  ],
  "summary": {
    "total_persons": 1,
    "compliant": 0,
    "non_compliant": 1
  }
}
```

### `POST /analyze-video`

Analyze a full video with object tracking.

**Request:**
```bash
curl -X POST "http://127.0.0.1:8000/analyze-video" \
     -F "file=@video.mp4"
```

**Response:**
```json
{
  "persons": [
    {
      "track_id": "1",
      "status": "NON-COMPLIANT",
      "violations": ["Missing Vest"],
      "appearances": 45,
      "avg_confidence": 0.85,
      "image": "base64..."
    }
  ],
  "summary": {
    "total_frames": 720,
    "processed_frames": 144,
    "unique_persons": 6,
    "compliant": 1,
    "non_compliant": 5
  }
}
```

---

## 📈 Results

### Training Progress

- **Baseline (Zero-shot):** mAP50 = 0.003
- **After Training:** mAP50 = **0.811**
- **Improvement:** **270× better**

### Key Achievements

- ✅ **Compliance Logic** with Point-in-Box association (better than IoU for small PPE)
- ✅ **Object Tracking** for accurate person counting in videos
- ✅ **Real-time Inference** via WebRTC
- ✅ **10 Documented Failure Cases** with detailed analysis

---

## ⚠️ Limitations

### Model Limitations

1. **Crowded scenes** — Detection accuracy drops with >10 workers
2. **Small objects** — Hardhats smaller than 2% of image may be missed
3. **Out-of-domain images** — Non-construction images may cause false positives
4. **Class imbalance** — NO-Hardhat has lower performance (mAP50 = 0.547)

### System Limitations

1. **Real-time mode** — Limited to ~3-8 FPS on CPU
2. **Video processing** — Uses sampling (every 5 frames) for speed
3. **Test set size** — Only 82 images (small for statistical significance)

### Known Issues

- **Live summary panel** requires manual refresh (Streamlit limitation)
- **Video output** not streamed (frames extracted instead)

---

## 🔬 Methodology

### Data

- **Dataset:** [Construction Site Safety Image Dataset](https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow) (Roboflow)
- **Total images:** 2,801 (Train: 2,605 / Valid: 114 / Test: 82)
- **Classes:** Reduced from 10 to 5 (Person, Hardhat, NO-Hardhat, Safety Vest, NO-Safety Vest)
- **Leakage check:** 2.47% (verified safe)

### Training

- **Model:** YOLOv8s (11M parameters)
- **Transfer learning** from COCO-pretrained weights
- **Epochs:** 35
- **Image size:** 640×640
- **Augmentation:** Default YOLO (mosaic, flip, HSV)

### Evaluation

- **Test set:** 82 images, 476 instances
- **Metrics:** mAP50, mAP50-95, Precision, Recall
- **Threshold tuning:** Evaluated on validation set

---

## 📚 Documentation

Detailed documentation available in the `docs/` folder:

- [Data Quality Report](docs/data_quality_report.md)
- [Error Analysis](docs/error_analysis.md)
- [Experiments Log](docs/experiments.md)
- [Compliance Logic](docs/compliance_logic.md)

---

## 👨‍💻 Author

**Shimaa Gomaa**

- GitHub: [@shimaagomaa446-sys](https://github.com/shimaagomaa446-sys)
- Kaggle: [@shimaagomaa88](https://www.kaggle.com/shimaagomaa88)

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

### Dataset License

The dataset used is licensed under **CC BY 4.0**.

---

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) — YOLOv8 implementation
- [Roboflow](https://roboflow.com/) — Dataset source
- [FastAPI](https://fastapi.tiangolo.com/) — Backend framework
- [Streamlit](https://streamlit.io/) — Frontend framework

---

<div align="center">

**⭐ If you find this project useful, please give it a star! ⭐**


</div>