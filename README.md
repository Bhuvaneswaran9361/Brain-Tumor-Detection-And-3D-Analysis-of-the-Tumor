# Brain Tumor Detection & Analysis System

## Overview

NeuroScan AI is an AI-powered medical imaging platform designed for brain tumor detection, classification, segmentation, visualization, and automated report generation.

The project combines:

* **FastAPI Backend** for inference and model serving
* **React + Vite Frontend** for interactive visualization
* **Deep Learning Models** for tumor detection and segmentation
* **3D Reconstruction** for tumor visualization
* **Grad-CAM Explainability** for model interpretation
* **Clinical Report Generation** in PDF format

---

# Project Architecture

```text
Frontend (React + Vite)
        │
        ▼
FastAPI Backend API
        │
        ├── HADF Filtering
        ├── YOLO-NAS Detection
        ├── En-DeNet Segmentation
        ├── Grad-CAM Visualization
        ├── 3D Reconstruction
        └── PDF Report Generation
```

---

# Project Structure

```text
full_project_report/
│
├── backend/
│   ├── main.py
│   ├── gpu_test.py
│   ├── test_inference.py
│   ├── train_models.py
│   ├── requirements.txt
│   │
│   └── models/
│       ├── endenet_model.py
│       ├── gradcam.py
│       ├── hadf_filter.py
│       ├── inference.py
│       ├── reconstruction_3d.py
│       ├── report_generator.py
│       └── yolonas_model.py
│
├── frontend/
│   ├── package.json
│   ├── vite.config.js
│   ├── index.html
│   │
│   └── src/
│       ├── main.jsx
│       ├── App.jsx
│       ├── index.css
│       │
│       ├── components/
│       │   ├── LoadingOverlay.jsx
│       │   ├── MetricsGrid.jsx
│       │   ├── ModelInfoPanel.jsx
│       │   ├── ProbabilityChart3D.jsx
│       │   ├── SegmentationPanel.jsx
│       │   ├── TumorViewer3D.jsx
│       │   └── UploadZone.jsx
│       │
│       └── utils/
│           └── api.js
│
└── README.md
```

---

# Core Features

## Backend Features

* FastAPI REST API
* Brain tumor detection pipeline
* MRI image preprocessing
* HADF filtering
* YOLO-NAS object detection
* En-DeNet segmentation
* Grad-CAM explainability
* 3D tumor reconstruction
* Batch prediction support
* PDF medical report generation
* Streaming prediction progress
* GPU support using PyTorch

---

## Frontend Features

* Modern React UI
* Drag & drop image upload
* Real-time prediction visualization
* 3D tumor rendering
* Interactive metrics dashboard
* Segmentation overlays
* Probability charts
* Responsive design
* Smooth animations using Framer Motion

---

# Technologies Used

## Backend

| Technology | Purpose                 |
| ---------- | ----------------------- |
| Python     | Core backend language   |
| FastAPI    | API framework           |
| PyTorch    | Deep learning framework |
| OpenCV     | Image processing        |
| NumPy      | Numerical operations    |
| SciPy      | Scientific computing    |
| Matplotlib | Visualization           |
| Pillow     | Image handling          |

---

## Frontend

| Technology        | Purpose              |
| ----------------- | -------------------- |
| React             | Frontend framework   |
| Vite              | Frontend bundler     |
| Three.js          | 3D rendering         |
| React Three Fiber | React 3D integration |
| Framer Motion     | Animations           |
| Axios             | API requests         |
| React Dropzone    | File upload          |

---

# System Requirements

## Minimum Requirements

| Component | Requirement                      |
| --------- | -------------------------------- |
| OS        | Windows / Linux / macOS          |
| RAM       | 8 GB                             |
| Python    | 3.10+                            |
| Node.js   | 18+                              |
| GPU       | Optional                         |
| CUDA      | Recommended for GPU acceleration |

---

# Backend Setup

## Step 1 – Navigate to Backend

```bash
cd backend
```

---

## Step 2 – Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## Step 3 – Install Dependencies

```bash
pip install -r requirements.txt
```

Installed packages include:

```text
fastapi
uvicorn
python-multipart
Pillow
numpy
opencv-python-headless
torch
torchvision
scipy
scikit-image
matplotlib
pydantic
```

---

## Step 4 – Run Backend Server

### Development Mode

```bash
python main.py --reload
```

### Production Mode

```bash
python main.py --host 0.0.0.0 --port 8000
```

---

## Backend API URLs

| Endpoint              | Description             |
| --------------------- | ----------------------- |
| `/health`             | Health check            |
| `/api/model-info`     | Model information       |
| `/api/predict`        | Single image prediction |
| `/api/report`         | Generate PDF report     |
| `/api/predict/stream` | Streaming inference     |
| `/api/batch`          | Batch image analysis    |
| `/api/classes`        | Tumor class metadata    |
| `/docs`               | Swagger documentation   |
| `/redoc`              | ReDoc documentation     |

---

# Frontend Setup

## Step 1 – Navigate to Frontend

```bash
cd frontend
```

---

## Step 2 – Install Dependencies

```bash
npm install
```

---

## Step 3 – Start Frontend

```bash
npm run dev
```

Frontend will run at:

```text
http://localhost:5173
```

---

## Step 4 – Build Production Version

```bash
npm run build
```

---

## Step 5 – Preview Production Build

```bash
npm run preview
```

---

# Full Project Execution

## Start Backend

Open terminal 1:

```bash
cd backend
python main.py --reload
```

Backend URL:

```text
http://127.0.0.1:8000
```

---

## Start Frontend

Open terminal 2:

```bash
cd frontend
npm install
npm run dev
```

Frontend URL:

```text
http://localhost:5173
```

---

# Workflow Execution

## Step-by-Step Flow

1. Upload MRI image from frontend
2. Frontend sends image to FastAPI backend
3. HADF filter preprocesses MRI image
4. YOLO-NAS detects tumor region
5. En-DeNet performs segmentation
6. Grad-CAM generates explainability heatmap
7. 3D reconstruction generates tumor visualization
8. Backend returns prediction results
9. Frontend visualizes analytics and 3D output
10. PDF report can be downloaded

---

# Running GPU Test

```bash
python gpu_test.py
```

This checks:

* CUDA availability
* GPU memory
* PyTorch GPU support

---

# Running Inference Test

```bash
python test_inference.py
```

---

# Model Training

Run:

```bash
python train_models.py
```

You can customize:

* Dataset paths
* Epochs
* Batch size
* Learning rate
* Optimizers
* Model architecture

---

# Example API Request

## Prediction Request

```bash
curl -X POST "http://127.0.0.1:8000/api/predict" \
  -F "file=@mri_scan.jpg"
```

---

# Example API Response

```json
{
  "prediction": "Glioma",
  "confidence": 98.7,
  "segmentation_mask": "base64_image",
  "gradcam": "base64_heatmap",
  "tumor_volume": "3D_data",
  "report": "generated"
}
```

---

# 3D Visualization

The frontend includes:

* TumorViewer3D
* ProbabilityChart3D
* Interactive Three.js rendering
* Zoom and rotation controls

---

# Security Recommendations

For production deployment:

* Add authentication
* Enable HTTPS
* Validate uploaded files
* Limit upload size
* Use environment variables
* Add rate limiting
* Configure CORS securely

---

# Deployment

## Backend Deployment

Recommended platforms:

* Docker
* AWS EC2
* Azure App Service
* Render
* Railway
* DigitalOcean

---

## Frontend Deployment

Recommended platforms:

* Vercel
* Netlify
* Firebase Hosting

---

# Suggested Future Improvements

* User authentication system
* Database integration
* Patient management system
* Multi-model ensemble inference
* DICOM support
* Real-time collaborative diagnostics
* Cloud GPU inference
* Mobile application
* PACS integration

---

# Troubleshooting

## Python Not Found

Install Python 3.10+ and add it to PATH.

---

## Node Modules Error

Delete:

```bash
node_modules
package-lock.json
```

Then reinstall:

```bash
npm install
```

---

## CUDA Not Detected

Check:

```bash
nvidia-smi
```

Reinstall matching CUDA-enabled PyTorch.

---

## Port Already in Use

Change backend port:

```bash
python main.py --port 9000
```

---

# License

This project is intended for educational and research purposes.

---

# Disclaimer

This software is not certified for clinical diagnosis.
It should only be used for research, educational, and experimental purposes.

---

# Authors

Developed as an advanced AI-based Brain Tumor Detection and Visualization platform.

---

# Final Notes

NeuroScan AI demonstrates a complete end-to-end medical AI workflow integrating:

* Computer Vision
* Deep Learning
* Medical Imaging
* Explainable AI
* 3D Reconstruction
* Interactive Frontend Visualization
* Automated Reporting

This project can serve as:

* Final year project
* Research prototype
* AI healthcare demo
* Medical imaging framework
* Deep learning portfolio project
