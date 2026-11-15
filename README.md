# Smart-Traffic-Analytics

## Intelligent Traffic Monitoring and Congestion Prediction

An end-to-end computer-vision starter project for analyzing traffic videos, detecting road users, estimating traffic density, and producing interpretable congestion indicators through an interactive Streamlit dashboard.

> **Project status:** This repository is a development-ready prototype scaffold. It uses a pretrained Ultralytics YOLO model for object detection when installed and configured. It does not contain a custom-trained traffic model or validated real-world congestion forecasts. Evaluate on representative local traffic footage before reporting results.

## Objectives
- Detect vehicles and other road users in uploaded video footage.
- Count detected vehicles by class in sampled frames.
- Estimate traffic density and classify congestion using configurable thresholds.
- Visualize vehicle counts and congestion indicators over time.
- Export frame-level analysis as CSV.
- Provide a modular base for tracking, lane-wise analysis, speed estimation and forecasting.

## Features
- Video upload (MP4, MOV, AVI)
- YOLO-based object detection (pretrained weights)
- Vehicle-class filtering (car, motorcycle, bus, truck)
- Frame sampling to control processing time
- Annotated preview frame
- Frame-level vehicle counts and density estimates
- Rule-based congestion labels: Low, Moderate, High
- CSV export and interactive charts

## Architecture
```text
Traffic video
    ↓
Frame sampling and validation
    ↓
YOLO object detection
    ↓
Vehicle-class filtering
    ↓
Frame-level vehicle counting
    ↓
Traffic density estimation
    ↓
Congestion classification
    ↓
Dashboard, visualizations and CSV report
```

## Repository structure
```text
.
├── dashboard/app.py
├── src/
│   ├── detector.py
│   ├── traffic_analysis.py
│   └── video_utils.py
├── data/
│   ├── input/README.md
│   └── output/README.md
├── models/README.md
├── reports/figures/README.md
├── docs/
│   ├── architecture.md
│   ├── methodology.md
│   └── limitations.md
├── tests/
├── requirements.txt
├── config.py
└── README.md
```

## Installation
Python 3.10 or 3.11 is recommended.
```bash
python -m venv .venv
# Windows PowerShell:
.venv\\Scripts\\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

The app uses Ultralytics YOLO and may download pretrained weights on first run. Ensure you have permission to use the selected weights and comply with their license and terms.

## Run
```bash
streamlit run dashboard/app.py
```
Upload a short traffic video, choose frame sampling interval and confidence threshold, then start analysis.

## Output
- Annotated sample frame
- Frame-level counts for car, motorcycle, bus and truck
- Estimated density and congestion category
- Time-series charts
- Downloadable CSV

## Important interpretation notes
The prototype estimates congestion from vehicle counts in sampled frames. Counts are not equivalent to traffic flow, occupancy or queue length. Results depend on camera angle, field of view, frame sampling, detector performance and scene conditions. Meaningful congestion thresholds should be calibrated for the camera location and road geometry.

Speed estimation requires camera calibration (homography), known road reference measurements and reliable object tracking. It is not implemented as a validated feature in this starter.

## Suggested extensions
1. Add object tracking with persistent IDs (e.g. ByteTrack).
2. Define virtual counting lines and count unique line crossings.
3. Add lane segmentation and lane-wise congestion metrics.
4. Calibrate camera geometry for speed estimation.
5. Train or fine-tune on a locally representative traffic dataset.
6. Build a time-series forecasting model using historical counts and external variables.
7. Add alerting and multi-camera monitoring.

## Evaluation checklist
Report precision, recall and mAP for object detection using labelled test footage. For tracking, report IDF1/HOTA or a suitable tracking metric. For congestion prediction, define the prediction horizon and ground-truth congestion labels, then report MAE/RMSE for numeric flow or suitable classification metrics for congestion classes. Do not invent results.

## Data and privacy
Use footage you own or are authorized to process. Avoid collecting or publishing identifiable faces, license plates or other personal data. Follow local privacy, traffic-surveillance and data-retention requirements. This project is for research and demonstration, not enforcement or safety-critical control.

## Author

**Mohamed Riyaz S.**  
Integrated M.Tech, Computer Science and Engineering (Business Analytics)  
VIT Chennai
