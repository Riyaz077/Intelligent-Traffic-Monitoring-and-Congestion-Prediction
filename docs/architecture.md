# System Architecture

1. **Video input:** User uploads a permitted traffic video.
2. **Frame sampling:** OpenCV reads the video and samples every Nth frame.
3. **Object detection:** A pretrained Ultralytics YOLO detector identifies supported COCO vehicle classes.
4. **Aggregation:** Detections are counted per sampled frame.
5. **Congestion heuristic:** Configurable count thresholds map samples to Low, Moderate or High labels.
6. **Visualization:** Streamlit displays an annotated frame, count trend, label distribution and downloadable CSV.

This version does not yet perform persistent object tracking, lane assignment, calibrated speed measurement or a trained future-congestion forecast.
