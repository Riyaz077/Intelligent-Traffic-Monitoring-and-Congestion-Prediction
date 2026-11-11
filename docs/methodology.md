# Methodology

## Prototype workflow
- Select an authorized video and document camera location, date, view and conditions.
- Sample frames at a fixed interval to manage processing time.
- Run the selected YOLO weights at a documented confidence threshold.
- Retain detections for car, motorcycle, bus and truck classes.
- Aggregate class counts and total detections for each sampled frame.
- Apply configurable count thresholds to create exploratory congestion labels.
- Export frame-level results for further analysis.

## Recommended rigorous evaluation
- Label a representative test set and calculate object-detection precision, recall and mAP@0.5 / mAP@0.5:0.95.
- Evaluate tracking with IDF1 or HOTA after adding a tracker.
- Define ground-truth congestion labels and prediction horizon before building a forecasting model.
- Use temporal train/validation/test splits to avoid leakage.
- Report performance by time of day, weather, camera angle and road type.
