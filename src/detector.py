"""YOLO object detection wrapper for traffic video frames."""
from __future__ import annotations
import cv2

VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck"}

class TrafficDetector:
    def __init__(self, model_name="yolov8n.pt", confidence=0.35):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Ultralytics is not installed. Run: pip install ultralytics") from exc
        self.model = YOLO(model_name)
        self.confidence = float(confidence)

    def detect(self, frame):
        """Return annotated BGR frame and class counts for one frame."""
        results = self.model.predict(frame, conf=self.confidence, verbose=False)
        counts = {name: 0 for name in sorted(VEHICLE_CLASSES)}
        annotated = frame.copy()
        if not results:
            return annotated, counts, []
        result = results[0]
        names = result.names
        detections = []
        if result.boxes is None:
            return annotated, counts, detections
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            label = names.get(cls_id, str(cls_id))
            if label not in VEHICLE_CLASSES:
                continue
            confidence = float(box.conf[0].item())
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            counts[label] += 1
            detections.append({
                "class": label, "confidence": confidence,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2
            })
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (30, 220, 90), 2)
            cv2.putText(annotated, f"{label} {confidence:.2f}", (x1, max(20, y1 - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 220, 90), 2)
        return annotated, counts, detections
