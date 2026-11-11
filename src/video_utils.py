"""Video validation and frame sampling helpers."""
import cv2

SUPPORTED_EXTENSIONS = {"mp4", "mov", "avi", "mkv"}

def open_video(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("Could not open video. Check the file format and encoding.")
    return cap

def get_video_metadata(cap):
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    return {
        "fps": float(fps),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
    }
