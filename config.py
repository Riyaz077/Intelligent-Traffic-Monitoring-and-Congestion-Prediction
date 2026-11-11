from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_INPUT = ROOT / "data" / "input"
DATA_OUTPUT = ROOT / "data" / "output"
MODEL_DIR = ROOT / "models"
FIGURE_DIR = ROOT / "reports" / "figures"
DEFAULT_MODEL = "yolov8n.pt"
VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck"}
DEFAULT_CONFIDENCE = 0.35
DEFAULT_FRAME_INTERVAL = 15

for folder in (DATA_INPUT, DATA_OUTPUT, MODEL_DIR, FIGURE_DIR):
    folder.mkdir(parents=True, exist_ok=True)
