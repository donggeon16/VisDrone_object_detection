from pathlib import Path
import torch

BASE = Path(__file__).resolve().parents[2]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASSES = ["ignored regions", "pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor", "others"]