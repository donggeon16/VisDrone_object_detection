from pathlib import Path
import torch

BASE = Path(__file__).resolve().parents[2]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASSES = {
    0: "ignored regions", 
    1: "pedestrian", 
    2: "people", 
    3: "bicycle", 
    4: "car", 
    5: "van", 
    6: "truck", 
    7: "tricycle", 
    8: "awning-tricycle", 
    9: "bus", 
    10: "motor", 
    11: "others"
    }

CLASS_MAPPING = {
    1: 0,   # pedestrian
    4: 1,   # car
    10: 2   # motor
}