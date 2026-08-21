from pathlib import Path
import re
from tensorflow.keras.models import load_model

BASE_DIR = Path(__file__).resolve().parent

model = load_model(
    BASE_DIR / "20251208-101322_simclr_full_eval.keras",
    compile=False,
)


classes = [
    "Normal cycling", 
    "Crossing curb"
]


#preprocessing helper function
def segmentation_pipeline(data):
    
