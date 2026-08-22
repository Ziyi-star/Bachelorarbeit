from pathlib import Path
from tensorflow.keras.models import load_model


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "1s_30hz"
    / "2class_unbalanced_lab"
    / "20251128-121500_simclr_full_eval.keras"
)

classes = [
    "Normal cycling", 
    "Crossing curb"
]


#preprocessing helper function: segment_acceleration_data_overlapping_numpy
def predict_pipeline(text):
    text = re.sub(r'[!@#$(),\n"%^*?\:;~`0-9]', " ", text)
    text = re.sub(r"[[]]", " ", text)
    text = text.lower()
    pred = model.predict([text])
    return classes[pred[0]]
    
