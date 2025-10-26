import urllib.request
from pathlib import Path

MODELS = {
    "gesture_recognizer.task": "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task",
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "pose_landmarker_full.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
}
MODEL_DIR = Path("models")

def get_model_path(model_name: str) -> str:
    """
    Verifica si un modelo existe, lo descarga si no, y devuelve la ruta como string.
    """
    if model_name not in MODELS:
        raise ValueError(f"Modelo desconocido: {model_name}")

    MODEL_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / model_name

    if not model_path.exists():
        print(f"Descargando {model_name}...")
        url = MODELS[model_name]
        urllib.request.urlretrieve(url, model_path)
        print(f"Modelo {model_name} descargado en {model_path}")

    return str(model_path)
