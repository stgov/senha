import urllib.request
from pathlib import Path


class ModelDownloader:
    """Descargador automático de modelos de MediaPipe"""
    
    MODELS = {
        "gesture_recognizer.task": "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task",
        "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
        "pose_landmarker_full.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
    }
    
    def __init__(self, models_dir="models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
    
    def download_model(self, model_name, url):
        """Descargar un modelo específico"""
        model_path = self.models_dir / model_name
        
        if model_path.exists():
            print(f"✓ Modelo ya existe: {model_name}")
            return model_path
        
        print(f"📥 Descargando {model_name}...")
        try:
            # Descargar con barra de progreso
            def reporthook(blocknum, blocksize, totalsize):
                readsofar = blocknum * blocksize
                if totalsize > 0:
                    percent = readsofar * 100 / totalsize
                    s = f"\r{percent:5.1f}% {readsofar:,} / {totalsize:,} bytes"
                    print(s, end='')
                else:
                    print(f"\r{readsofar:,} bytes", end='')
            
            urllib.request.urlretrieve(url, model_path, reporthook)
            print(f"\n✓ Descarga completada: {model_name}")
            return model_path
        
        except Exception as e:
            print(f"\n✗ Error descargando {model_name}: {e}")
            if model_path.exists():
                model_path.unlink()
            raise
    
    def download_all(self):
        """Descargar todos los modelos necesarios"""
        print("🚀 Verificando modelos de MediaPipe...")
        
        for model_name, url in self.MODELS.items():
            self.download_model(model_name, url)
        
        print("\n✅ Todos los modelos están listos")
    
    def download_required(self, pose_model="lite"):
        """Descargar solo los modelos requeridos"""
        print("🚀 Verificando modelos necesarios...")
        
        # Siempre descargar gesture recognizer
        self.download_model(
            "gesture_recognizer.task",
            self.MODELS["gesture_recognizer.task"]
        )
        
        # Descargar el modelo de pose específico
        pose_model_name = f"pose_landmarker_{pose_model}.task"
        if pose_model_name in self.MODELS:
            self.download_model(pose_model_name, self.MODELS[pose_model_name])
        else:
            raise ValueError(f"Modelo de pose inválido: {pose_model}")
        
        print("\n✅ Modelos necesarios listos")


if __name__ == "__main__":
    # Script standalone para descargar modelos
    import sys
    
    downloader = ModelDownloader()
    
    if len(sys.argv) > 1 and sys.argv[1] == "all":
        downloader.download_all()
    else:
        # Por defecto, descargar modelos necesarios
        pose = sys.argv[1] if len(sys.argv) > 1 else "lite"
        downloader.download_required(pose)
