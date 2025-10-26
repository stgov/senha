import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python  # <-- AÑADIDO: Importar el módulo 'python'
from mediapipe.tasks.python import vision
# (Línea eliminada)

class MediaPipeProcessor:
    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        Inicializa el procesador de gestos de MediaPipe.
        
        Args:
            model_path (str): Ruta al modelo .task de gestos.
            device (str): 'cuda' o 'cpu'.
        """
        print(f"[MediaPipe] Cargando modelo de gestos: {model_path}")
        
        # Configurar delegados de GPU si se usa 'cuda'
        delegate = None
        if device == 'cuda':
            try:
                # La línea que usaba gpu_options_lib se ha eliminado,
                # ya que no es necesaria para habilitar el delegado.
                
                # --- CORRECCIÓN 1 ---
                # Usar la nueva importación
                delegate = python.BaseOptions.Delegate.GPU
                print("[MediaPipe] Delegado de GPU (CUDA) habilitado.")
            except Exception as e:
                print(f"[WARN] No se pudo inicializar el delegado de GPU de MediaPipe: {e}")
                print("[WARN] MediaPipe funcionará en CPU.")
                delegate = None

        # Opciones base con el modelo y el delegado
        # --- CORRECCIÓN 2 ---
        # Usar la nueva importación
        base_options = python.BaseOptions(
            model_asset_path=model_path,
            delegate=delegate
        )
        
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE, # Modo síncrono
            num_hands=2
        )
        
        try:
            self.recognizer = vision.GestureRecognizer.create_from_options(options)
            print("[MediaPipe] Reconocefor de gestos cargado.")
        except Exception as e:
            print(f"[ERROR] No se pudo cargar el modelo de gestos de MediaPipe: {e}")
            print("Asegúrate de que el archivo del modelo existe y es válido.")
            raise e

    def process(self, roi_rgb: np.ndarray) -> vision.GestureRecognizerResult:
        """
        Procesa un ROI (en RGB) para detectar gestos.
        
        Args:
            roi_rgb: La imagen del ROI en formato RGB.
            
        Returns:
            El objeto GestureRecognizerResult crudo de MediaPipe.
        """
        try:
            # Asegurarse de que el array sea contiguo en memoria
            if not roi_rgb.flags['C_CONTIGUOUS']:
                roi_rgb = np.ascontiguousarray(roi_rgb, dtype=np.uint8)

            # Crear imagen de MediaPipe
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=roi_rgb)
            
            # Reconocer gestos
            gesture_result = self.recognizer.recognize(mp_image)
            return gesture_result
            
        except Exception as e:
            # print(f"[Error MP Process]: {e}")
            # Devolver un resultado vacío en caso de error (ej. ROI muy pequeño)
            return vision.GestureRecognizerResult(gestures=[], hand_landmarks=[], handedness=[], hand_world_landmarks=[])

