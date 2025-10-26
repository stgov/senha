import mediapipe as mp
import numpy as np

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

class MediaPipeProcessor:
    """
    Ejecuta GestureRecognizer en modo síncrono (IMAGE) sobre un ROI dado.
    (Se eliminó PoseLandmarker para optimización).
    """
    
    def __init__(self, gesture_model_path: str):
        
        print("Loading MediaPipe Gesture model...")
        gesture_options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=gesture_model_path),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=2 # Buscará hasta 2 manos dentro del ROI de la persona
        )
        self.gesture_recognizer = GestureRecognizer.create_from_options(gesture_options)
        print("MediaPipe Gesture model loaded.")

    def process(self, roi_rgb: np.ndarray) -> dict:
        """
        Procesa un solo ROI para gestos.

        Args:
            roi_rgb: El ROI de la persona en formato RGB.

        Returns:
            Un diccionario con resultados:
            {
                "hand_landmarks": (landmarks | None),
                "gesture_name": (str | None)
            }
        """
        if roi_rgb.size == 0:
            return {}

        # MediaPipe requiere un array C-contiguo.
        # np.ascontiguousarray() crea una copia con el formato de memoria correcto.
        try:
            roi_rgb_contiguous = np.ascontiguousarray(roi_rgb, dtype=np.uint8)
        except ValueError:
             # Si el ROI está corrupto o vacío
            return {}

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=roi_rgb_contiguous)
        
        results = {
            "hand_landmarks": None,
            "gesture_name": None
        }

        # Detección de Gestos
        try:
            gesture_result = self.gesture_recognizer.recognize(mp_image)
            
            if gesture_result.gestures and gesture_result.hand_landmarks:
                # Devuelve el gesto con mayor score (de la primera mano detectada)
                top_gesture = gesture_result.gestures[0][0]
                results["gesture_name"] = top_gesture.category_name
                # Devuelve todos los landmarks de manos detectados
                results["hand_landmarks"] = gesture_result.hand_landmarks
                
        except Exception as e:
            # print(f"Error en Gesture recognition: {e}")
            pass # Ignorar si no detecta manos

        return results

