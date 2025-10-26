import cv2
import numpy as np
from skimage.feature import hog

# Dimensiones fijas para los embeddings
# HSV (H+S+V) = 48 + 24 + 24 = 96
# HOG = 192
# LBP = 32
# TOTAL = 320
EMBEDDING_DIM = 320
HOG_DIM = 192
LBP_DIM = 32

# Configuración de HOG (simplificado para ROIs pequeños)
HOG_ORIENTATIONS = 8
HOG_PIXELS_PER_CELL = (16, 16) # Celdas más grandes
HOG_CELLS_PER_BLOCK = (2, 2)  # Bloques estándar

# Tamaño fijo para análisis
ROI_SIZE = (96, 192) # w, h (vertical)

def extract_embedding(roi: np.ndarray) -> np.ndarray:
    """
    Extrae un embedding de características combinadas (Color, Forma, Textura).
    El vector resultante SIEMPRE tendrá 'EMBEDDING_DIM' dimensiones.
    """
    
    features = []
    
    # --- Validación de ROI ---
    if roi.size == 0:
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)
        
    try:
        # Redimensionar a tamaño fijo para consistencia
        roi_resized = cv2.resize(roi, ROI_SIZE)
        gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2HSV)
    except cv2.error:
        # Error si el ROI es inválido (ej. 0x0)
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    # --- 1. Color (HSV Histogram) - 96 dims ---
    # cv2.calcHist devuelve float32, por eso normalize() funciona aquí
    hist_h = cv2.calcHist([hsv], [0], None, [48], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [24], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [24], [0, 256])
    
    cv2.normalize(hist_h, hist_h)
    cv2.normalize(hist_s, hist_s)
    cv2.normalize(hist_v, hist_v)
    
    color_features = np.concatenate((hist_h.flatten(), hist_s.flatten(), hist_v.flatten()))
    features.append(color_features)

    # --- 2. Forma (HOG) - HOG_DIM (192) dims ---
    hog_features = hog(
        gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        feature_vector=True
    )
    
    # Asegurar que HOG tenga exactamente HOG_DIM dimensiones
    if hog_features.shape[0] < HOG_DIM:
        hog_features_padded = np.pad(hog_features, (0, HOG_DIM - hog_features.shape[0]), 'constant')
    else:
        hog_features_padded = hog_features[:HOG_DIM]
        
    features.append(hog_features_padded)

    # --- 3. Textura (LBP Simplificado) - LBP_DIM (32) dims ---
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # np.histogram devuelve int64, ¡ese es el problema!
    hist_grad, _ = np.histogram(magnitude.flatten(), bins=LBP_DIM, range=(0, 255*2))
    
    # --- CORRECCIÓN DEL ERROR ---
    # Convertir a float ANTES de normalizar in-place
    hist_grad_float = hist_grad.astype(np.float32)
    cv2.normalize(hist_grad_float, hist_grad_float) # Ahora sí funciona (float -> float)
    features.append(hist_grad_float)
    # ---------------------------

    # --- Concatenar y Normalizar ---
    try:
        embedding = np.concatenate(features).astype(np.float32)
    except ValueError as e:
        print(f"Error concatenando features: {e}. Shapes: {[f.shape for f in features]}")
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)
        
    # Validar dimensión final
    if embedding.shape[0] != EMBEDDING_DIM:
        embedding_padded = np.pad(embedding, (0, EMBEDDING_DIM - embedding.shape[0]), 'constant')
        embedding = embedding_padded[:EMBEDDING_DIM]

    # Normalización L2 (crucial para similitud Coseno en ChromaDB)
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
        
    return embedding

