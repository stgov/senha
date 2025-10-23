"""
Extracción de embeddings para Re-ID de personas.
"""

import cv2
import numpy as np
from skimage.feature import hog


def extract_person_embedding(frame: np.ndarray, bbox: tuple, debug: bool = False) -> np.ndarray:
    """
    Extrae un embedding de 320 dimensiones combinando HSV, HOG y gradientes.
    
    Args:
        frame: Frame BGR completo
        bbox: (x1, y1, x2, y2) del bounding box
        debug: Mostrar información de debug
    
    Returns:
        Embedding normalizado de 320 dimensiones
    """
    x1, y1, x2, y2 = map(int, bbox)
    h, w = frame.shape[:2]
    
    # Validar y ajustar bbox
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    if x2 <= x1 or y2 <= y1:
        if debug:
            print(f"⚠️ BBox inválido: {bbox}")
        return np.zeros(320, dtype=np.float32)
    
    # Recortar ROI
    roi = frame[y1:y2, x1:x2]
    
    if roi.size == 0:
        return np.zeros(320, dtype=np.float32)
    
    # Redimensionar a tamaño estándar
    roi_resized = cv2.resize(roi, (64, 128))
    
    # === 1. Histograma HSV (96 dims) ===
    hsv = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256]).flatten()
    hist_hsv = np.concatenate([hist_h, hist_s, hist_v])
    hist_hsv = hist_hsv / (hist_hsv.sum() + 1e-6)  # Normalizar
    
    # === 2. HOG (64 dims) ===
    gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
    hog_features = hog(
        gray,
        orientations=8,
        pixels_per_cell=(16, 16),
        cells_per_block=(1, 1),
        visualize=False,
        feature_vector=True
    )
    
    # === 3. Gradientes de textura (32 dims) ===
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(sobelx**2 + sobely**2)
    
    # Dividir en bloques y promediar
    block_h, block_w = 32, 16
    gradient_features = []
    for i in range(4):
        for j in range(4):
            block = gradient_mag[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
            gradient_features.append(block.mean())
            gradient_features.append(block.std())
    
    gradient_features = np.array(gradient_features, dtype=np.float32)
    
    # === Combinar características ===
    embedding = np.concatenate([
        hist_hsv,           # 96 dims
        hog_features[:64],  # 64 dims (truncado si es más largo)
        gradient_features   # 32 dims
    ])
    
    # Asegurar 320 dimensiones exactas
    if len(embedding) < 320:
        embedding = np.pad(embedding, (0, 320 - len(embedding)))
    elif len(embedding) > 320:
        embedding = embedding[:320]
    
    # Normalización L2
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    
    if debug:
        print(f"📊 Embedding extraído: shape={embedding.shape}, norm={norm:.4f}")
    
    return embedding.astype(np.float32)
