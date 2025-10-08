# Sistema de Re-Identificación de Personas con Gestos

Sistema avanzado de seguimiento y re-identificación de personas con reconocimiento de gestos usando MediaPipe y base de datos vectorial.

## Características Principales

### Re-Identificación (Re-ID)
- **Embeddings visuales**: Genera firmas únicas basadas en histogramas de color HSV y características de textura
- **Base de datos vectorial**: Usa ChromaDB en memoria para búsqueda por similitud coseno
- **Persistencia de identidad**: Reconoce personas que salen y vuelven al frame
- **IDs únicos**: Asigna identificadores persistentes (P001, P002, etc.)

### Detección y Marcado por Gestos
- **Closed Fist**: Detecta el gesto de puño cerrado
- **Marcado permanente**: La persona marcada mantiene su estado hasta el fin del video
- **Asociación inteligente**: Vincula gestos con la persona más cercana usando proximidad espacial

### Procesamiento
- Detección multi-persona (hasta 6 personas simultáneas)
- Detección multi-mano (hasta 12 manos)
- Suavizado de bounding boxes con zona muerta configurable
- Todo en memoria (sin persistencia en disco)

## Requisitos

- Python 3.11 (Python 3.12 no compatible)
- Webcam
- Linux/macOS para Docker (Windows usar ejecucion local)

## Instalación y Uso

### Instalación de Dependencias

```bash
# Instalar dependencias
pip install -r requirements.txt
```

### Descarga de Modelos

Los modelos se descargan automáticamente en la primera ejecución, o puedes hacerlo manualmente:

```bash
python scripts/model_downloader.py
```

### Ejecución del Sistema

```bash
# Ejecutar sistema con configuración por defecto
python main.py
```

### Configuración

Edita `main.py` para cambiar la configuración:

```python
config = TrackerConfig(
    pose_model="lite",              # "lite" o "full"
    camera_source=0,                # 0 para webcam, "ruta/video.mp4" para archivo
    num_poses=6,                    # Máximo de personas a detectar
    max_num_hands=12,               # Máximo de manos a detectar
    gesture_score_threshold=0.3,    # Umbral de confianza para gestos
)
```

### Opcion 2: Docker (Linux/macOS)

```bash
# Construir imagen
docker compose build

# Permitir acceso X11
xhost +local:docker

# Ejecutar
docker compose up

# Limpiar permisos X11
xhost -local:docker
```

## Estructura del Proyecto

```
senha-1/
├── main.py                    # Punto de entrada principal
├── requirements.txt           # Dependencias del proyecto
├── README.md                  # Este archivo
│
├── src/
│   ├── core/
│   │   ├── app.py            # Aplicación principal de tracking
│   │   ├── config.py         # Configuración del sistema
│   │   ├── model_manager.py  # Gestor de modelos MediaPipe
│   │   ├── person_reid.py    # Sistema de Re-Identificación
│   │   └── stats.py          # Estadísticas de tracking
│   │
│   ├── drawers/
│   │   ├── hand_drawer.py    # Visualización de manos y gestos
│   │   └── pose_drawer.py    # Visualización de poses y bboxes
│   │
│   └── smoothing/
│       └── bbox_smoother.py  # Suavizado de bounding boxes
│
├── scripts/
│   └── model_downloader.py   # Descarga automática de modelos
│
├── models/                    # Modelos MediaPipe (auto-descargados)
└── media/                     # Videos de prueba
```

## Cómo Funciona

### Flujo de Re-Identificación

1. **Detección**: MediaPipe detecta personas en el frame
2. **Extracción**: Se genera un embedding visual de cada persona
3. **Búsqueda**: Se busca en la base de datos vectorial por similitud
4. **Match/Nuevo**: 
   - Si similitud > 75%: Se re-identifica como persona existente
   - Si similitud < 75%: Se crea nueva entrada con ID único

### Marcado por Gesto

1. Persona detectada en el frame
2. Hace gesto **Closed_Fist** (puño cerrado)
3. Sistema asocia el gesto con la persona más cercana (< 200px)
4. Persona queda **marcada permanentemente** (borde amarillo)
5. El marcado persiste aunque salga y vuelva al frame

### Controles

- **[H]**: Toggle visualización de manos
- **[P]**: Toggle visualización de poses
- **[Q]**: Salir del sistema

### Visualización

- **Borde verde**: Persona no marcada
- **Borde amarillo grueso**: Persona marcada con Closed_Fist
- **ID persistente**: P001, P002, P003, etc.
- **Estadísticas Re-ID**: Total, activas, marcadas

## Tecnologías

### Core
- **MediaPipe 0.10.14**: Detección de gestos y pose
- **OpenCV 4.10**: Procesamiento de video
- **Python 3.11**: Runtime
- **NumPy 1.26**: Operaciones numéricas

### Re-ID
- **ChromaDB**: Base de datos vectorial en memoria
- **scikit-learn**: Normalización de embeddings
- **Similitud coseno**: Métrica de comparación de personas

### Embeddings Visuales
- Histogramas HSV (color)
- Gradientes Sobel (textura)
- Bordes Canny (estructura)
- Normalización L2

## Modelos y Parámetros

### Modelos de Pose

- **lite** (default): Más rápido, menos preciso
- **full**: Más preciso, más lento

### Parámetros de Re-ID

```python
PersonReID(
    similarity_threshold=0.75,    # Umbral de similitud coseno (0-1)
    max_absent_frames=500         # ~10 segundos a 30fps antes de olvidar
)
```

### Características del Embedding

- **Dimensión**: 256 features
  - 96 (histogramas HSV: 32+32+32)
  - 32 (gradientes)
  - 32 (bordes)
  - ... (normalizados a 256 total)
- **Actualización**: Promedio móvil (70% anterior + 30% nuevo)

## Limitaciones y Consideraciones

### Técnicas
- **Embeddings simples**: Para producción considerar modelos pre-entrenados (ResNet, OSNet)
- **Memoria volátil**: La base de datos se reinicia al cerrar el programa
- **Iluminación**: Requiere buena iluminación para detección óptima
- **Oclusiones**: El Re-ID puede fallar con oclusiones parciales

### Compatibilidad
- **Python 3.12**: No compatible (MediaPipe requiere Python 3.11)
- **Windows Docker**: Usar ejecución local (Docker Desktop no soporta cámara)

## Mejoras Futuras

- [ ] Embeddings con redes neuronales (ResNet, OSNet)
- [ ] Persistencia opcional en disco
- [ ] Re-ID multi-cámara
- [ ] Métricas de evaluación (mAP, CMC)
- [ ] Tracking temporal (Kalman filter)

## Licencia

MIT
