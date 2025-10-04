# Registro de Limpieza del Repositorio

## ✅ Archivos Eliminados

### Scripts no utilizados
- `camera_script.py` - Script de utilidad de cámara no usado
- `gesture_recorder.py` - Script de grabación de gestos no usado
- `gesture_video.mp4` - Video de ejemplo

### Carpetas eliminadas
- `notebooks/` - Notebooks de Jupyter no necesarios para producción
- `images/` - Imágenes de ejemplo
- `photos_with_gestures/` - Directorio de fotos generadas
- `photos_without_gestures/` - Directorio de fotos generadas
- `__pycache__/` - Cache de Python

### Archivos de configuración
- `models/.gitkeep` - No necesario con .gitignore

## 📦 Archivos Mantenidos

### Scripts principales
- `person_tracker.py` - Script principal del proyecto ⭐
- `model_downloader.py` - Descargador automático de modelos

### Configuración Docker
- `Dockerfile` - Imagen Docker CPU
- `Dockerfile.gpu` - Imagen Docker GPU  
- `docker-compose.yml` - Orquestación CPU
- `docker-compose.gpu.yml` - Orquestación GPU
- `requirements-docker.txt` - Dependencias Docker CPU
- `requirements-gpu.txt` - Dependencias Docker GPU

### Configuración Local
- `requirements.txt` - Dependencias para ejecución local
- `.gitignore` - Exclusiones de Git (actualizado)
- `.dockerignore` - Exclusiones de Docker (actualizado)

### Documentación
- `README.md` - Documentación principal (reescrito)
- `WINDOWS.md` - Guía específica para Windows

## 🎯 Cambios en Configuración

### .gitignore
- ✅ Agregado `models/` - Los modelos se descargan automáticamente
- ✅ Simplificado para incluir solo lo esencial

### .dockerignore
- ✅ Excluye scripts no utilizados
- ✅ Excluye documentación del build
- ✅ Los modelos se montan como volumen, no se copian

### Docker Compose
- ✅ Eliminado warning de `version`
- ✅ Nombres de contenedor actualizados
- ✅ Volúmenes configurados para hot-reload
- ✅ Modelos se descargan automáticamente en primer uso

## 🚀 Resultado Final

### Estructura limpia:
```
senha/
├── person_tracker.py           # Script principal
├── model_downloader.py         # Descargador de modelos
├── requirements.txt            # Deps locales
├── requirements-docker.txt     # Deps Docker CPU
├── requirements-gpu.txt        # Deps Docker GPU
├── Dockerfile                  # Imagen CPU
├── Dockerfile.gpu              # Imagen GPU NVIDIA
├── Dockerfile.rocm             # Imagen GPU AMD
├── docker-compose.yml          # Compose unificado (CPU/NVIDIA/AMD)
├── docker-compose.gpu.yml      # Compose GPU NVIDIA (legacy)
├── .dockerignore               # Exclusiones Docker
├── .gitignore                  # Exclusiones Git
├── README.md                   # Documentación
├── WINDOWS.md                  # Guía Windows
├── ROCM.md                     # Guía AMD ROCm
└── models/                     # (generado automáticamente)
```

### Ventajas:
1. ✅ Repositorio limpio y fácil de mantener
2. ✅ Solo archivos esenciales
3. ✅ Configuración Docker optimizada
4. ✅ Descarga automática de modelos
5. ✅ Documentación clara y concisa
6. ✅ Soporte para CPU, NVIDIA y AMD en un solo compose
7. ✅ Hot-reload para desarrollo
8. ✅ Guías específicas para cada plataforma

## 📝 Próximos Pasos

Para usar el proyecto:

### Local
```bash
python person_tracker.py lite
```

### Docker CPU
```bash
docker-compose up person-tracker
```

### Docker GPU NVIDIA
```bash
docker-compose up person-tracker-gpu
```

### Docker GPU AMD
```bash
docker-compose up person-tracker-rocm
```
