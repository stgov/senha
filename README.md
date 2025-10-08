# Person Tracker with Gesture Recognition

Sistema de seguimiento de personas activado por reconocimiento de gestos usando MediaPipe.

## Caracteristicas

- Deteccion de gesto Closed Fist (puno cerrado)
- Seguimiento automatico de la persona que hace el gesto
- Procesamiento en paralelo de modelos de gestos y pose
- Ejecucion en CPU con Python 3.11
- Soporte para Docker

## Requisitos

- Python 3.11 (Python 3.12 no compatible)
- Webcam
- Linux/macOS para Docker (Windows usar ejecucion local)

## Instalacion y Uso

### Opcion 1: Ejecucion Local

```bash
# Clonar repositorio
git clone <repo-url>
cd senha

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar (modelos se descargan automaticamente)
python person_tracker.py lite
python person_tracker.py full
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

## Estructura

```
senha/
 person_tracker.py
 model_downloader.py
 requirements.txt
 Dockerfile
 docker-compose.yml
 .dockerignore
 .gitignore
 README.md
```

## Uso

1. Ejecutar el script
2. Aparecer frente a la camara
3. Hacer gesto Closed Fist (puno cerrado)
4. El sistema comenzara a rastrear tus movimientos
5. Presionar q para salir

## Tecnologias

- MediaPipe 0.10.14 - Deteccion de gestos y pose
- OpenCV 4.10 - Procesamiento de video
- Python 3.11 - Runtime
- NumPy 1.26 - Operaciones numericas

## Modelos

Dos modelos de pose disponibles:

- lite (default): Mas rapido, menos preciso
- full: Mas preciso, mas lento

Los modelos se descargan automaticamente en la primera ejecucion.

## Notas

- Python 3.12 no compatible (elimina distutils requerido por MediaPipe)
- En Windows, usar ejecucion local (Docker Desktop no soporta camara nativamente)
- Requiere buena iluminacion para deteccion optima de gestos

## Licencia

MIT
