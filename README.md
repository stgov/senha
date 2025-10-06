# Person Tracker with Gesture Recognition

Sistema de seguimiento de personas activado por reconocimiento de gestos usando MediaPipe.

## 🚀 Características

- Detección de gesto "Closed Fist" para activar seguimiento
- Seguimiento específico de la persona que realiza el gesto
- Procesamiento paralelo de modelos (Gesture Recognizer + Pose Landmarker)
- Soporte para modelos Lite y Full
- Descarga automática de modelos
- Dockerizado con soporte CPU y GPU

## 📋 Requisitos

### Docker (Recomendado)

- Docker Engine 20.10+
- Docker Compose 2.0+
- Cámara web accesible

### Para GPU (Opcional)

**NVIDIA CUDA:**
- NVIDIA GPU con soporte CUDA
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

**AMD ROCm:**
- AMD GPU con soporte ROCm (RX 5000+, RX 6000+, RX 7000+)
- [ROCm instalado](https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html)
- Kernel Linux 5.15+

### Local (Alternativo)

- Python 3.10+
- Cámara web

## 🐳 Inicio Rápido con Docker

### Linux/macOS

```bash
# 1. Clonar repositorio
git clone <repo-url>
cd senha

# 2. Permitir acceso a X11
xhost +local:docker

# 3. Construir y ejecutar
docker-compose up person-tracker

# 4. Con modelo Full
docker-compose run --rm person-tracker python person_tracker.py full
```

### Windows con WSL2

```powershell
# 1. Abrir WSL2
wsl

# 2. Configurar display
export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0

# 3. Permitir X11
xhost +local:docker

# 4. Ejecutar
docker-compose up person-tracker
```

Ver [WINDOWS.md](WINDOWS.md) para más detalles sobre configuración en Windows.

Ver [ROCM.md](ROCM.md) para guía completa de configuración AMD ROCm.

## 🎮 GPU Acelerado

### NVIDIA CUDA

```bash
# 1. Verificar GPU disponible
docker run --rm --gpus all nvidia/cuda:12.3.1-base-ubuntu22.04 nvidia-smi

# 2. Construir imagen GPU
docker-compose -f docker-compose.gpu.yml build

# 3. Ejecutar con GPU
docker-compose -f docker-compose.gpu.yml up person-tracker-gpu
```

### AMD ROCm

```bash
# 1. Verificar GPU AMD disponible
docker run --rm --device=/dev/kfd --device=/dev/dri rocm/pytorch:latest rocm-smi

# 2. Construir imagen ROCm
docker-compose build person-tracker-rocm

# 3. Ejecutar con GPU AMD
docker-compose up person-tracker-rocm

# 4. Con modelo Full
docker-compose run --rm person-tracker-rocm python person_tracker.py full
```

**GPUs AMD soportadas:**
- RX 5000 series (RDNA 1)
- RX 6000 series (RDNA 2)
- RX 7000 series (RDNA 3)
- Radeon Pro series
- Instinct series

## 💻 Ejecución Local (Sin Docker)

```bash
# 1. Crear entorno virtual
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar (descarga automática de modelos)
python person_tracker.py lite
```

## 🎯 Uso

1. Ejecuta la aplicación
2. Colócate frente a la cámara
3. Haz el gesto de puño cerrado ("Closed Fist")
4. El sistema identificará tu posición y activará el seguimiento
5. Solo tú serás rastreado, otras personas en el frame serán ignoradas

### Controles

- **q**: Salir de la aplicación
- **r**: Resetear seguimiento (volver a modo detección)
- **t**: Alternar visualización de manos

## 📦 Modelos

Los modelos se descargan automáticamente la primera vez que ejecutas la aplicación.

Para descargar manualmente:

```bash
python model_downloader.py all
```

## 🛠️ Estructura del Proyecto

```
.
├── person_tracker.py           # Script principal
├── model_downloader.py         # Descargador automático de modelos
├── requirements.txt            # Dependencias locales (CPU)
├── requirements-docker.txt     # Dependencias Docker CPU
├── requirements-gpu.txt        # Dependencias Docker NVIDIA CUDA
├── requirements-rocm.txt       # Dependencias Docker AMD ROCm
├── Dockerfile                  # Imagen Docker CPU
├── Dockerfile.gpu              # Imagen Docker NVIDIA CUDA
├── Dockerfile.rocm             # Imagen Docker AMD ROCm
├── docker-compose.yml          # Orquestación unificada (CPU/NVIDIA/AMD)
├── .dockerignore               # Exclusiones Docker
├── .gitignore                  # Exclusiones Git
├── README.md                   # Este archivo
├── WINDOWS.md                  # Guía para Windows
└── ROCM.md                     # Guía para AMD ROCm
```

## 🐛 Solución de Problemas

### Docker: "Can't open display"

```bash
# Linux/macOS
xhost +local:docker

# WSL2
export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0
```

### Docker: "No se pudo abrir la cámara"

Asegúrate de que el dispositivo esté montado correctamente:

```bash
# Verificar cámara
ls -l /dev/video*

# Agregar usuario al grupo video
sudo usermod -aG video $USER
```

### Local: "No se pudo abrir la cámara"

Cierra otras aplicaciones que puedan estar usando la cámara (Teams, Zoom, Skype).

## 📊 Rendimiento

| Configuración | FPS Estimado | Latencia | Uso |
|---------------|--------------|----------|-----|
| CPU + Lite    | 15-20 fps    | ~50ms    | General |
| CPU + Full    | 8-12 fps     | ~100ms   | Mayor precisión |
| NVIDIA + Lite | 50-60 fps    | ~15ms    | Tiempo real |
| NVIDIA + Full | 30-40 fps    | ~30ms    | Máxima precisión |
| AMD + Lite    | 45-55 fps    | ~18ms    | Tiempo real |
| AMD + Full    | 25-35 fps    | ~35ms    | Alta precisión |

**Nota:** Rendimiento estimado basado en:
- CPU: Intel i7/Ryzen 7
- NVIDIA: RTX 3060 o superior
- AMD: RX 6600 XT o superior

## 📝 Licencia

MIT
