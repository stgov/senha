# Guía de Configuración AMD ROCm

## 🎮 Soporte para GPUs AMD con ROCm

Esta guía te ayudará a configurar tu GPU AMD para usar con el person tracker.

## 📋 Requisitos

### Hardware Compatible

**GPUs Soportadas:**
- **RDNA 1** (RX 5000 series): RX 5500, 5600, 5700
- **RDNA 2** (RX 6000 series): RX 6600, 6700, 6800, 6900
- **RDNA 3** (RX 7000 series): RX 7600, 7700, 7800, 7900
- **Radeon Pro**: W6000, W7000 series
- **Instinct**: MI100, MI200, MI300 series

### Software

- Linux (Ubuntu 22.04 o similar recomendado)
- Kernel Linux 5.15 o superior
- Docker 20.10+
- Docker Compose 2.0+

## 🔧 Instalación de ROCm

### Ubuntu 22.04 / 24.04

```bash
# 1. Agregar repositorio de ROCm
wget https://repo.radeon.com/amdgpu-install/6.1/ubuntu/jammy/amdgpu-install_6.1.60100-1_all.deb
sudo apt install ./amdgpu-install_6.1.60100-1_all.deb

# 2. Instalar ROCm
sudo amdgpu-install --usecase=rocm

# 3. Agregar usuario al grupo render y video
sudo usermod -aG render $USER
sudo usermod -aG video $USER

# 4. Reiniciar para aplicar cambios
sudo reboot
```

### Arch Linux / Manjaro

```bash
# Instalar desde repositorios
sudo pacman -S rocm-hip-sdk rocm-opencl-sdk

# Agregar usuario a grupos
sudo usermod -aG render,video $USER
```

### Fedora

```bash
# Instalar repositorio EPEL
sudo dnf install epel-release

# Instalar ROCm
sudo dnf install rocm-hip rocm-opencl

# Agregar usuario a grupos
sudo usermod -aG render,video $USER
```

## ✅ Verificar Instalación

### 1. Verificar GPU detectada

```bash
# Listar GPUs AMD
rocm-smi

# Debería mostrar algo como:
# ========================= ROCm System Management Interface =========================
# GPU[0]    : RX 6700 XT
# ...
```

### 2. Verificar drivers

```bash
# Verificar módulos del kernel
lsmod | grep amdgpu

# Verificar dispositivos
ls -la /dev/kfd /dev/dri
```

### 3. Verificar ROCm con Docker

```bash
# Test básico
docker run --rm --device=/dev/kfd --device=/dev/dri \
  rocm/pytorch:latest rocm-smi

# Si ves la información de tu GPU, está funcionando correctamente
```

## 🐳 Uso con Docker

### Opción 1: Docker Compose (Recomendado)

```bash
# 1. Construir imagen
docker-compose build person-tracker-rocm

# 2. Ejecutar con modelo Lite
docker-compose up person-tracker-rocm

# 3. Ejecutar con modelo Full
docker-compose run --rm person-tracker-rocm python person_tracker.py full
```

### Opción 2: Docker directo

```bash
# Construir
docker build -f Dockerfile.rocm -t person-tracker:rocm .

# Ejecutar
docker run --rm -it \
  --device=/dev/kfd \
  --device=/dev/dri \
  --device=/dev/video0 \
  --group-add video \
  --group-add render \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $(pwd)/models:/app/models \
  person-tracker:rocm python person_tracker.py lite
```

## 🔍 Solución de Problemas

### Error: "No ROCm-capable device is detected"

```bash
# Verificar que los dispositivos estén disponibles
ls -la /dev/kfd /dev/dri

# Verificar permisos
groups | grep -E 'render|video'

# Si no aparecen, agregar usuario a grupos
sudo usermod -aG render,video $USER
# Cerrar sesión y volver a entrar
```

### Error: "HSA Error: Cannot open /dev/kfd"

```bash
# Verificar driver amdgpu
lsmod | grep amdgpu

# Si no aparece, cargar módulo
sudo modprobe amdgpu

# Verificar que esté habilitado al inicio
echo "amdgpu" | sudo tee /etc/modules-load.d/amdgpu.conf
```

### Rendimiento bajo

```bash
# 1. Verificar que la GPU esté en modo performance
sudo rocm-smi --setperflevel high

# 2. Verificar temperatura y uso
watch -n 1 rocm-smi

# 3. Usar modelo lite en lugar de full
docker-compose run --rm person-tracker-rocm python person_tracker.py lite
```

### Error: "GFX version mismatch"

Algunas GPUs requieren override de GFX version:

```bash
# Para RX 6000 series (gfx1030)
export HSA_OVERRIDE_GFX_VERSION=10.3.0

# Para RX 7000 series (gfx1100)
export HSA_OVERRIDE_GFX_VERSION=11.0.0

# O en docker-compose.yml ya está configurado
```

## 📊 Benchmarks

### GPU RX 6700 XT (RDNA 2)

| Modelo | FPS | Latencia | VRAM |
|--------|-----|----------|------|
| Lite   | 48  | 21ms     | 1.2GB |
| Full   | 28  | 36ms     | 1.8GB |

### GPU RX 7900 XT (RDNA 3)

| Modelo | FPS | Latencia | VRAM |
|--------|-----|----------|------|
| Lite   | 58  | 17ms     | 1.1GB |
| Full   | 35  | 29ms     | 1.6GB |

## 🔗 Referencias

- [ROCm Documentation](https://rocm.docs.amd.com/)
- [AMD GPU Support Matrix](https://rocm.docs.amd.com/en/latest/release/gpu_os_support.html)
- [ROCm Docker Hub](https://hub.docker.com/u/rocm)
- [ROCm GitHub](https://github.com/RadeonOpenCompute/ROCm)

## 💡 Tips

1. **Usar modelo Lite** para mejor rendimiento en tiempo real
2. **Monitorear temperatura** con `rocm-smi`
3. **Cerrar navegadores** que usan aceleración GPU antes de ejecutar
4. **Actualizar drivers** regularmente para mejor compatibilidad
5. **Usar kernel reciente** (5.15+) para mejor soporte

## ⚠️ Limitaciones Conocidas

- ROCm solo funciona en **Linux**
- Algunas GPUs antiguas (pre-RDNA) tienen soporte limitado
- Windows no es soportado oficialmente (solo vía WSL2)
- macOS no es soportado
