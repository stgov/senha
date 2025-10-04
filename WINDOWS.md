# Guía de Ejecución en Windows

## Opciones para ejecutar en Windows

### ✅ Opción 1: Ejecución Local (RECOMENDADO para Windows)

La forma más sencilla en Windows es ejecutar directamente con Python:

```powershell
# Activar entorno virtual
.\.venv\Scripts\Activate.ps1

# Ejecutar (descarga automática de modelos)
python person_tracker.py lite

# O con modelo full
python person_tracker.py full
```

### ⚙️ Opción 2: Docker Desktop en Windows (LIMITADO)

Docker Desktop en Windows tiene limitaciones con:
- Acceso a cámara web (requiere WSL2)
- Display de ventanas GUI (requiere X server)

**NO RECOMENDADO** a menos que uses WSL2.

### 🐧 Opción 3: WSL2 + Docker (AVANZADO)

Para usar Docker en Windows con acceso completo a hardware:

#### 1. Instalar WSL2

```powershell
# En PowerShell como Administrador
wsl --install
wsl --set-default-version 2

# Instalar Ubuntu
wsl --install -d Ubuntu
```

#### 2. Configurar en WSL2

```bash
# Abrir WSL2
wsl

# Instalar Docker en WSL2
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Reiniciar WSL2
exit
# Cerrar y abrir WSL2 nuevamente
```

#### 3. Instalar X Server en Windows

Opciones:
- **VcXsrv** (recomendado): https://sourceforge.net/projects/vcxsrv/
- **Xming**: https://sourceforge.net/projects/xming/
- **X410** (pago): Microsoft Store

#### 4. Configurar VcXsrv

1. Ejecutar XLaunch
2. Seleccionar "Multiple windows"
3. Display number: 0
4. Start no client
5. **IMPORTANTE**: Marcar "Disable access control"
6. Guardar configuración

#### 5. Configurar Display en WSL2

```bash
# En .bashrc o .zshrc
export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0
export LIBGL_ALWAYS_INDIRECT=1

# O configurar manualmente
export DISPLAY=172.x.x.x:0  # IP de Windows
```

#### 6. Permitir conexiones del firewall

```powershell
# En PowerShell como Administrador
New-NetFirewallRule -DisplayName "WSL2 X11" -Direction Inbound -LocalPort 6000 -Protocol TCP -Action Allow
```

#### 7. Copiar proyecto a WSL2

```bash
# En WSL2
cd ~
mkdir -p projects
cd projects

# Copiar desde Windows
cp -r /mnt/c/Users/santi/Documents/Codigo/senha .
cd senha
```

#### 8. Configurar cámara en WSL2

```bash
# Verificar cámara
ls -l /dev/video*

# Agregar usuario al grupo video
sudo usermod -a -G video $USER

# Puede requerir instalación de usbipd en Windows
# Ver: https://github.com/dorssel/usbipd-win
```

#### 9. Ejecutar Docker

```bash
# Construir
docker-compose build person-tracker

# Ejecutar
export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0
xhost +local:docker
docker-compose up person-tracker
```

## 🎯 Resumen de Recomendaciones

| Método | Dificultad | Funcionalidad | Recomendado |
|--------|-----------|---------------|-------------|
| Python Local | ⭐ Fácil | 100% | ✅ SÍ |
| Docker Desktop | ⭐⭐⭐ Difícil | 30% | ❌ NO |
| WSL2 + Docker | ⭐⭐⭐⭐ Muy Difícil | 95% | ⚠️ Solo si necesitas Docker |

## 🚀 Inicio Rápido (Recomendado)

```powershell
# 1. Clonar/Descargar proyecto
cd C:\Users\santi\Documents\Codigo\senha

# 2. Crear entorno virtual (si no existe)
python -m venv .venv

# 3. Activar entorno
.\.venv\Scripts\Activate.ps1

# 4. Instalar dependencias
pip install -r requirements.txt

# 5. Ejecutar (descarga automática de modelos)
python person_tracker.py lite
```

## ❓ Problemas Comunes

### Error: "No se pudo abrir la cámara"

```powershell
# Verificar que la cámara no esté en uso
# Cerrar Teams, Zoom, Skype, etc.

# Verificar permisos de cámara en Windows
# Configuración > Privacidad > Cámara
```

### Error: "ModuleNotFoundError"

```powershell
# Asegurarte de estar en el entorno virtual
.\.venv\Scripts\Activate.ps1

# Reinstalar dependencias
pip install --upgrade pip
pip install -r requirements.txt
```

### Rendimiento lento

```powershell
# Usar modelo lite en lugar de full
python person_tracker.py lite

# O cerrar otras aplicaciones
```

## 📞 Soporte

Si tienes problemas:
1. Usa la ejecución local (Python directo)
2. Verifica que la cámara funcione con otras apps
3. Asegúrate de tener Python 3.12+
4. Revisa los logs de error
