# Python 3.11 (3.12 elimina distutils requerido por mediapipe)
FROM python:3.11-slim-bookworm

# Establecer variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# Instalar dependencias del sistema necesarias para OpenCV y MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Dependencias de OpenCV
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # Dependencias para GUI (X11)
    libx11-6 \
    libxcb1 \
    libxau6 \
    libxdmcp6 \
    # Herramientas útiles
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /app

# Copiar requirements y instalar dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la aplicación
COPY person_tracker.py .
COPY model_downloader.py .

# Crear directorio para modelos (será montado como volumen)
RUN mkdir -p /app/models

# Exponer variable de entorno para display
ENV DISPLAY=:0

# Usuario no root para mayor seguridad
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

# Comando por defecto
CMD ["python", "person_tracker.py", "lite"]
