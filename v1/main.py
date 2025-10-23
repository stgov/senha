#!/usr/bin/env python3
"""
Sistema de Reconocimiento y Re-Identificación de Personas con Gestos (v1).

Este script inicia el sistema de tracking con Re-ID que:
- Detecta personas y les asigna IDs persistentes
- Re-identifica personas que salen y vuelven al frame
- Marca permanentemente personas que hacen el gesto Closed_Fist
"""

from v1.core.app import TrackerApp
from v1.core.config import TrackerConfig


def main():
    """Punto de entrada principal del sistema."""

    # Configuración del sistema
    config = TrackerConfig(
        pose_model="lite",              # Modelo de pose: "lite" o "full"
        camera_source=0, # Fuente de video (0 para cámara, ruta para archivo)
        frame_width=640,
        frame_height=480,
        min_confidence=0.5,              # Confianza mínima para detecciones
        num_poses=3,                     # Máximo de personas a detectar
        max_num_hands=6,                # Máximo de manos a detectar
        gesture_score_threshold=0.3,     # Umbral de confianza para gestos
        smoothing_dead_zone=15,          # Zona muerta para suavizado de bbox
        smoothing_factor=0.3,            # Factor de suavizado
        window_title="Person Re-ID Tracker [v1 - MediaPipe]"
    )

    # Iniciar aplicación
    app = TrackerApp(config)

    try:
        print("\n" + "="*60)
        print("  Sistema de Re-Identificación de Personas v1")
        print("="*60)
        print("\n📋 Funcionalidades:")
        print("  • Re-ID: Identifica personas que salen y vuelven al frame")
        print("  • Closed_Fist: Marca permanentemente a la persona")
        print("  • Base de datos vectorial en memoria (ChromaDB)")
        print("  • Detección: MediaPipe Pose Landmarker + Hand Gesture")
        print("\n🎮 Controles:")
        print("  [H] - Toggle visualización de manos")
        print("  [P] - Toggle visualización de poses")
        print("  [Q] - Salir")
        print("\n" + "="*60 + "\n")
        
        app.run()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrumpido por el usuario")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        raise
    finally:
        print("\n✅ Sistema finalizado correctamente\n")


if __name__ == "__main__":
    main()
