"""
=====================================================================
 Cliente Asíncrono de Inferencia en la Nube con Roboflow Serverless
 
 Procesa imágenes en segundo plano para obtener segmentación de manos
 y guantes con modelos de Deep Learning en la nube sin congelar el
 streaming en tiempo real de OpenCV.
=====================================================================
"""

import base64
import json
import threading
import time
import urllib.request
import urllib.error
import cv2
import numpy as np

DEFAULT_API_KEY = "eXuNHP4XxjR7LPk22RZ8"
DEFAULT_WORKFLOW_URL = "https://serverless.roboflow.com/nina-zeisel/workflows/general-segmentation-api"


class ClienteRoboflowAsync:
    """Cliente asíncrono para inferencia serverless en Roboflow."""

    def __init__(self, api_key: str = DEFAULT_API_KEY, url: str = DEFAULT_WORKFLOW_URL, activo: bool = True):
        self.api_key = api_key
        self.url = url
        self.activo = activo
        
        self._frame_pendiente = None
        self._bloqueo = threading.Lock()
        self._ultimo_resultado = []
        self._ultima_actualizacion = 0.0
        self._hilo = None
        self._ejecutando = False
        self._en_peticion = False
        self._disponible = False

        if self.activo and self.api_key:
            self._iniciar_hilo()

    def _iniciar_hilo(self):
        self._ejecutando = True
        self._hilo = threading.Thread(target=self._bucle_inferencia, daemon=True)
        self._hilo.start()

    def enviar_frame(self, frame_bgr):
        """Envía el fotograma más reciente a la cola de inferencia en la nube."""
        if not self._ejecutando:
            return
        with self._bloqueo:
            self._frame_pendiente = frame_bgr.copy()

    def obtener_predicciones(self, max_edad: float = 1.5):
        """Retorna las últimas predicciones si tienen menos de `max_edad` segundos."""
        ahora = time.time()
        with self._bloqueo:
            if (ahora - self._ultima_actualizacion) <= max_edad:
                return list(self._ultimo_resultado), self._disponible
            return [], self._disponible

    def _bucle_inferencia(self):
        while self._ejecutando:
            frame_a_procesar = None
            with self._bloqueo:
                if self._frame_pendiente is not None:
                    frame_a_procesar = self._frame_pendiente
                    self._frame_pendiente = None

            if frame_a_procesar is None:
                time.sleep(0.05)
                continue

            # Redimensionar para reducir ancho de banda y latencia
            h_orig, w_orig, _ = frame_a_procesar.shape
            frame_small = cv2.resize(frame_a_procesar, (480, 360))
            
            # Codificar en JPEG
            _, buffer = cv2.imencode(".jpg", frame_small, [cv2.IMWRITE_JPEG_QUALITY, 70])
            img_b64 = base64.b64encode(buffer).decode("utf-8")

            payload = {
                "inputs": {
                    "image": {
                        "type": "base64",
                        "value": img_b64
                    },
                    "classes": "hand"
                }
            }
            data_bytes = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.url,
                data=data_bytes,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                },
                method="POST"
            )

            try:
                self._en_peticion = True
                with urllib.request.urlopen(req, timeout=3.5) as resp:
                    body = resp.read().decode("utf-8")
                    data = json.loads(body)
                    outputs = data.get("outputs", [{}])[0]
                    raw_preds = outputs.get("predictions", {}).get("predictions", [])
                    
                    # Escalar coordenadas al tamaño original
                    scale_x = w_orig / 480.0
                    scale_y = h_orig / 360.0

                    predicciones_normalizadas = []
                    for p in raw_preds:
                        cx = p.get("x", 0.0) * scale_x
                        cy = p.get("y", 0.0) * scale_y
                        w = p.get("width", 0.0) * scale_x
                        h = p.get("height", 0.0) * scale_y
                        conf = p.get("confidence", 0.0)
                        clase = p.get("class", "").strip()

                        # Convertir centro a esquina superior izquierda
                        bx = max(0, int(cx - w / 2.0))
                        by = max(0, int(cy - h / 2.0))
                        bw = min(w_orig - bx, int(w))
                        bh = min(h_orig - by, int(h))

                        predicciones_normalizadas.append({
                            "class": clase,
                            "confidence": conf,
                            "bbox": (bx, by, bw, bh),
                            "center": (cx, cy)
                        })

                    with self._bloqueo:
                        self._ultimo_resultado = predicciones_normalizadas
                        self._ultima_actualizacion = time.time()
                        self._disponible = True

            except Exception as _e:
                with self._bloqueo:
                    self._disponible = False
            finally:
                self._en_peticion = False

            # Esperar un breve intervalo antes de la siguiente petición
            time.sleep(0.15)

    def cerrar(self):
        self._ejecutando = False
        if self._hilo:
            self._hilo.join(timeout=1.0)
