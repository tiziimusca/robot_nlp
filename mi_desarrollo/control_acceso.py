"""
=====================================================================
 Control de Acceso y Verificación de EPP en Tiempo Real
 
 Permite al usuario mostrar a la cámara si tiene guantes u otros
 elementos de protección (gafas, guardapolvo), emitiendo en tiempo real:
   🛑 STOP / ACCESO DENEGADO (cuando faltan elementos)
   🟢 PUEDE PASAR / ACCESO CONCEDIDO (cuando porta los elementos requeridos)
 
 Opciones de uso:
   python control_acceso.py                  # Modo interactivo con cámara web
   python control_acceso.py --solo-guantes   # Inspeccionar únicamente guantes
   python control_acceso.py --con-robot      # Enviar acciones al simulador/robot
   python control_acceso.py --camara 0       # Seleccionar índice de cámara
=====================================================================
"""

import sys
import time
import argparse
import threading
import queue
from pathlib import Path
import cv2
import numpy as np

DIR_ACTUAL = Path(__file__).resolve().parent
DIR_ENTORNO = str(DIR_ACTUAL.parent / "entorno")
if DIR_ENTORNO not in sys.path:
    sys.path.insert(0, DIR_ENTORNO)

# Importar evaluador de EPP y cliente Roboflow
from evaluador_epp import EvaluadorSeguridadEPP, DATASET_DIR
from roboflow_detector import ClienteRoboflowAsync

ROBOT_DISPONIBLE = False
try:
    from sim.robot import Robot
    ROBOT_DISPONIBLE = True
except Exception as _err:
    print(f"  [AVISO] No se pudo importar sim.robot: {_err}")
    ROBOT_DISPONIBLE = False


class RobotManager:
    """Administrador asíncrono de gestos y comandos para el robot sin congelar el video."""

    def __init__(self, conectar=True):
        self.robot = None
        self.conectado = False
        self.cola_comandos = queue.Queue(maxsize=5)
        self.activo = True
        self.hilo = None

        if conectar and ROBOT_DISPONIBLE:
            self._iniciar_conexion()

    def _iniciar_conexion(self):
        try:
            print("  [ROBOT] Conectando con el simulador de Unitree en 127.0.0.1...")
            self.robot = Robot()
            estado = self.robot.conectar()
            self.conectado = True
            print(f"  [ROBOT] CONEXIÓN EXITOSA con {self.robot.modelo.upper()}. Estado inicial: {estado}")
            self.hilo = threading.Thread(target=self._bucle_robot, daemon=True)
            self.hilo.start()
        except Exception as e:
            print(f"  [ROBOT AVISO] Simulador no detectado ({e}).")
            print("  [ROBOT AVISO] Para ver el movimiento 3D, abra INICIAR_SIMULADOR.bat primero.")
            self.robot = None
            self.conectado = False

    def _bucle_robot(self):
        """Ejecuta los gestos del robot en segundo plano."""
        while self.activo:
            try:
                comando = self.cola_comandos.get(timeout=0.2)
            except queue.Empty:
                continue

            if not self.robot or not self.conectado:
                continue

            try:
                if comando == "SALUDAR_Y_PASAR":
                    print("  [ROBOT GESTO] >>> Ejecutando saludo y dando paso (torso inclinado y brazos al costado)...")
                    try:
                        self.robot.dar_paso()
                    except Exception as _e:
                        print(f"  [ROBOT AVISO] Dar paso: {_e}")
                elif comando == "ALTO":
                    print("  [ROBOT ACCIÓN] >>> Señal de ALTO: Brazo derecho extendido y palma levantada.")
                    try:
                        self.robot.hacer_alto()
                    except Exception as _e:
                        print(f"  [ROBOT AVISO] Alto: {_e}")
            except Exception as e:
                print(f"  [ROBOT ERROR] {e}")

    def ordenar_saludo(self):
        """Pide al robot hacer el gesto de dar paso (sin bloquear el hilo principal)."""
        if self.conectado:
            while not self.cola_comandos.empty():
                try:
                    self.cola_comandos.get_nowait()
                except Exception:
                    break
            try:
                self.cola_comandos.put_nowait("SALUDAR_Y_PASAR")
            except Exception:
                pass

    def ordenar_alto(self):
        """Pide al robot ponerse en alto."""
        if self.conectado:
            while not self.cola_comandos.empty():
                try:
                    self.cola_comandos.get_nowait()
                except Exception:
                    break
            try:
                self.cola_comandos.put_nowait("ALTO")
            except Exception:
                pass

    def cerrar(self):
        self.activo = False
        if self.robot and self.conectado:
            try:
                self.robot.detenerse()
                self.robot.desconectar()
            except Exception:
                pass


class ControlAccesoVisual:
    """Controlador de acceso interactivo con detección de guantes a cualquier distancia y robot."""

    def __init__(self, conectar_robot=True, device_id=0, usar_roboflow=True):
        self.device_id = device_id
        self.evaluador = EvaluadorSeguridadEPP()
        self.robot_mgr = RobotManager(conectar=conectar_robot)
        self.roboflow = ClienteRoboflowAsync(activo=usar_roboflow)

        # Temporizadores de histéresis anti-parpadeo
        self.TIEMPO_CONFIRMACION_PASAR = 1.2   # Segundos continuos con ambos guantes para autorizar
        self.TIEMPO_CONFIRMACION_ALTO = 0.8    # Segundos sin guantes para volver a ALTO
        
        self.tiempo_inicio_deteccion = 0.0
        self.tiempo_inicio_falta = 0.0
        self.estado_autorizado = False
        self.progreso_verificacion = 0.0

    def procesar_camara(self):
        print("\n" + "=" * 75)
        print("  SISTEMA DE CONTROL DE ACCESO EPP - DETECCIÓN DE AMBOS GUANTES")
        print("=" * 75)
        print("  Motor de Visión : Híbrido (Roboflow Cloud + Clasificador Local 3-Clases)")
        print("  REGLA           : Debe mostrar AMBAS manos con guantes durante 1.2s.")
        print(f"  Robot conectado : {'SÍ (Gestos activos)' if self.robot_mgr.conectado else 'NO (Simulación en pantalla)'}")
        print("\n  Comandos:")
        print("    [ESC] o [q] : Salir")
        print("    [g]         : Guardar captura 'CON GUANTE'")
        print("    [s]         : Guardar captura 'SIN GUANTE'")
        print("=" * 75)

        cap = cv2.VideoCapture(self.device_id)
        if not cap.isOpened():
            print(f"\n  [ERROR] No se pudo abrir la cámara {self.device_id}.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        prev_time = time.time()
        self.robot_mgr.ordenar_alto()
        contador_frames = 0
        rf_activo_global = False

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Efecto espejo
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            ahora = time.time()

            # Enviar fotograma a Roboflow en segundo plano cada 3 frames
            if contador_frames % 3 == 0:
                self.roboflow.enviar_frame(frame)
            contador_frames += 1

            # Obtener detecciones recientes de la nube (no bloqueante)
            preds_rf, rf_activo = self.roboflow.obtener_predicciones()
            if rf_activo:
                rf_activo_global = True

            # 1. Evaluar AMBAS manos (combina nube + clasificador local)
            res_manos = self.evaluador.evaluar_ambas_manos(frame, roboflow_preds=preds_rf)
            ambos_guantes_presentes = res_manos["cumple_protocolo"]

            # 2. Máquina de Estados con Histéresis Temporal (Anti-Flicker)
            if not self.estado_autorizado:
                # Estamos en ALTO: ¿Vemos ambos guantes de forma continua?
                if ambos_guantes_presentes:
                    if self.tiempo_inicio_deteccion == 0.0:
                        self.tiempo_inicio_deteccion = ahora
                    tiempo_transcurrido = ahora - self.tiempo_inicio_deteccion
                    self.progreso_verificacion = min(1.0, tiempo_transcurrido / self.TIEMPO_CONFIRMACION_PASAR)

                    if tiempo_transcurrido >= self.TIEMPO_CONFIRMACION_PASAR:
                        self.estado_autorizado = True
                        self.progreso_verificacion = 1.0
                        print("\n  >>> 🟢 [PUEDE PASAR] ¡Ambos guantes verificados! El robot saluda y da paso.")
                        self.robot_mgr.ordenar_saludo()
                else:
                    self.tiempo_inicio_deteccion = 0.0
                    self.progreso_verificacion = 0.0
            else:
                # Estamos en PUEDE PASAR: ¿Se quitaron los guantes o bajaron las manos?
                if not ambos_guantes_presentes:
                    if self.tiempo_inicio_falta == 0.0:
                        self.tiempo_inicio_falta = ahora
                    if (ahora - self.tiempo_inicio_falta) >= self.TIEMPO_CONFIRMACION_ALTO:
                        self.estado_autorizado = False
                        self.tiempo_inicio_deteccion = 0.0
                        self.tiempo_inicio_falta = 0.0
                        self.progreso_verificacion = 0.0
                        print(f"\n  >>> 🛑 [ALTO] Acceso interrumpido: {res_manos['mensaje']}")
                        self.robot_mgr.ordenar_alto()
                else:
                    self.tiempo_inicio_falta = 0.0

            # 3. Dibujar Interfaz Gráfica (HUD)
            frame_hud = self.dibujar_interfaz(frame, res_manos, self.estado_autorizado, self.progreso_verificacion, rf_activo=rf_activo_global)

            # FPS
            fps = 1.0 / (ahora - prev_time + 1e-6)
            prev_time = ahora
            cv2.putText(frame_hud, f"FPS: {fps:.1f}", (w - 100, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

            cv2.imshow("Control de Acceso EPP - Unitree G1 / Go2", frame_hud)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('g'):
                self._guardar_foto(frame, "con_guante")
            elif key == ord('s'):
                self._guardar_foto(frame, "sin_guante")

        cap.release()
        cv2.destroyAllWindows()
        self.roboflow.cerrar()
        self.robot_mgr.cerrar()

    def _guardar_foto(self, frame, tag):
        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        p = DATASET_DIR / f"captura_{tag}_{int(time.time())}.jpg"
        cv2.imwrite(str(p), frame)
        print(f"  [CAPTURADO] {p.name}")

    def dibujar_interfaz(self, frame, res_manos, autorizado, progreso, rf_activo=False):
        out = frame.copy()
        h, w, _ = out.shape

        color_estado = (0, 220, 0) if autorizado else (0, 0, 240)

        # 1. Borde perimetral
        cv2.rectangle(out, (0, 0), (w, h), color_estado, 5)

        # 2. Cartel Superior Prominente (Banner)
        overlay = out.copy()
        cv2.rectangle(overlay, (0, 0), (w, 85), color_estado, -1)
        cv2.addWeighted(overlay, 0.85, out, 0.15, 0, out)

        if autorizado:
            cv2.putText(out, "PUEDE PASAR - ACCESO CONCEDIDO", (25, 40), cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2)
            cv2.putText(out, "[OK] Ambos guantes verificados. Robot ejecutando saludo.", (25, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
        else:
            cv2.putText(out, "ALTO - NO PUEDE PASAR", (25, 40), cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2)
            cv2.putText(out, f"{res_manos['mensaje']}", (25, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)

        # 3. Barra de Progreso de Verificación
        if not autorizado and progreso > 0.0:
            bar_w = int(w * 0.70)
            bar_h = 16
            bar_x = int(w * 0.15)
            bar_y = 95
            cv2.rectangle(out, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
            cv2.rectangle(out, (bar_x, bar_y), (bar_x + int(bar_w * progreso), bar_y + bar_h), (0, 220, 0), -1)
            cv2.rectangle(out, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 1)
            cv2.putText(out, f"Confirmando guantes: {int(progreso * 100)}%", (bar_x + 10, bar_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)

        # 4. Zonas de Inspección: Mano Izquierda y Mano Derecha
        rois = res_manos.get("rois", {})
        x1_izq, y1_izq, x2_izq, y2_izq = rois.get("izq", (int(w * 0.04), int(h * 0.15), int(w * 0.48), int(h * 0.90)))
        x1_der, y1_der, x2_der, y2_der = rois.get("der", (int(w * 0.52), int(h * 0.15), int(w * 0.96), int(h * 0.90)))

        res_izq = res_manos["mano_izquierda"]
        res_der = res_manos["mano_derecha"]

        # Mano Izquierda
        c_izq = (0, 220, 0) if res_izq["tiene_guante"] else ((0, 0, 230) if res_izq["detectado"] else (130, 130, 130))
        cv2.rectangle(out, (x1_izq, y1_izq), (x2_izq, y2_izq), c_izq, 2)
        cv2.putText(out, "MANO IZQUIERDA", (x1_izq + 10, y1_izq + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c_izq, 2)
        txt_izq = "[OK] CON GUANTE" if res_izq["tiene_guante"] else ("[X] MANO DESNUDA" if res_izq["detectado"] else "Coloque mano")
        cv2.putText(out, txt_izq, (x1_izq + 10, y2_izq - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c_izq, 2)

        bbox_izq = res_izq.get("bbox")
        if bbox_izq:
            bx, by, bw, bh = bbox_izq
            cv2.rectangle(out, (x1_izq + bx, y1_izq + by), (x1_izq + bx + bw, y1_izq + by + bh), c_izq, 2)

        # Mano Derecha
        c_der = (0, 220, 0) if res_der["tiene_guante"] else ((0, 0, 230) if res_der["detectado"] else (130, 130, 130))
        cv2.rectangle(out, (x1_der, y1_der), (x2_der, y2_der), c_der, 2)
        cv2.putText(out, "MANO DERECHA", (x1_der + 10, y1_der + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c_der, 2)
        txt_der = "[OK] CON GUANTE" if res_der["tiene_guante"] else ("[X] MANO DESNUDA" if res_der["detectado"] else "Coloque mano")
        cv2.putText(out, txt_der, (x1_der + 10, y2_der - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c_der, 2)

        bbox_der = res_der.get("bbox")
        if bbox_der:
            bx, by, bw, bh = bbox_der
            cv2.rectangle(out, (x1_der + bx, y1_der + by), (x1_der + bx + bw, y1_der + by + bh), c_der, 2)

        # 5. Pie de página
        cv2.rectangle(out, (0, h - 26), (w, h), (25, 25, 25), -1)
        tag_vision = "ROBOFLOW CLOUD (Activo)" if rf_activo else "ML LOCAL (Hibrido)"
        ayuda = f"Vision: {tag_vision} | Robot: {'CONECTADO (Gestos Activos)' if self.robot_mgr.conectado else 'SIMULACION'} | [Q] Salir"
        cv2.putText(out, ayuda, (15, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

        return out


def main():
    parser = argparse.ArgumentParser(description="Control de Acceso Visual EPP (AMBOS GUANTES REQUERIDOS)")
    parser.add_argument("--camara", type=int, default=0, help="Índice de la cámara (por defecto 0)")
    parser.add_argument("--con-robot", action="store_true", default=True, help="Conectar con el simulador del robot (por defecto True)")
    parser.add_argument("--sin-robot", action="store_true", help="Desactivar conexión con el robot")
    parser.add_argument("--sin-roboflow", action="store_true", help="Desactivar inferencia en la nube de Roboflow")
    args = parser.parse_args()

    conectar_robot = not args.sin_robot
    usar_roboflow = not args.sin_roboflow

    controlador = ControlAccesoVisual(
        conectar_robot=conectar_robot,
        device_id=args.camara,
        usar_roboflow=usar_roboflow
    )
    controlador.procesar_camara()


if __name__ == "__main__":
    main()
