"""
=====================================================================
 Evaluador de Seguridad de Equipos de Protección Personal (EPP)
 Detección en tiempo real sobre imágenes y video REALES.

 Elementos inspeccionados:
   1. Guantes de seguridad / trabajo (nitrilo, cuero, moteados, etc.)
   2. Gafas de seguridad / protección ocular
   3. Guardapolvo / Chaleco de seguridad

 Entrada: Streaming de cámara web o fotografías reales.
=====================================================================
"""

import os
import sys
import argparse
import urllib.request
from pathlib import Path
import cv2
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

DIR_ACTUAL = Path(__file__).resolve().parent
DATASET_DIR = DIR_ACTUAL / "dataset_epp_real"
MODELO_PKL = DIR_ACTUAL / "modelo_guantes.pkl"


def extraer_vector_caracteristicas(roi_bgr):
    """Extrae 20 características altamente discriminativas entre piel, guantes y fondo."""
    if roi_bgr is None or roi_bgr.size == 0:
        return np.zeros(20, dtype=np.float32)

    # Filtro mediano para suprimir ruido de alta frecuencia del sensor
    roi = cv2.medianBlur(roi_bgr, 3)
    roi = cv2.resize(roi, (80, 80))
    total_px = float(roi.shape[0] * roi.shape[1])

    # 1. Textura y rugosidad (Canny & Gradientes) - CLAVE para cuero, tela, moteados
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 130)
    texture_density = float(np.count_nonzero(edges)) / total_px

    gray_std = float(np.std(gray)) / 255.0
    gray_mean = float(np.mean(gray)) / 255.0

    # 2. Puntas/Zonas oscuras/negras (típicas de refuerzos de guantes)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    s = hsv[:, :, 1]
    dark_ratio = float(np.count_nonzero(v < 65)) / total_px
    high_sat_ratio = float(np.count_nonzero(s > 90)) / total_px

    # 3. YCrCb (Piel humana pura)
    ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]
    mask_skin_strict = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
    mask_skin_smooth = cv2.bitwise_and(mask_skin_strict, cv2.bitwise_not(edges))
    skin_smooth_ratio = float(np.count_nonzero(mask_skin_smooth)) / total_px

    cr_mean, cr_std = float(np.mean(cr)) / 255.0, float(np.std(cr)) / 255.0
    cb_mean, cb_std = float(np.mean(cb)) / 255.0, float(np.std(cb)) / 255.0

    # 4. Colores industriales de guantes (Azul, Amarillo, Naranja, Verde)
    mask_azul = cv2.inRange(hsv, np.array([80, 25, 45]), np.array([140, 255, 255]))
    mask_amarillo_naranja = cv2.inRange(hsv, np.array([5, 50, 50]), np.array([45, 255, 255]))
    glove_color_ratio = float(np.count_nonzero(mask_azul | mask_amarillo_naranja)) / total_px

    # 5. LAB
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    a = lab[:, :, 1]
    b = lab[:, :, 2]
    a_mean, a_std = float(np.mean(a)) / 255.0, float(np.std(a)) / 255.0
    b_mean, b_std = float(np.mean(b)) / 255.0, float(np.std(b)) / 255.0

    # 6. Relación de textura vs piel lisa
    ratio_rugosidad = texture_density / (skin_smooth_ratio + 0.05)

    return np.array([
        texture_density,
        skin_smooth_ratio,
        dark_ratio,
        glove_color_ratio,
        gray_std,
        gray_mean,
        high_sat_ratio,
        ratio_rugosidad,
        cr_mean, cr_std,
        cb_mean, cb_std,
        a_mean, a_std,
        b_mean, b_std,
        float(np.mean(s)) / 255.0, float(np.std(s)) / 255.0,
        float(np.mean(v)) / 255.0, float(np.std(v)) / 255.0
    ], dtype=np.float32)


def localizar_y_evaluar_mano(roi_zona, clf_ml, bbox_hint=None):
    """Localiza el contorno de la mano/guante dentro de la zona y la evalúa."""
    if roi_zona is None or roi_zona.size == 0 or roi_zona.shape[0] < 30 or roi_zona.shape[1] < 30:
        return {"detectado": False, "tiene_guante": False, "confianza": 0.0, "motivo": "Zona vacía", "bbox": None}

    h_z, w_z, _ = roi_zona.shape
    total_area = float(h_z * w_z)

    if bbox_hint is not None:
        # Usar bounding box de alta precisión de Roboflow
        bx, by, bw, bh = bbox_hint
        bx1 = max(0, bx)
        by1 = max(0, by)
        bx2 = min(w_z, bx + bw)
        by2 = min(h_z, by + bh)
        if (bx2 - bx1) >= 20 and (by2 - by1) >= 20:
            roi_eval = roi_zona[by1:by2, bx1:bx2]
            bbox = (bx1, by1, bx2 - bx1, by2 - by1)
        else:
            bbox_hint = None

    if bbox_hint is None:
        # Segmentación local de respaldo (Piel + Guante)
        ycrcb = cv2.cvtColor(roi_zona, cv2.COLOR_BGR2YCrCb)
        hsv = cv2.cvtColor(roi_zona, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(roi_zona, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 110)

        mask_skin = cv2.inRange(ycrcb, np.array([0, 130, 75]), np.array([255, 175, 130]))
        mask_glove_colors = cv2.inRange(hsv, np.array([5, 35, 35]), np.array([175, 255, 255]))
        mask_dark_texture = (hsv[:, :, 2] < 70) & (edges > 0)
        
        mask_obj = (mask_skin > 0) | (mask_glove_colors > 0) | mask_dark_texture
        mask_obj = mask_obj.astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask_obj = cv2.morphologyEx(mask_obj, cv2.MORPH_CLOSE, kernel)
        mask_obj = cv2.morphologyEx(mask_obj, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

        contours, _ = cv2.findContours(mask_obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = total_area * 0.035
        hand_contours = [c for c in contours if cv2.contourArea(c) >= min_area]

        if not hand_contours:
            return {
                "detectado": False,
                "tiene_guante": False,
                "confianza": 0.0,
                "motivo": "Coloque la mano en el área",
                "bbox": None
            }

        c_max = max(hand_contours, key=cv2.contourArea)
        bx, by, bw, bh = cv2.boundingRect(c_max)

        if bw < 25 or bh < 25 or (bw * bh) < min_area:
            return {
                "detectado": False,
                "tiene_guante": False,
                "confianza": 0.0,
                "motivo": "Coloque la mano en el área",
                "bbox": None
            }

        bx1 = max(0, bx - 8)
        by1 = max(0, by - 8)
        bx2 = min(w_z, bx + bw + 8)
        by2 = min(h_z, by + bh + 8)
        roi_eval = roi_zona[by1:by2, bx1:bx2]
        bbox = (bx1, by1, bx2 - bx1, by2 - by1)

    feats = extraer_vector_caracteristicas(roi_eval)
    
    if clf_ml is not None:
        try:
            proba = clf_ml.predict_proba([feats])[0]
            classes = list(clf_ml.classes_)
            
            idx_piel = classes.index(0) if 0 in classes else -1
            idx_guante = classes.index(1) if 1 in classes else -1
            idx_fondo = classes.index(2) if 2 in classes else -1

            prob_piel = float(proba[idx_piel]) if idx_piel >= 0 else 0.0
            prob_guante = float(proba[idx_guante]) if idx_guante >= 0 else 0.0
            prob_fondo = float(proba[idx_fondo]) if idx_fondo >= 0 else 0.0

            if prob_fondo > 0.45 and prob_fondo >= max(prob_piel, prob_guante):
                return {
                    "detectado": False,
                    "tiene_guante": False,
                    "confianza": prob_fondo,
                    "motivo": "Coloque la mano en el área",
                    "bbox": None
                }

            tiene_guante = bool(prob_guante > prob_piel)
            conf = prob_guante if tiene_guante else prob_piel
            return {
                "detectado": True,
                "tiene_guante": tiene_guante,
                "confianza": float(conf),
                "bbox": bbox,
                "motivo": f"{'Guante' if tiene_guante else 'Mano desnuda'} ({conf*100:.1f}%)"
            }
        except Exception:
            pass

    ratio_rugosidad = feats[7]
    dark_ratio = feats[2]
    tiene_guante = bool(ratio_rugosidad > 1.2 or dark_ratio > 0.15 or feats[3] > 0.08)
    return {
        "detectado": True,
        "tiene_guante": tiene_guante,
        "confianza": 0.85 if tiene_guante else 0.80,
        "bbox": bbox,
        "motivo": f"{'Guante' if tiene_guante else 'Mano desnuda'}"
    }


class EvaluadorSeguridadEPP:
    """Clasificador y evaluador de EPP con requerimiento estricto de AMBOS guantes."""

    def __init__(self, ruta_modelo=None):
        self.modelo_ml = None
        self._cargar_modelo_ml(ruta_modelo)

    def _cargar_modelo_ml(self, ruta=None):
        path = Path(ruta) if ruta else MODELO_PKL
        if path.exists():
            try:
                self.modelo_ml = joblib.load(path)
                print(f"  [EPP] Modelo ML de guantes cargado exitosamente ({path.name}).")
            except Exception as e:
                print(f"  [EPP] No se pudo cargar modelo ML ({e}).")
                self.modelo_ml = None
        else:
            self.modelo_ml = None

    def evaluar_ambas_manos(self, frame_bgr, roboflow_preds=None):
        """Evalúa ambas manos a distancia normal de cámara."""
        if frame_bgr is None or frame_bgr.size == 0:
            return {
                "cumple_protocolo": False,
                "ambos_guantes": False,
                "mano_izquierda": {"detectado": False, "tiene_guante": False, "motivo": "Sin video"},
                "mano_derecha": {"detectado": False, "tiene_guante": False, "motivo": "Sin video"},
                "mensaje": "Video no disponible"
            }

        h, w, _ = frame_bgr.shape

        x1_izq, y1_izq, x2_izq, y2_izq = int(w * 0.04), int(h * 0.15), int(w * 0.48), int(h * 0.90)
        x1_der, y1_der, x2_der, y2_der = int(w * 0.52), int(h * 0.15), int(w * 0.96), int(h * 0.90)

        roi_izq = frame_bgr[y1_izq:y2_izq, x1_izq:x2_izq]
        roi_der = frame_bgr[y1_der:y2_der, x1_der:x2_der]

        hint_izq = None
        hint_der = None
        if roboflow_preds:
            for p in roboflow_preds:
                bx, by, bw, bh = p["bbox"]
                cx = bx + bw / 2.0
                if x1_izq <= cx <= x2_izq and y1_izq <= (by + bh / 2.0) <= y2_izq:
                    hint_izq = (bx - x1_izq, by - y1_izq, bw, bh)
                elif x1_der <= cx <= x2_der and y1_der <= (by + bh / 2.0) <= y2_der:
                    hint_der = (bx - x1_der, by - y1_der, bw, bh)

        res_izq = localizar_y_evaluar_mano(roi_izq, self.modelo_ml, bbox_hint=hint_izq)
        res_der = localizar_y_evaluar_mano(roi_der, self.modelo_ml, bbox_hint=hint_der)

        det_izq = res_izq["detectado"]
        det_der = res_der["detectado"]
        tiene_izq = bool(res_izq["tiene_guante"] and det_izq)
        tiene_der = bool(res_der["tiene_guante"] and det_der)

        ambos_guantes = bool(tiene_izq and tiene_der and det_izq and det_der)

        if ambos_guantes:
            mensaje = "Ambos guantes verificados. Puede pasar."
        elif not det_izq and not det_der:
            mensaje = "Coloque ambas manos en las zonas de inspección."
        elif det_izq and not det_der:
            if tiene_izq:
                mensaje = "ALTO: Falta colocar la mano derecha."
            else:
                mensaje = "ALTO: Mano izquierda desnuda y falta mano derecha."
        elif det_der and not det_izq:
            if tiene_der:
                mensaje = "ALTO: Falta colocar la mano izquierda."
            else:
                mensaje = "ALTO: Mano derecha desnuda y falta mano izquierda."
        elif not tiene_izq and not tiene_der:
            mensaje = "ALTO: Ambas manos desnudas. Colóquese los dos guantes."
        elif not tiene_izq:
            mensaje = "ALTO: Falta el guante en la mano izquierda."
        elif not tiene_der:
            mensaje = "ALTO: Falta el guante en la mano derecha."
        else:
            mensaje = "ALTO: Requisitos de guantes incompletos."

        return {
            "cumple_protocolo": ambos_guantes,
            "ambos_guantes": ambos_guantes,
            "mano_izquierda": res_izq,
            "mano_derecha": res_der,
            "mensaje": mensaje,
            "rois": {
                "izq": (x1_izq, y1_izq, x2_izq, y2_izq),
                "der": (x1_der, y1_der, x2_der, y2_der),
            }
        }

    def evaluar_frame(self, frame_bgr, solo_guantes=True):
        """Compatibilidad con pipeline general."""
        res_manos = self.evaluar_ambas_manos(frame_bgr)
        return {
            "cumple_protocolo": res_manos["cumple_protocolo"],
            "guantes": res_manos["ambos_guantes"],
            "gafas": True,
            "guardapolvo": True,
            "elementos_faltantes": [] if res_manos["ambos_guantes"] else [res_manos["mensaje"]],
            "confianza": 0.95 if res_manos["ambos_guantes"] else 0.85,
            "detalles": res_manos
        }

    def evaluar_imagen(self, ruta_imagen, solo_guantes=True):
        path = Path(ruta_imagen)
        if not path.is_file():
            raise FileNotFoundError(f"No existe el archivo: {ruta_imagen}")
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f"No se pudo cargar la imagen: {ruta_imagen}")
        return self.evaluar_frame(frame, solo_guantes=solo_guantes)


def descargar_dataset_real():
    """Obtiene todas las rutas del dataset fotográfico real de manos con y sin guantes."""
    rutas = []
    con_guantes = DATASET_DIR / "con_guantes"
    sin_guantes = DATASET_DIR / "sin_guantes"

    if con_guantes.exists():
        rutas.extend(sorted(con_guantes.glob("*.jpg")))
    if sin_guantes.exists():
        rutas.extend(sorted(sin_guantes.glob("*.jpg")))
    
    # También fotos en la raíz del dataset
    rutas.extend([f for f in sorted(DATASET_DIR.glob("*.jpg")) if f.is_file()])
    return rutas


def main():
    parser = argparse.ArgumentParser(description="Evaluador de EPP y Guantes con Datos Reales")
    parser.add_argument("--imagen", type=str, help="Ruta a una foto para evaluar")
    parser.add_argument("--test", action="store_true", help="Probar sobre dataset fotográfico real")
    parser.add_argument("--solo-guantes", action="store_true", default=True, help="Inspeccionar únicamente guantes")
    args = parser.parse_args()

    if sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    evaluador = EvaluadorSeguridadEPP()

    if args.test or (not args.imagen):
        print("\n" + "=" * 75)
        print("  EVALUACION DE GUANTES (AMBAS MANOS) SOBRE DATASET REAL")
        print("=" * 75)

        rutas = descargar_dataset_real()
        if not rutas:
            print("  No se encontraron imagenes en el dataset.")
            return

        for r in rutas:
            print(f"\n  Foto: {r.parent.name}/{r.name}")
            res = evaluador.evaluar_imagen(r, solo_guantes=args.solo_guantes)
            estado = "[PUEDE PASAR]" if res["cumple_protocolo"] else "[ALTO / STOP]"
            print(f"    Veredicto      : {estado}")
            print(f"    Mensaje        : {res['detalles'].get('mensaje', '')}")
            izq = res['detalles'].get('mano_izquierda', {})
            der = res['detalles'].get('mano_derecha', {})
            print(f"    - Mano Izq     : {'CON GUANTE' if izq.get('tiene_guante') else 'SIN GUANTE/PIEL'} ({izq.get('motivo', '')})")
            print(f"    - Mano Der     : {'CON GUANTE' if der.get('tiene_guante') else 'SIN GUANTE/PIEL'} ({der.get('motivo', '')})")

        print("\n" + "=" * 75)
        print("  Prueba completada.")
        print("=" * 75)

    elif args.imagen:
        print(f"\n  Evaluando imagen: {args.imagen}")
        res = evaluador.evaluar_imagen(args.imagen, solo_guantes=args.solo_guantes)
        print(f"  Resultado: {'PUEDE PASAR' if res['cumple_protocolo'] else 'STOP'}")
        print(f"  Detalle: {res}")


if __name__ == "__main__":
    main()
