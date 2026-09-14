# =====================================================================
#  TP07 - Inteligencia Artificial
#  Agente que interpreta comandos en lenguaje natural
#
#  ESTE ES EL ARCHIVO DONDE ESCRIBIS TU PROGRAMA.
#
#  Antes de ejecutarlo:
#    1. Abri INICIAR_SIMULADOR (elegi G1 o Go2)
#    2. Espera a que aparezca la ventana con el robot
#    3. Recien ahi ejecuta este archivo
#
#  Nombre y apellido:  .....................................
#  Comision:           .....................................
# =====================================================================

from robot import Robot

from ejecutor import Ejecutor
from evaluar import evaluar

import re

# Pone tu nombre: aparece en el reporte que entregas.
ALUMNO = "Estudiante"


# =====================================================================
#  ETAPA 1 - CLASIFICADOR DE INTENCION
# =====================================================================
class ClasificadorIntencion:
    """Decide QUE quiere el usuario, sin mirar los numeros todavia."""

    TIPOS = ("MOVER", "GIRAR", "DETENERSE", "SALUDO",
             "CONSULTAR_ESTADO", "DESCONOCIDO")

    def __init__(self):
        self.modelo = None

    def clasificar(self, texto):
        t = texto.lower().strip()

        # Detenerse (prioridad para comandos de frenado)
        if re.search(r"\b(detente|deten|detén|para|pará|frena|frená|quieto|no avances|pará todo|stop)\b", t):
            return "DETENERSE"

        # Saludo
        if re.search(r"\b(saluda|saludo|hacé un saludo|hace un saludo)\b", t):
            return "SALUDO"

        # Consultar estado
        if re.search(r"\b(bateria|batería|estado|cómo estás|como estas)\b", t):
            return "CONSULTAR_ESTADO"

        # Girar (incluye comandos con gira/rota/media vuelta o giro explicito)
        if re.search(r"\b(girá|gira|rota|rotá|vuelta|media vuelta)\b", t):
            return "GIRAR"

        # Mover
        if re.search(r"\b(avanzá|avanza|movete|muévete|caminá|camina|adelante|retrocedé|retrocede|atrás|atras|andá|anda)\b", t):
            return "MOVER"

        # Direcciones de giro sin verbo explicito (ej: "a la izquierda")
        if re.search(r"\b(a la izquierda|a la derecha)\b", t):
            return "GIRAR"

        return "DESCONOCIDO"


# =====================================================================
#  ETAPA 2 - EXTRACTOR DE PARAMETROS
# =====================================================================
class ExtractorParametros:
    """Saca los numeros del texto. Sigue en unidades humanas."""

    def extraer(self, texto, tipo):
        params = {}
        t = texto.lower().strip()

        # Distancia en metros (ej: 2 metros, 100 metros, 0.5 metros, 1 metro)
        m_dist = re.search(r"(\d+(?:\.\d+)?)\s*(?:metros|metro|m)\b", t)
        if m_dist:
            params["distancia_m"] = float(m_dist.group(1))

        # Angulo en grados (ej: 90 grados, 270 grados, 45°)
        m_ang = re.search(r"(\d+(?:\.\d+)?)\s*(?:grados|grado|°)", t)
        if m_ang:
            val = float(m_ang.group(1))
            params["angulo_deg"] = int(val) if val.is_integer() else val
        elif "media vuelta" in t:
            params["angulo_deg"] = 180

        # Velocidad m/s (ej: a 2 m/s, 0.2 m/s) o adverbios (despacio, rápido)
        m_vel = re.search(r"(\d+(?:\.\d+)?)\s*(?:m/s|ms)\b", t)
        if m_vel:
            params["velocidad_ms"] = float(m_vel.group(1))
        elif re.search(r"\b(despacio|lento)\b", t):
            params["velocidad_ms"] = 0.2
        elif re.search(r"\b(rápido|rapido|veloz)\b", t):
            params["velocidad_ms"] = 0.5

        # Direccion (derecha, izquierda, atras, adelante)
        if "derecha" in t:
            params["direccion"] = "derecha"
        elif "izquierda" in t:
            params["direccion"] = "izquierda"
        elif re.search(r"\b(atrás|atras|retrocedé|retrocede)\b", t):
            params["direccion"] = "atras"
        elif re.search(r"\b(adelante|avanzá|avanza)\b", t):
            params["direccion"] = "adelante"

        return params


# =====================================================================
#  ETAPA 3 - VALIDADOR DE SEGURIDAD
# =====================================================================
class ValidadorSeguridad:
    """La ultima barrera antes del robot."""

    PALABRAS_PELIGROSAS = ("salta", "salto", "corre", "corré", "sprint",
                           "empuja", "empujá", "golpea", "rompe", "tira",
                           "cae", "fuerza")

    def __init__(self, perfil):
        self.perfil = perfil

    def validar(self, texto, tipo, parametros):
        t = texto.lower().strip()

        # 1. Palabras peligrosas en el texto original
        for p in self.PALABRAS_PELIGROSAS:
            if re.search(r"\b" + re.escape(p) + r"\b", t):
                return False, "palabra peligrosa"

        # 2. Velocidad pedida por encima del maximo permitido
        if "velocidad_ms" in parametros:
            if parametros["velocidad_ms"] > 0.5:
                return False, "velocidad > máximo"

        # 3. Distancia excesiva (> 5 metros)
        if "distancia_m" in parametros:
            if parametros["distancia_m"] > 5.0:
                return False, "distancia > máximo"

        # 4. Angulo mayor a 180 grados
        if "angulo_deg" in parametros:
            if parametros["angulo_deg"] > 180:
                return False, "ángulo > 180"

        return True, ""


# =====================================================================
#  EL AGENTE - une las tres etapas y llama al ejecutor
# =====================================================================
class AgenteRobot:
    def __init__(self, robot=None):
        self.robot = robot
        self.clasificador = ClasificadorIntencion()
        self.extractor = ExtractorParametros()
        self.validador = ValidadorSeguridad(
            robot.perfil if robot else _perfil_por_defecto())
        self.ejecutor = Ejecutor(robot) if robot else None
        self.historial = []

    def procesar(self, texto):
        tipo = self.clasificador.clasificar(texto)
        parametros = self.extractor.extraer(texto, tipo)
        es_seguro, motivo = self.validador.validar(texto, tipo, parametros)

        if not es_seguro:
            return {
                "tipo": tipo,
                "parametros": parametros,
                "ejecutar": False,
                "bloqueado": True,
                "confianza": 0.0,
                "texto_original": texto,
                "mensaje": motivo,
            }

        if tipo == "DESCONOCIDO":
            return {
                "tipo": "DESCONOCIDO",
                "parametros": {},
                "ejecutar": False,
                "bloqueado": False,
                "confianza": 0.0,
                "texto_original": texto,
                "mensaje": "fuera de dominio",
            }

        ejecutado = False
        if self.ejecutor is not None:
            try:
                self.ejecutor.ejecutar(tipo, parametros)
                ejecutado = True
            except Exception as e:
                return {
                    "tipo": tipo,
                    "parametros": parametros,
                    "ejecutar": False,
                    "bloqueado": False,
                    "confianza": 0.5,
                    "texto_original": texto,
                    "mensaje": str(e),
                }
        else:
            ejecutado = True

        return {
            "tipo": tipo,
            "parametros": parametros,
            "ejecutar": ejecutado,
            "bloqueado": False,
            "confianza": 1.0,
            "texto_original": texto,
            "mensaje": "orden procesada correctamente",
        }


def _perfil_por_defecto():
    """Permite evaluar el agente sin abrir el simulador."""
    import sys
    from pathlib import Path
    entorno = Path(__file__).resolve().parent.parent / "entorno"
    if str(entorno) not in sys.path:
        sys.path.insert(0, str(entorno))
    from sim.safety import perfil
    return perfil("tp07")


# =====================================================================
#  PROGRAMA PRINCIPAL - no hace falta que lo toques
# =====================================================================
def main():
    import sys

    # Modo sin robot: solo evalua los 25 casos. Sirve para trabajar el
    # clasificador sin tener el simulador abierto.
    sin_robot = "--sin-robot" in sys.argv

    robot = None
    if not sin_robot:
        robot = Robot()
        robot.conectar()

    try:
        agente = AgenteRobot(robot)
        evaluar(agente)

        if robot is not None:
            print("\n  Escribi ordenes para el robot. Enter vacio para salir.")
            while True:
                try:
                    texto = input("\n  > ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if not texto:
                    break
                r = agente.procesar(texto)
                print(f"    {r['tipo']}  {r.get('mensaje', '')}")
    finally:
        if robot is not None:
            robot.detenerse()
            robot.desconectar()


if __name__ == "__main__":
    main()
