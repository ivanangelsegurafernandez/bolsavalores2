# -*- coding: utf-8 -*-

# === BLOQUE 0 — OBJETIVOS DEL PROGRAMA 5R6M-1-2-4-8-16 ===
#
# Este script coordina:
# - Lectura de CSV enriquecidos de los bots fulll45–fulll50
# - Control de Martingala 1-2-4-8-16
# - Gestión de tokens DEMO/REAL
# - IA (XGBoost) para probabilidades de éxito
# - HUD visual con Prob IA, % éxito, saldo, meta y eventos
#
# ÍNDICE DE BLOQUES:
#   BLOQUE 1 — IMPORTS Y ENTORNO BÁSICO
#   BLOQUE 2 — CONFIGURACIÓN GLOBAL (MARTINGALA, HUD, AUDIO, IA)
#   BLOQUE 3 — CONFIGURACIÓN DE REENTRENAMIENTO Y MODOS IA
#   BLOQUE 4 — AUDIO (INIT Y REPRODUCCIÓN)
#   BLOQUE 5 — TOKENS, BOT_NAMES Y ESTADO GLOBAL
#   BLOQUE 6 — LOCKS, FIRMAS Y UTILIDADES CSV
#   BLOQUE 7 — ORDEN DE REAL Y CONTROL DE TOKEN
#   BLOQUE 8 — NORMALIZACIÓN Y PUNTAJE DE ESTRATEGIA
#   BLOQUE 9 — DETECCIÓN DE MARTINGALA Y REINICIOS
#   BLOQUE 10 — IA: DATASET, MODELO Y PREDICCIÓN
#   BLOQUE 11 — HUD Y PANEL VISUAL
#   BLOQUE 12 — CONTROL MANUAL REAL Y CONDICIONES SEGURAS
#   BLOQUE 13 — LOOP PRINCIPAL, WEBSOCKET Y TECLADO
#   BLOQUE 99 — RESUMEN FINAL DE LO QUE SE LOGRA
#
# Nota:
#   Esta organización NO cambia la lógica del programa.
#   Solo añade estructura para facilitar futuras modificaciones.
#
# === FIN BLOQUE 0 ===

# === BLOQUE 1 — IMPORTS Y ENTORNO BÁSICO ===
import os, csv, time, random, asyncio, websockets, json, re
from collections import deque
from colorama import Fore, Style, init
import pygame
try:
    import winsound
except ImportError:
    winsound = None
from unicodedata import normalize
import threading
from datetime import datetime, timedelta
from contextlib import contextmanager
import sys
import shutil
import joblib
import numpy as np
import pandas as pd

import math
import hashlib
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

import warnings
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but StandardScaler was fitted with feature names"
)

# ============================================================
# XGBoost (robusto): permite correr aunque xgboost no esté
# ============================================================
try:
    import xgboost as xgb  # opcional (por compatibilidad)
    from xgboost import XGBClassifier
    _XGBOOST_OK = True
except Exception:
    xgb = None
    XGBClassifier = None
    _XGBOOST_OK = False

# --- Teclado Windows (seguro y único) ---
try:
    import msvcrt as _msvcrt
    class _MSWrap:
        def __bool__(self): return True
        def kbhit(self):
            try: return _msvcrt.kbhit()
            except Exception: return False
        def getch(self):
            try: return _msvcrt.getch()
            except Exception: return b''
    msvcrt = _MSWrap()
    HAVE_MSVCRT = True
except Exception:
    class _DummyMS:
        def __bool__(self): return False
        def kbhit(self): return False
        def getch(self): return b''
    msvcrt = _DummyMS()
    HAVE_MSVCRT = False

# Forzar la ruta fija al directorio del script
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"📁 Directorio de trabajo fijado a: {script_dir}")
except Exception as e:
    print(f"⚠️ No se pudo cambiar al directorio del script: {e}. Usando cwd actual.")

init(autoreset=True)
# === FIN BLOQUE 1 ===

# === BLOQUE 2 — CONFIGURACIÓN GLOBAL (MARTINGALA, HUD, AUDIO, IA) ===
# === CONFIGURACIÓN DE MARTINGALA ===
MARTI_ESCALADO = [1, 2, 4, 8, 16]  # Escalado ajustado a 5 pasos
MONTO_TOL = 0.01  # Tolerancia para redondeos
SONAR_TAMBIEN_EN_DEMO = False  # Activar sonidos para victorias en DEMO
SONAR_SOLO_EN_GATEWIN = True   # Solo sonar dentro de la ventana GateWIN
SONAR_FUERA_DE_GATEWIN = False # Permitir sonidos fuera de GateWIN si se se habilita
AUDIO_TIMEOUT_S = 0  # 0 significa sin timeout

# === REMATE (modo cierre solo con WIN) ===
MODO_REMATE = True           # Continuar hasta WIN o fin de Martingala
REMATE_SIN_TOPE = False      # Limitado por MAX_CICLOS

# === HUD / Layout ===
HUD_LAYOUT = "bottom_center"  # Fijado en centro inferior
HUD_VISIBLE = True       # Para ocultarlo con tecla

# --- Oráculo visual ---
ORACULO_THR_MIN   = 0.75
ORACULO_N_MIN     = 40
ORACULO_DELTA_PRE = 0.05

# Umbral único (verde + aviso IA)  -> esto NO lo tocamos
IA_VERDE_THR = 0.75
AUTO_REAL_THR = 0.75  # umbral fijo para auto-promoción a REAL

# Umbral "operativo/UI" (señales actuales, semáforo, etc.)
# OJO: también se usa como piso en get_umbral_operativo(), así que NO lo bajamos para no cambiar conducta del bot.
IA_METRIC_THRESHOLD = IA_VERDE_THR

# ✅ Umbral SOLO para auditoría/calibración (señales CERRADAS en ia_signals_log)
# Esto es lo que querías: contar cierres desde 60% sin afectar la operativa.
IA_CALIB_THRESHOLD = 0.60
IA_CALIB_GOAL_THRESHOLD = 0.70  # objetivo: medir cierres fuertes (≥70%)
IA_CALIB_MIN_CLOSED = 200  # mínimo recomendado para considerar estable la auditoría

# Umbral del aviso de audio (archivo ia_scifi_02_ia53_dry.wav)
AUDIO_IA53_THR = 0.75

# Anti-spam + rearme
AUDIO_IA53_COOLDOWN_S = 20     # no repetir más de 1 vez cada X segundos por bot
AUDIO_IA53_RESET_HYST = 0.03   # se rearma cuando cae por debajo de (thr - hyst)

# === Caché de sonidos ===
SOUND_CACHE = {}
SOUND_LOAD_ERRORS = set()
SOUND_PATHS = {
    "ganancia_real": "ganabot.wav",
    "ganancia_demo": "ganabot.wav",
    "perdida_real": "perdida.wav",
    "perdida_demo": "perdida.wav",
    "meta_15": "meta15.wav",
    "racha_detectada": "detectaracha.wav",
    "test": "test.wav",
    "ia_53": "ia_scifi_08_53porciento_dry.wav",

}
AUDIO_AVAILABLE = False
META_ACEPTADA = False
MODAL_ACTIVO = False
sonido_disparado = False
# === FIN BLOQUE 2 ===

# === BLOQUE 3 — CONFIGURACIÓN DE REENTRENAMIENTO Y MODOS IA ===
# === CONFIGURACIÓN DE REENTRENAMIENTO ===
RETRAIN_INTERVAL_ROWS = 100     # por volumen
RETRAIN_INTERVAL_MIN  = 15      # por tiempo
MIN_NEW_ROWS_FOR_TIME = 20      # al menos 20 filas nuevas para reentrenar por tiempo
MAX_DATASET_ROWS = 10000
last_retrain_count = 0
last_retrain_ts    = time.time()  # Inicializado al boot para arranque en frío
_entrenando_lock = threading.Lock()  # Lock para antireentradas en maybe_retrain

# === MODO ENTRENAMIENTO CON POCA DATA (no toca la lógica de IA) ===
LOW_DATA_MODE = True           # True = permite entrenar con muy pocas filas
MIN_FIT_ROWS_PROD = 100        # umbral “confiable” para producción (lo que ya usabas)
MIN_FIT_ROWS_LOW  = 4          # umbral mínimo para permitir fit “experimental”
RELIABLE_POS_MIN  = 20         # mínimos para considerar fiable (calibración/umbral estable)
RELIABLE_NEG_MIN  = 20

# Modo manual desactivado: priorizamos automatización completa por Prob IA.
# Si luego quieres volver al modo manual, ponlo en True.
MODO_REAL_MANUAL = False

# Martingala global
marti_paso = 0
marti_activa = False

# Contador global de ciclos de martingala (HUD + orquestación automática)
# 0 = sin pérdidas consecutivas en REAL; 1..MAX_CICLOS = racha de pérdidas vigente.
marti_ciclos_perdidos = 0

# Nueva: Umbrales mínimos para historial IA
MIN_IA_SENIALES_CONF = 10  # Mínimo señales cerradas para confiar en prob_hist
MIN_AUC_CONF = 0.65        # AUC mínimo para audios/colores verdes
MAX_CLASS_IMBALANCE = 0.8  # Máx proporción pos/neg para entrenar (evita 99% wins)
AUC_DROP_TOL = 0.05        # Tolerancia para no machacar modelo si AUC baja

# Semáforo de calibración (lectura rápida PredMedia/Real/Inflación/n)
SEM_CAL_N_ROJO = 30
SEM_CAL_N_AMARILLO = 100
SEM_CAL_INFL_OK_PP = 5.0
SEM_CAL_INFL_WARN_PP = 15.0

# ============================================================
# Defaults IA (centralizados, sin duplicados)
# ============================================================
MIN_TRAIN_ROWS  = 250
TEST_SIZE_FRAC  = 0.20
MIN_TEST_ROWS   = 40
THR_DEFAULT = 0.50

# Split honesto: TRAIN_BASE (pasado) / CALIB (más reciente) / TEST (último)
CALIB_SIZE_FRAC = 0.15
MIN_CALIB_ROWS = 80

# Feature list canónica (si tu reentreno define otra, ahí la cambias UNA vez)
# ============================================================
# Feature set CORE (13) — estable y sin mutaciones
# ============================================================
FEATURE_NAMES_CORE_13 = [
    "rsi_9","rsi_14","sma_5","sma_20","cruce_sma","breakout",
    "rsi_reversion","racha_actual","payout","puntaje_estrategia",
    "volatilidad","es_rebote","hora_bucket",
]

# Por defecto entrenamos SOLO con las 13 core (modo estable)
FEATURE_NAMES_DEFAULT = list(FEATURE_NAMES_CORE_13)

class ModeloXGBCalibrado:
    """
    Wrapper picklable para calibrar probabilidades con un holdout temporal (CALIB),
    sin re-entrenar el modelo base. El modelo espera X ya escalado.
    calib_kind: "sigmoid" (Platt con LogisticRegression sobre logit(p)) o "isotonic".
    """
    def __init__(self, modelo_base, calib_kind: str, calib_obj):
        self.modelo_base = modelo_base
        self.calib_kind = str(calib_kind)
        self.calib_obj = calib_obj

    def _calibrar_p(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        p = np.clip(p, 1e-6, 1.0 - 1e-6)

        if self.calib_kind == "sigmoid":
            z = np.log(p / (1.0 - p)).reshape(-1, 1)
            p_cal = self.calib_obj.predict_proba(z)[:, 1]
            return np.clip(p_cal, 1e-6, 1.0 - 1e-6)

        # isotonic
        p_cal = self.calib_obj.transform(p)
        return np.clip(np.asarray(p_cal, dtype=float), 1e-6, 1.0 - 1e-6)

    def predict_proba(self, X):
        p_base = self.modelo_base.predict_proba(X)[:, 1]
        p_cal = self._calibrar_p(p_base)
        return np.vstack([1.0 - p_cal, p_cal]).T

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)

# === FIN BLOQUE 3 ===

# === BLOQUE 4 — AUDIO (INIT Y REPRODUCCIÓN) ===
# Inicialización de audio
def init_audio():
    global AUDIO_AVAILABLE, SOUND_CACHE

    # No asumimos nada: recalculamos disponibilidad cada vez
    AUDIO_AVAILABLE = False

    # 1) Asegurar mixer (si no está listo)
    if pygame.mixer.get_init():
        AUDIO_AVAILABLE = True
    else:
        drivers = ['directsound', 'winmm', 'wasapi', None]
        configs = [
            (44100, -16, 2, 1024),
            (22050, -16, 2, 512),
            (44100, -16, 1, 1024),
        ]
        for driver in drivers:
            for freq, size, channels, buffer in configs:
                try:
                    if driver:
                        os.environ["SDL_AUDIODRIVER"] = driver
                    pygame.mixer.pre_init(frequency=freq, size=size, channels=channels, buffer=buffer)
                    pygame.mixer.init()
                    AUDIO_AVAILABLE = True
                    break
                except Exception:
                    pass
            if AUDIO_AVAILABLE:
                break

    # 2) Fallback winsound (aunque no tengamos pygame)
    if not AUDIO_AVAILABLE and winsound:
        AUDIO_AVAILABLE = True

    # 3) Cargar sonidos SOLO si mixer está operativo
    if pygame.mixer.get_init():
        base_dir = os.path.dirname(__file__)
        for event, filename in SOUND_PATHS.items():
            if event in SOUND_LOAD_ERRORS:
                continue
            path = os.path.join(base_dir, filename)
            if os.path.exists(path):
                try:
                    SOUND_CACHE[event] = pygame.mixer.Sound(path)
                except Exception:
                    SOUND_LOAD_ERRORS.add(event)

def reproducir_evento(evento, es_demo=False, dentro_gatewin=True):
    global sonido_disparado

    if not AUDIO_AVAILABLE:
        return

    # Reglas de GateWIN/DEMO (mismas que tenías)
    if evento != "ia_53":
        if SONAR_SOLO_EN_GATEWIN and (not dentro_gatewin) and (not SONAR_FUERA_DE_GATEWIN):
            return
        if es_demo and not SONAR_TAMBIEN_EN_DEMO:
            return

    # 1) Preferir pygame si está cargado
    try:
        if evento in SOUND_CACHE:
            SOUND_CACHE[evento].play()
            sonido_disparado = True
            return
    except Exception:
        pass

    # 2) Fallback winsound (si pygame no está usable o no cargó el sonido)
    if winsound:
        try:
            filename = SOUND_PATHS.get(evento)
            if not filename:
                return
            base_dir = os.path.dirname(__file__)
            path = os.path.join(base_dir, filename)
            if os.path.exists(path):
                winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
                sonido_disparado = True
        except Exception:
            pass
# === FIN BLOQUE 4 ===

# === BLOQUE 5 — TOKENS, BOT_NAMES Y ESTADO GLOBAL ===
# Leer tokens del usuario
def leer_tokens_usuario():
    if not os.path.exists("tokens_usuario.txt"):
        return None, None
    try:
        with open("tokens_usuario.txt", "r", encoding="utf-8") as file:
            lines = [line.strip() for line in file.readlines()]
            if len(lines) < 2:
                return None, None
            token_demo, token_real = lines[0], lines[1]
            if not token_demo or not token_real:
                return None, None
            return token_demo, token_real
    except Exception:
        return None, None

# Escritura atómica de token
def write_token_atomic(path, content):
    temp_path = path + ".tmp"
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, path)
        return True
    except Exception:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass
        return False


BOT_NAMES = ["fulll45","fulll46","fulll47","fulll48","fulll49","fulll50"]
IA53_TRIGGERED = {bot: False for bot in BOT_NAMES}
IA53_LAST_TS = {bot: 0.0 for bot in BOT_NAMES}
TOKEN_FILE = "token_actual.txt"
DERIV_WS_URL = "wss://ws.derivws.com/websockets/v3?app_id=1089"
saldo_real = "--"
SALDO_INICIAL = None
META = None
meta_mostrada = False
eventos_recentes = deque(maxlen=8)
reinicio_forzado = asyncio.Event()

salir = False
pausado = False
reinicio_manual = False

LIMPIEZA_PANEL_HASTA = 0
ULTIMA_ACT_SALDO = 0
REFRESCO_SALDO = 12
MAX_CICLOS = len(MARTI_ESCALADO)
huellas_usadas = {bot: set() for bot in BOT_NAMES}
SNAPSHOT_FILAS = {bot: 0 for bot in BOT_NAMES}
REAL_ENTRY_BASELINE = {bot: 0 for bot in BOT_NAMES}  # filas al entrar/reafirmar REAL
OCULTAR_HASTA_NUEVO = {bot: False for bot in BOT_NAMES}
t_inicio_indef = {bot: None for bot in BOT_NAMES}
last_update_time = {bot: time.time() for bot in BOT_NAMES}
LAST_REAL_CLOSE_SIG = {bot: None for bot in BOT_NAMES}  # evita procesar el mismo cierre REAL varias veces
REAL_OWNER_LOCK = None  # owner REAL en memoria (evita carreras de lectura de archivo)

try:
    last_sig_por_bot
except NameError:
    last_sig_por_bot = {b: None for b in BOT_NAMES}

estado_bots = {
    bot: {
        "resultados": [], 
        "token": "DEMO", 
        "trigger_real": False,
        "ganancias": 0, 
        "perdidas": 0, 
        "porcentaje_exito": None,
        "tamano_muestra": 0,
        "prob_ia": None,              # guardará prob REAL (0..1). OJO: ya NO la forzamos a 0 por “no señal”
        "ia_ready": False,           # True solo si logramos armar features + predecir sin error
        "ia_last_err": None,         # texto corto del motivo si no se pudo predecir
        "ia_last_prob_ts": 0.0,      # timestamp de la última prob calculada
        "ciclo_actual": 1,
        "modo_real_anunciado": False, 
        "ultimo_resultado": None,
        "reintentar_ciclo": False,
        "remate_active": False,
        "remate_start": None,
        "remate_reason": "",
        "fuente": None,  
        "real_activado_en": 0.0,  
        "ignore_cierres_hasta": 0.0,
        "modo_ia": "off",  # Nueva para modo (off, low_data, modelo)
        "ia_seniales": 0,  # contadores para medir IA
        "ia_aciertos": 0,
        "ia_fallos": 0,
        "ia_senal_pendiente": False,  # Flag para operación recomendada por IA
        "ia_prob_senal": None         # NUEVO: prob IA en el momento de la señal
    }
    for bot in BOT_NAMES
}
IA90_stats = {bot: {"n": 0, "ok": 0, "pct": 0.0} for bot in BOT_NAMES}
# --- BLINDAJE: asegurar símbolos críticos si faltan (no pisa definiciones reales) ---
if "RENDER_LOCK" not in globals():
    RENDER_LOCK = threading.Lock()

if "agregar_evento" not in globals():
    def agregar_evento(msg: str):
        try:
            ts = time.strftime("%H:%M:%S")
            eventos_recentes.appendleft(f"{ts} {msg}")
        except Exception:
            try:
                print(msg)
            except Exception:
                pass
# --- /BLINDAJE ---

# === FIN BLOQUE 5 ===

# === BLOQUE 6 — LOCKS, FIRMAS Y UTILIDADES CSV ===
def _firma_registro(feature_names, row_vals, label):
    """
    Firma estable anti-duplicados:
    - Formato fijo para floats (evita variaciones 0.1 vs 0.10000000002)
    """
    parts = []
    for v in row_vals:
        try:
            parts.append(f"{float(v):.6f}")
        except Exception:
            parts.append(str(v))
    try:
        parts.append(str(int(label)))
    except Exception:
        parts.append(str(label))
    return "|".join(parts)

# Contar filas en CSV (sin header)
def contar_filas_csv(bot_name: str) -> int:
    ruta = f"registro_enriquecido_{bot_name}.csv"
    if not os.path.exists(ruta):
        return 0
    for encoding in ["utf-8", "latin-1", "windows-1252"]:
        try:
            with open(ruta, "r", newline="", encoding=encoding, errors="replace") as f:
                n = sum(1 for _ in f) - 1
                return max(0, n)
        except Exception as e:
            print(f"⚠️ Error contando filas en {ruta}: {e}")
            continue
    return 0

# Contar filas en dataset_incremental.csv (sin contar header)
def contar_filas_incremental() -> int:
    """
    Devuelve número de filas (sin header) de dataset_incremental.csv.

    Optimizado:
    - Cachea (pos, size, rows) para evitar re-escaneo completo en cada llamada.
    - Si el archivo crece, cuenta solo las líneas nuevas.
    """
    try:
        path = "dataset_incremental.csv"

        if not os.path.exists(path):
            contar_filas_incremental._cache = {"pos": 0, "rows": 0, "size": 0}
            return 0

        cache = getattr(contar_filas_incremental, "_cache", None)
        size = os.path.getsize(path)

        # Recuento completo en binario (robusto a encoding)
        def _count_full_rows() -> int:
            total = 0
            last_byte = b""
            with open(path, "rb") as f:
                while True:
                    chunk = f.read(1024 * 1024)
                    if not chunk:
                        break
                    total += chunk.count(b"\n")
                    last_byte = chunk[-1:]
            # Si el archivo NO termina en \n, hay una línea final sin salto
            if size > 0 and last_byte != b"\n":
                total += 1
            # Quita header si existe
            return max(0, total - 1)

        # Sin cache o el archivo se redujo/truncó: recuenta todo
        if (not cache) or (size < int(cache.get("size", 0) or 0)) or (int(cache.get("pos", 0) or 0) > size):
            rows = _count_full_rows()
            contar_filas_incremental._cache = {"pos": size, "rows": rows, "size": size}
            return rows

        # Si no cambió, devuelve cache
        if size == int(cache.get("size", 0) or 0):
            return int(cache.get("rows", 0) or 0)

        # Creció: cuenta solo líneas nuevas desde la última posición
        pos = int(cache.get("pos", 0) or 0)
        new_lines = 0
        with open(path, "rb") as f:
            f.seek(pos)
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                new_lines += chunk.count(b"\n")

        rows = int(cache.get("rows", 0) or 0) + new_lines
        contar_filas_incremental._cache = {"pos": size, "rows": rows, "size": size}
        return rows

    except Exception:
        return 0

# Lock de archivo
@contextmanager
def file_lock(path="real.lock", timeout=5.0, stale_after=30.0):
    """
    Lock por archivo (cross-platform) con protección anti-colisión:

    - NO borra el lock de otro proceso activo.
    - Solo intenta limpiar locks *stale* (viejos) si supera stale_after segundos.
    - Si no logra adquirir lock, continúa SIN exclusión (como ya venías haciendo),
      pero sin destruir el lock ajeno.
    """
    start_time = time.time()
    fd = None
    acquired = False

    try:
        # 1) Intento normal por timeout
        while (time.time() - start_time) < float(timeout):
            try:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                acquired = True
                break
            except FileExistsError:
                time.sleep(0.10)
            except Exception:
                time.sleep(0.10)

        # 2) Si no se pudo, evaluar si el lock parece "stale"
        if not acquired:
            age = None
            try:
                age = time.time() - os.path.getmtime(path)
            except Exception:
                age = None

            if age is not None and age > float(stale_after):
                # Solo si es viejo de verdad, intentamos limpiar
                try:
                    os.remove(path)
                    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                    acquired = True
                except Exception as e:
                    try:
                        print(f"⚠️ Lock stale no se pudo limpiar ({path}): {e}. Continúo sin exclusión.")
                    except Exception:
                        pass
            else:
                # Lock reciente: NO tocarlo
                try:
                    print(f"⚠️ No se adquirió lock ({path}) en {timeout}s (lock reciente). Continúo sin exclusión.")
                except Exception:
                    pass

        # 3) Ejecutar la sección crítica (con o sin lock adquirido)
        yield

    finally:
        if acquired and fd is not None:
            try:
                os.close(fd)
            except Exception:
                pass
            try:
                os.remove(path)
            except Exception:
                pass
# ============================================================
# DATASET INCREMENTAL — Reparación de esquema "mutante"
# (header viejo / columnas extra / filas con campos de más)
# Objetivo: mantener SIEMPRE un CSV estable para pandas/IA.
# ============================================================

# Reusar el set core (13) para que incremental y entrenamiento nunca diverjan
try:
    INCREMENTAL_FEATURES_V2 = list(FEATURE_NAMES_CORE_13)
except Exception:
    INCREMENTAL_FEATURES_V2 = [
        "rsi_9","rsi_14","sma_5","sma_20","cruce_sma","breakout",
        "rsi_reversion","racha_actual","payout","puntaje_estrategia",
        "volatilidad","es_rebote","hora_bucket",
    ]
# === LOCK ESTRICTO (solo para escrituras sensibles como incremental.csv) ===
@contextmanager
def file_lock_required(path: str, timeout: float = 6.0, stale_after: float = 30.0):
    """
    Igual que file_lock, pero:
    - Si NO adquiere lock, NO ejecuta la sección crítica (yield False).
    - Para escrituras que NO toleran concurrencia (append CSV).
    """
    start_time = time.time()
    fd = None
    acquired = False

    try:
        while (time.time() - start_time) < float(timeout):
            try:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                acquired = True
                break
            except FileExistsError:
                time.sleep(0.10)
            except Exception:
                time.sleep(0.10)

        if not acquired:
            age = None
            try:
                age = time.time() - os.path.getmtime(path)
            except Exception:
                age = None

            if age is not None and age > float(stale_after):
                try:
                    os.remove(path)
                    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                    acquired = True
                except Exception:
                    acquired = False

        yield acquired

    finally:
        if acquired and fd is not None:
            try:
                os.close(fd)
            except Exception:
                pass
            try:
                os.remove(path)
            except Exception:
                pass
# === /LOCK ESTRICTO ===

def _canonical_incremental_cols(feature_names: list | None = None) -> list:
    fn = feature_names if feature_names else INCREMENTAL_FEATURES_V2
    return list(fn) + ["result_bin"]

def _safe_float(x):
    try:
        if x is None:
            return None
        if isinstance(x, str):
            x = x.strip()
            if x == "":
                return None
        v = float(x)
        if not np.isfinite(v):
            return None
        return v
    except Exception:
        return None

def _safe_int01(x):
    try:
        if x is None:
            return None
        if isinstance(x, str):
            x = x.strip()
            if x == "":
                return None
        v = int(float(x))
        if v not in (0, 1):
            return None
        return v
    except Exception:
        return None

def reparar_dataset_incremental_mutante(ruta: str = "dataset_incremental.csv", cols: list | None = None) -> bool:
    """
    Repara dataset_incremental.csv cuando quedó 'mutante' por:
    - header corrupto (ej: racha...ia) o incompleto
    - filas con más/menos columnas (ej: bot_id, activo_id metidos)
    - mezcla de esquemas (Expected X fields, saw Y)

    Estrategia:
    - Reescribe un CSV limpio con columnas canónicas (cols).
    - Si el archivo actual tiene columnas canónicas presentes, mapea por header.
    - Si el header no es usable, intenta rescate por POSICIÓN:
        * len>=16: [0..12] + [15] (drop bot_id/activo_id)
        * len==15: [0..12] + [14]
        * len==14: [0..12] + [13]  (13 feats + label)
        * len>len(cols): toma primeras (len(cols)-1) + última como label
    - Crea backup del archivo original con sufijo .bak_<epoch>.
    """
    cols = cols or _canonical_incremental_cols()
    if not os.path.exists(ruta):
        return False

    # Leer header y detectar mutación
    header_list = None
    enc_usado = None
    for enc in ("utf-8", "latin-1", "windows-1252"):
        try:
            with open(ruta, "r", newline="", encoding=enc, errors="replace") as f:
                first = f.readline()
            header_list = [h.strip() for h in first.strip().split(",")] if first else []
            enc_usado = enc
            break
        except Exception:
            continue

    if header_list is None:
        return False

    # Si el header ya es canónico, igual escanear rápido por longitudes
    header_ok = (header_list == cols)
    header_has_canonical = set(cols).issubset(set(header_list))

    needs_repair = not header_ok

    # Escaneo rápido de longitudes (si hay mezcla de campos, se marca mutante)
    try:
        with open(ruta, "r", newline="", encoding=enc_usado or "utf-8", errors="replace") as f:
            reader = csv.reader(f)
            _ = next(reader, None)  # header
            for j, row in enumerate(reader, start=1):
                if not row:
                    continue
                if len(row) != len(header_list):
                    needs_repair = True
                    break
                if j >= 3000:
                    break
    except Exception:
        needs_repair = True

    if not needs_repair:
        return False

    # Armar filas limpias
    cleaned_rows = []
    header_index = {name: i for i, name in enumerate(header_list)} if header_has_canonical else {}

    for enc in ("utf-8", "latin-1", "windows-1252"):
        try:
            with open(ruta, "r", newline="", encoding=enc, errors="replace") as f:
                reader = csv.reader(f)
                _ = next(reader, None)  # header
                for row in reader:
                    if not row:
                        continue

                    new_row = None

                    # 1) Si el header contiene columnas canónicas, mapear por nombre
                    if header_has_canonical:
                        try:
                            new_row = [row[header_index[c]] if header_index[c] < len(row) else "" for c in cols]
                        except Exception:
                            new_row = None

                    # 2) Rescate por posición si el header es inútil
                    if new_row is None:
                        ncols = len(cols)
                        rlen = len(row)

                        # Casos exactos: (13 feats + extras + label)
                        # Usamos SIEMPRE row[-1] como label (más seguro) y tomamos las 13 primeras como feats.
                        if ncols == 14 and rlen == 16:
                            # 13 feats + bot_id + activo_id + label
                            new_row = [row[i] for i in range(13)] + [row[-1]]
                        elif ncols == 14 and rlen == 15:
                            # 13 feats + (1 extra) + label
                            new_row = [row[i] for i in range(13)] + [row[-1]]
                        elif ncols == 14 and rlen == 14:
                            # 13 feats + label
                            new_row = [row[i] for i in range(13)] + [row[-1]]
                        elif rlen >= ncols:
                            # Caso general: toma primeras features y el último como label
                            new_row = list(row[:ncols - 1]) + [row[-1]]
                        else:
                            continue


                    # Validación y casteo (features float, label int 0/1)
                    feats = []
                    ok = True
                    for x in new_row[:-1]:
                        v = _safe_float(x)
                        if v is None:
                            ok = False
                            break
                        feats.append(v)

                    lab = _safe_int01(new_row[-1])
                    if not ok or lab is None:
                        continue

                    cleaned_rows.append(feats + [lab])
            break
        except Exception:
            continue

    # Reescritura atómica con backup
    tmp = ruta + ".tmp_repair"
    try:
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for r in cleaned_rows:
                w.writerow(r)
            f.flush()
            os.fsync(f.fileno())

        backup = f"{ruta}.bak_{int(time.time())}"
        backed_up = False

        # 1) Intento preferido: renombrar (rápido y atómico)
        try:
            os.replace(ruta, backup)
            backed_up = True
        except Exception:
            backed_up = False

        # 2) Fallback: copiar (cuando rename falla por permisos/locks)
        if not backed_up:
            try:
                shutil.copy2(ruta, backup)
                backed_up = True
            except Exception:
                backed_up = False

        # 3) Si NO hay backup, NO pisamos el original
        if not backed_up:
            raise RuntimeError("No se pudo crear backup del incremental; se aborta reparación para no perder datos.")

        os.replace(tmp, ruta)
        return True

    except Exception as e:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        fn_evt = globals().get("agregar_evento", None)
        try:
            if callable(fn_evt):
                fn_evt(f"⚠️ Incremental: reparación falló: {e}")
            else:
                print(f"⚠️ Incremental: reparación falló: {e}")
        except Exception:
            print(f"⚠️ Incremental: reparación falló: {e}")
        return False

# Firma persistente anti-duplicados
_SIG_DIR = ".sigcache"
os.makedirs(_SIG_DIR, exist_ok=True)

def _sig_path(bot): 
    safe = str(bot).replace("/", "_").replace("\\", "_")
    return os.path.join(_SIG_DIR, f"{safe}.sig")

def _load_recent_sigs(bot: str, max_keep: int = 50) -> list:
    """
    Devuelve lista de firmas recientes (últimas N) desde disco.
    Compatible con formato viejo (1 sola firma).
    """
    try:
        p = _sig_path(bot)
        if not os.path.exists(p):
            return []
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            lines = [ln.strip() for ln in f.read().splitlines() if ln.strip()]
        if not lines:
            return []
        return lines[-int(max_keep):]
    except Exception:
        return []

def _sig_in_cache(bot: str, sig: str, max_keep: int = 50) -> bool:
    try:
        return sig in set(_load_recent_sigs(bot, max_keep=max_keep))
    except Exception:
        return False

def _append_sig_cache(bot: str, sig: str, max_keep: int = 50):
    """
    Guarda firma al final, manteniendo solo últimas N (y sin duplicados internos).
    """
    try:
        max_keep = int(max_keep)
        if max_keep < 5:
            max_keep = 5

        lst = _load_recent_sigs(bot, max_keep=max_keep)
        # mover al final si existe
        lst = [x for x in lst if x != sig] + [sig]
        lst = lst[-max_keep:]

        with open(_sig_path(bot), "w", encoding="utf-8") as f:
            f.write("\n".join(lst))
    except Exception:
        pass
# === COMPAT: helpers legacy (evita NameError y mantiene tu lógica actual) ===
def _load_last_sig(bot: str) -> str | None:
    """
    Compatibilidad: versiones antiguas esperaban una sola firma.
    Hoy guardamos varias en _sigcache; devolvemos la última.
    """
    try:
        lst = _load_recent_sigs(bot, max_keep=50)
        if not lst:
            return None
        return lst[-1]
    except Exception:
        return None

def _save_last_sig(bot: str, sig: str):
    """
    Compatibilidad: guarda como “última firma”, manteniendo historial.
    """
    try:
        _append_sig_cache(bot, sig, max_keep=50)
    except Exception:
        pass
# === /COMPAT ===

def _make_sig(row_dict):
    """Firma estable para comparar filas entre reinicios (sin timestamp si no existe)."""
    try:
        # Orden determinista
        data = {k: row_dict.get(k) for k in sorted(row_dict.keys())}
        s = json.dumps(data, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()
    except:
        return None
# Nueva: Validar fila para incremental (blindaje contra basura)
def validar_fila_incremental(fila_dict, feature_names):
    # Asegura numericidad real
    for k in feature_names:
        v = fila_dict.get(k, None)
        try:
            v = float(v)
            if not np.isfinite(v):
                return False, f"{k}=NaN/inf"
        except Exception:
            return False, f"{k}=no numérico"
        fila_dict[k] = v  # normaliza en sitio

    # Rangos lógicos básicos
    if not (0 <= fila_dict.get("rsi_9", 50) <= 100):
        return False, "RSI_9 fuera de 0-100"
    if not (0 <= fila_dict.get("rsi_14", 50) <= 100):
        return False, "RSI_14 fuera de 0-100"
    if not (0 <= fila_dict.get("payout", 0.0) <= 1.5):
        return False, "Payout fuera de 0-1.5"
    if "volatilidad" in fila_dict and not (0 <= fila_dict["volatilidad"] <= 1):
        return False, "Volatilidad fuera de 0-1"
    if "es_rebote" in fila_dict and not (0 <= fila_dict["es_rebote"] <= 1):
        return False, "es_rebote fuera de 0-1"
    if "hora_bucket" in fila_dict and not (0 <= fila_dict["hora_bucket"] <= 1):
        return False, "hora_bucket fuera de 0-1"

    return True, ""
        
def _anexar_incremental_desde_bot_CANON(bot: str, fila_dict_or_full: dict, label: int | None = None, feature_names: list | None = None) -> bool:
    """
    Anexa 1 fila al dataset_incremental.csv de forma estable:
    - Header canónico (anti "mutante")
    - Lock dedicado (incremental.lock) para evitar choques
    - Repair del CSV SOLO bajo lock (evita corrupción por concurrencia)
    - Retry ante PermissionError (Excel/OneDrive/AV)
    - Anti-duplicado por firma persistente (_sigcache por bot)
    """
    try:
        ruta = "dataset_incremental.csv"
        feats = feature_names or INCREMENTAL_FEATURES_V2
        cols = _canonical_incremental_cols(feats)

        if not isinstance(fila_dict_or_full, dict) or not fila_dict_or_full:
            return False

        # Label: aceptar parámetro o leer del dict
        if label is None:
            lb = fila_dict_or_full.get("result_bin", None)
            try:
                label = int(float(lb))
            except Exception:
                return False

        try:
            label = int(label)
        except Exception:
            return False
        if label not in (0, 1):
            return False

        # Dict solo con features canónicas
        fila_dict = {k: fila_dict_or_full.get(k, None) for k in feats}

        # Validación fuerte
        ok, why = validar_fila_incremental(fila_dict, feats)
        if not ok:
            fn_evt = globals().get("agregar_evento", None)
            try:
                if callable(fn_evt):
                    fn_evt(f"⚠️ Incremental: fila descartada {bot}: {why}")
            except Exception:
                pass
            return False

        row_vals = [float(fila_dict[k]) for k in feats]
        sig = _firma_registro(feats, row_vals, label)

        # Anti-duplicado persistente (últimas N)
        if _sig_in_cache(bot, sig, max_keep=50):
            return False

        attempts = 8
        base_sleep = 0.08

        with file_lock_required("incremental.lock", timeout=6.0, stale_after=30.0) as got:
            if not got:
                fn_evt = globals().get("agregar_evento", None)
                try:
                    if callable(fn_evt):
                        fn_evt("⚠️ Incremental: no se pudo adquirir lock (incremental.lock). Fila omitida para evitar corrupción.")
                except Exception:
                    pass
                return False

            # ✅ Bajo lock: asegurar existencia + header estable + repair si hace falta
            if os.path.exists(ruta):
                try:
                    with open(ruta, "r", encoding="utf-8", errors="replace", newline="") as f:
                        first = f.readline().strip()
                    header_now = [h.strip() for h in first.split(",")] if first else []
                    if header_now != cols:
                        reparar_dataset_incremental_mutante(ruta=ruta, cols=cols)
                except Exception:
                    try:
                        reparar_dataset_incremental_mutante(ruta=ruta, cols=cols)
                    except Exception:
                        pass
            else:
                try:
                    with open(ruta, "w", newline="", encoding="utf-8") as f:
                        w = csv.writer(f)
                        w.writerow(cols)
                        f.flush()
                        os.fsync(f.fileno())
                except Exception:
                    return False

            # Append con retry
            for n in range(attempts):
                try:
                    with open(ruta, "a", newline="", encoding="utf-8") as f:
                        w = csv.writer(f)
                        w.writerow(row_vals + [label])
                        f.flush()
                        os.fsync(f.fileno())

                    _save_last_sig(bot, sig)
                    return True

                except PermissionError:
                    time.sleep(base_sleep * (n + 1) + random.uniform(0, 0.07))
                    continue
                except Exception:
                    break

        return False

    except Exception:
        return False
        
# === Canonización: aunque existan duplicados en el archivo, esta es la versión oficial ===
anexar_incremental_desde_bot = _anexar_incremental_desde_bot_CANON
       
# === FIN BLOQUE 6 ===

# === BLOQUE 7 — ORDEN DE REAL Y CONTROL DE TOKEN ===
# === ORDEN DE REAL (handshake maestro→bot) ===
ORDEN_DIR = "orden_real"

def _ensure_dir(p):
    try:
        os.makedirs(p, exist_ok=True)
    except Exception as e:
        print(f"⚠️ Falló creación de dir {p}: {e}")

def _atomic_write(path: str, text: str):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def path_orden(bot: str) -> str:
    _ensure_dir(ORDEN_DIR)
    return os.path.join(ORDEN_DIR, f"{bot}.json")

# === PATCH: REAL INMEDIATO EN HUD AL EMITIR ORDEN (sin esperar compra) ===
# Objetivo:
# - Al emitir una ORDEN manual (bot+ciclo), reservar REAL y mostrarlo YA en HUD.
# - Evitar recursión/doble llamada.
# - Si activar_real_inmediato se llama desde otro flujo (no orden_real),
#   asegurar que el bot tenga también su orden_real.json escrita (sin recursión).

_last_real_push_ts = {bot: 0.0 for bot in BOT_NAMES}

def limpiar_orden_real(bot: str):
    """
    Evita re-entradas fantasma:
    si se liberó REAL, la orden ya no debe quedar viva.
    """
    try:
        p = path_orden(bot)
        if os.path.exists(p):
            os.remove(p)
    except Exception:
        pass

def _set_ui_token_holder(holder: str | None):
    """
    Sincroniza UI + estado interno:
    - token (texto REAL/DEMO)
    - trigger_real (lógica)
    Además: si un bot DEJA de ser holder REAL, limpia su estado REAL residual
    para evitar "REAL fantasma" y escudos pegados.
    """
    try:
        now = time.time()
        for b in BOT_NAMES:
            # ultra defensivo: si por algo falta el dict del bot, lo crea
            if b not in estado_bots or not isinstance(estado_bots.get(b), dict):
                estado_bots[b] = {}

            is_holder = bool(holder) and (b == holder)
            prev_token = estado_bots[b].get("token", "DEMO")

            # UI base
            estado_bots[b]["token"] = "REAL" if is_holder else "DEMO"
            estado_bots[b]["trigger_real"] = True if is_holder else False

            # Si dejó de ser REAL, limpiar residuos (solo si antes era REAL)
            if (not is_holder) and (prev_token == "REAL"):
                estado_bots[b]["modo_real_anunciado"] = False
                estado_bots[b]["real_activado_en"] = 0.0

                # micro-colchón anti-carreras: evitamos leer cierres viejos justo al soltar token
                estado_bots[b]["ignore_cierres_hasta"] = now + 1.5

                estado_bots[b]["fuente"] = None

                # IA / pending
                estado_bots[b]["ia_senal_pendiente"] = False
                estado_bots[b]["ia_prob_senal"] = None

                # Remate
                estado_bots[b]["remate_active"] = False
                estado_bots[b]["remate_start"] = None
                estado_bots[b]["remate_reason"] = ""

                # Ciclo vuelve a default (solo al perder REAL)
                estado_bots[b]["ciclo_actual"] = 1

    except Exception:
        pass

def _enforce_single_real_standby(owner: str | None):
    """
    Si hay owner REAL activo, deja a los demás bots en standby estricto:
    - token DEMO visual
    - sin señal IA pendiente
    """
    try:
        if owner not in BOT_NAMES:
            return
        for b in BOT_NAMES:
            if b == owner:
                continue
            estado_bots[b]["token"] = "DEMO"
            estado_bots[b]["ia_senal_pendiente"] = False
            estado_bots[b]["ia_prob_senal"] = None
    except Exception:
        pass

def _enforce_single_real_standby(owner: str | None):
    """
    Si hay owner REAL activo, deja a los demás bots en standby estricto:
    - token DEMO visual
    - sin señal IA pendiente
    """
    try:
        if owner not in BOT_NAMES:
            return
        for b in BOT_NAMES:
            if b == owner:
                continue
            estado_bots[b]["token"] = "DEMO"
            estado_bots[b]["ia_senal_pendiente"] = False
            estado_bots[b]["ia_prob_senal"] = None
    except Exception:
        pass

def _enforce_single_real_standby(owner: str | None):
    """
    Si hay owner REAL activo, deja a los demás bots en standby estricto:
    - token DEMO visual
    - sin señal IA pendiente
    """
    try:
        if owner not in BOT_NAMES:
            return
        for b in BOT_NAMES:
            if b == owner:
                continue
            estado_bots[b]["token"] = "DEMO"
            estado_bots[b]["ia_senal_pendiente"] = False
            estado_bots[b]["ia_prob_senal"] = None
    except Exception:
        pass

def _escribir_orden_real_raw(bot: str, ciclo: int):
    """
    Escritura RAW de orden_real (sin activar_real_inmediato, sin recursión).
    """
    ciclo = max(1, min(int(ciclo), MAX_CICLOS))
    payload = {"bot": bot, "ciclo": ciclo, "ts": time.time()}
    try:
        _atomic_write(path_orden(bot), json.dumps(payload, ensure_ascii=False))
        agregar_evento(f"📝 Orden REAL escrita para {bot}: ciclo #{ciclo}")
    except Exception as e:
        try:
            agregar_evento(f"⚠️ Falló escritura de orden para {bot}: {e}")
        except Exception:
            pass

def activar_real_inmediato(bot: str, ciclo: int, origen: str = "orden_real"):

    """
    Reserva REAL y actualiza HUD de forma INMEDIATA.

    Regla anti “órdenes fantasma”:
    - SOLO si origen == "manual" se auto-escribe orden_real.json aquí.
    - Si la orden viene por escribir_orden_real(...), ese wrapper YA escribe el JSON.
    - Flujos de sync/UI/token jamás deben escribir orden_real.json.
    """
    global LIMPIEZA_PANEL_HASTA, sonido_disparado, marti_paso, REAL_OWNER_LOCK, REAL_ENTRY_BASELINE

    try:
        if bot not in BOT_NAMES:
            return

        now = time.time()

        # 🔒 No permitir reemplazar owner REAL activo por otro bot.
        # Solo se puede activar si no hay owner o si es el mismo bot.
        try:
            owner_lock = REAL_OWNER_LOCK if REAL_OWNER_LOCK in BOT_NAMES else leer_token_actual()
        except Exception:
            owner_lock = REAL_OWNER_LOCK if REAL_OWNER_LOCK in BOT_NAMES else None
        if owner_lock in BOT_NAMES and owner_lock != bot:
            try:
                agregar_evento(f"🔒 REAL bloqueado: {owner_lock.upper()} sigue activo. Ignorando intento de {bot.upper()}.")
            except Exception:
                pass
            try:
                if origen == "orden_real":
                    limpiar_orden_real(bot)
            except Exception:
                pass
            return


        # Anti doble-disparo (tecla rebotona)
        if (now - _last_real_push_ts.get(bot, 0.0)) < 0.25:
            return
        _last_real_push_ts[bot] = now

        ciclo_obj = max(1, min(int(ciclo), MAX_CICLOS))

        # Baseline REAL: a partir de aquí recién aceptamos cierres para este turno.
        try:
            REAL_ENTRY_BASELINE[bot] = int(contar_filas_csv(bot) or 0)
        except Exception:
            REAL_ENTRY_BASELINE[bot] = 0

        # Idempotencia token_sync: evita re-enganche/spam si ya está el mismo holder/ciclo.
        if origen == "token_sync":
            try:
                owner_now = leer_token_actual()
                cyc_now = int(estado_bots.get(bot, {}).get("ciclo_actual", 1) or 1)
                if owner_now == bot and cyc_now == ciclo_obj:
                    return
            except Exception:
                pass

        # ✅ Solo “manual” auto-escribe orden_real (la orden explícita)
        if origen == "manual":
            try:
                _escribir_orden_real_raw(bot, ciclo_obj)
            except Exception:
                pass

        prev_holder = None
        try:
            prev_holder = leer_token_actual()  # sincroniza UI
        except Exception:
            prev_holder = None

        # Reservar lock owner en memoria + token REAL en archivo
        REAL_OWNER_LOCK = bot

        # Reservar token REAL en archivo SOLO cuando corresponde:
        # - orden_real: orden explícita ya escrita por wrapper
        # - manual: el propio activar_real_inmediato puede escribir orden_real
        # - token_sync: sincroniza token sin tocar orden_real.json
        if origen in ("orden_real", "manual", "token_sync"):
            with file_lock():
                write_token_atomic(TOKEN_FILE, f"REAL:{bot}")



        # 2) Estado interno inmediato (HUD)
        _set_ui_token_holder(bot)
        estado_bots[bot]["trigger_real"] = True
        estado_bots[bot]["ciclo_actual"] = ciclo_obj

        # Marcas de “entrada a real”
        first_entry = not bool(estado_bots[bot].get("modo_real_anunciado", False))
        if first_entry or (prev_holder != bot):
            estado_bots[bot]["modo_real_anunciado"] = True
            estado_bots[bot]["real_activado_en"] = now
            estado_bots[bot]["ignore_cierres_hasta"] = now + 15.0

            # Snapshot visual/diagnóstico (independiente del baseline REAL)
            try:
                SNAPSHOT_FILAS[bot] = contar_filas_csv(bot)
            except Exception:
                SNAPSHOT_FILAS[bot] = 0

            # Mantener marti_paso global coherente con el ciclo elegido
            try:
                marti_paso = ciclo_obj - 1
            except Exception:
                pass

            try:
                agregar_evento(f"🚨 REAL INMEDIATO ({origen}) → {bot.upper()} | ciclo #{ciclo_obj}")
            except Exception:
                pass

            # Sonido de activación (solo si tu config lo permite)
            try:
                reproducir_evento("racha_detectada", es_demo=False, dentro_gatewin=True)
            except Exception:
                pass

            LIMPIEZA_PANEL_HASTA = 0
            sonido_disparado = False

        # 3) Redibujar panel YA
        try:
            fn_panel = globals().get("mostrar_panel", None)
            if callable(fn_panel):
                with RENDER_LOCK:
                    fn_panel()
        except Exception:
            pass

        # 4) Saldo en background (no frena UI)
        try:
            loop = asyncio.get_running_loop()
            fn_saldo = globals().get("obtener_saldo_real", None)
            if callable(fn_saldo):
                loop.create_task(fn_saldo())
            else:
                fn_refresh = globals().get("refresh_saldo_real", None)
                if callable(fn_refresh):
                    loop.create_task(fn_refresh(forzado=True))
        except Exception:
            pass

        # 5) Log de promociones
        try:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            with open("registro_promociones.txt", "a", encoding="utf-8") as log:
                log.write(f"{timestamp} - Token REAL (inmediato) asignado a {bot} (ciclo {ciclo_obj})\n")
        except Exception:
            pass

    except Exception:
        pass

def escribir_orden_real(bot: str, ciclo: int) -> bool:
    global REAL_OWNER_LOCK
    """
    Wrapper oficial:
    - Escribe orden_real.json (RAW)
    - Activa REAL inmediato en HUD + token file
    """
    ciclo = max(1, min(int(ciclo), MAX_CICLOS))

    # 🔒 No crear orden si ya hay otro owner REAL activo.
    try:
        owner_lock = REAL_OWNER_LOCK if REAL_OWNER_LOCK in BOT_NAMES else leer_token_actual()
    except Exception:
        owner_lock = REAL_OWNER_LOCK if REAL_OWNER_LOCK in BOT_NAMES else None

    if owner_lock in BOT_NAMES and owner_lock != bot:
        try:
            agregar_evento(f"🔒 Orden REAL bloqueada para {bot.upper()}: {owner_lock.upper()} está activo.")
        except Exception:
            pass
        return False

    # ✅ Auditoría Real vs Ficticia: abrir señal SOLO si esta orden está respaldada por IA (prob >= umbral)
    try:
        st = estado_bots.get(str(bot), {}) if isinstance(estado_bots, dict) else {}
        prob_sig = st.get("prob_ia")
        modo_sig = str(st.get("modo_ia") or "").upper()
        thr_sig = float(get_umbral_operativo())
        if isinstance(prob_sig, (int, float)) and modo_sig not in ("", "OFF", "0") and float(prob_sig) >= thr_sig:
            ep_sig = ia_audit_get_last_pre_epoch(str(bot))
            if isinstance(ep_sig, (int, float)) and int(ep_sig) > 0:
                log_ia_open(str(bot), int(ep_sig), float(prob_sig), float(thr_sig), str(st.get("fuente") or "ORDEN_REAL"))
    except Exception:
        pass

    _escribir_orden_real_raw(bot, ciclo)
    activar_real_inmediato(bot, ciclo, origen="orden_real")

    owner_after = REAL_OWNER_LOCK if REAL_OWNER_LOCK in BOT_NAMES else leer_token_actual()
    return owner_after == bot
# === FIN PATCH REAL INMEDIATO ===
# === IA ACK (handshake maestro→bot: confirma que el PRE-TRADE ya fue evaluado) ===
IA_ACK_DIR = "ia_ack"
_LAST_IA_ACK_HEARTBEAT_TS = 0.0

def path_ia_ack(bot: str) -> str:
    _ensure_dir(IA_ACK_DIR)
    return os.path.join(IA_ACK_DIR, f"{bot}.json")

def escribir_ia_ack(bot: str, epoch: int | None, prob: float | None, modo_ia: str, meta: dict | None):
    """
    Escribe un ACK por-bot para que el bot muestre la prob IA asociada a su PRE.
    Incluye:
      - prob: prob calibrada (0..1) o None
      - prob_raw: prob sin calibrar (0..1), si está disponible
      - calib_factor: factor aplicado (si aplica)
      - auc / thr / reliable desde model_meta
    """
    try:
        ack_path = path_ia_ack(bot)
        os.makedirs(os.path.dirname(ack_path), exist_ok=True)

        st = estado_bots.get(str(bot), {}) if isinstance(estado_bots, dict) else {}

        payload = {
            "bot": str(bot),
            "epoch": int(epoch) if epoch is not None else 0,
            "prob": float(prob) if isinstance(prob, (int, float)) else None,
            # prob_hud/modo_hud = valor vigente que pinta el HUD del maestro (fuente visual principal)
            "prob_hud": float(st.get("prob_ia")) if isinstance(st.get("prob_ia"), (int, float)) else None,
            "modo_hud": str(st.get("modo_ia", "off") or "off").upper(),
            "prob_raw": float(st.get("prob_ia_raw")) if isinstance(st.get("prob_ia_raw"), (int, float)) else None,
            "calib_factor": float(st.get("cal_factor")) if isinstance(st.get("cal_factor"), (int, float)) else None,
            "auc": float((meta or {}).get("auc", 0.0) or 0.0),
            "thr": float((meta or {}).get("threshold", 0.0) or 0.0),
            "reliable": bool((meta or {}).get("reliable", True)),
            "modo": str(modo_ia).upper() if modo_ia else "OFF",
            "ts": time.time()
        }

        with open(ack_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def refrescar_ia_ack_desde_hud(intervalo_s: float = 1.0):
    """
    Heartbeat de ACK: mantiene `ia_ack/<bot>.json` sincronizado con el HUD.
    Objetivo: que los bots vean la prob IA vigente del maestro durante GateWin,
    incluso si no entraron filas nuevas de CSV en ese instante.
    """
    global _LAST_IA_ACK_HEARTBEAT_TS
    now = time.time()
    if (now - float(_LAST_IA_ACK_HEARTBEAT_TS or 0.0)) < float(intervalo_s):
        return

    meta = leer_model_meta() or {}
    for bot in BOT_NAMES:
        try:
            st = estado_bots.get(bot, {}) if isinstance(estado_bots, dict) else {}
            ep = st.get("ultimo_epoch_pretrade", 0)
            if ep is None:
                ep = 0
            ep = int(float(ep)) if str(ep).strip() != "" else 0
            if ep <= 0:
                continue

            p = st.get("prob_ia", None)
            modo = str(st.get("modo_ia", "off") or "off").upper()
            escribir_ia_ack(bot, ep, p if isinstance(p, (int, float)) else None, modo, meta)
        except Exception:
            continue

    _LAST_IA_ACK_HEARTBEAT_TS = now
# Leer token actual
def leer_token_actual():
    """
    Lee token_actual.txt y además sincroniza el HUD (estado_bots[*]["token"])
    para que REAL/DEMO se refleje sin esperar compra del bot.
    Prioriza lock en memoria para evitar parpadeos DEMO durante REAL en curso.
    """
    holder = REAL_OWNER_LOCK if REAL_OWNER_LOCK in BOT_NAMES else None

    # Si ya hay owner REAL en memoria, mantenemos sincronía visual inmediata.
    if holder in BOT_NAMES:
        _set_ui_token_holder(holder)
        _enforce_single_real_standby(holder)
        return holder

    if not os.path.exists(TOKEN_FILE):
        _set_ui_token_holder(None)
        return None
    try:
        with open(TOKEN_FILE, encoding="utf-8", errors="replace") as f:
            linea = (f.read() or "").strip()
        if linea.startswith("REAL:"):
            bot_name = linea.split(":", 1)[1].strip()
            if bot_name in BOT_NAMES:
                holder = bot_name
            elif bot_name == "none":
                holder = None
        _set_ui_token_holder(holder)
        if holder in BOT_NAMES:
            _enforce_single_real_standby(holder)
        return holder
    except Exception as e:
        try:
            print(f"⚠️ Error leyendo token: {e}")
        except Exception:
            pass
        fallback = REAL_OWNER_LOCK if REAL_OWNER_LOCK in BOT_NAMES else None
        _set_ui_token_holder(fallback)
        if fallback in BOT_NAMES:
            _enforce_single_real_standby(fallback)
        return fallback

# Escribir token actual
async def escribir_token_actual(bot):
    """
    Sync UI/token: refleja REAL en HUD y token file.
    ⚠️ Regla: este flujo NO debe generar orden_real.json.
    """
    try:
        try:
            _ = leer_token_actual()  # sincroniza UI
        except Exception:
            pass

        ciclo_objetivo = estado_bots.get(bot, {}).get("ciclo_actual", 1)
        try:
            ciclo_objetivo = int(ciclo_objetivo)
        except Exception:
            ciclo_objetivo = 1

        # ✅ origen "sync_ui": NO debe escribir orden_real.json
        activar_real_inmediato(bot, ciclo_objetivo, origen="token_sync")

        # No bloqueamos: saldo en background
        try:
            asyncio.create_task(obtener_saldo_real())
        except Exception:
            try:
                asyncio.create_task(refresh_saldo_real(forzado=True))
            except Exception:
                pass

    except Exception:
        pass

# Activar remate
def activar_remate(bot: str, reason: str):
    if not estado_bots[bot]["remate_active"]:
        estado_bots[bot]["remate_active"] = True
        estado_bots[bot]["remate_start"] = datetime.now()
        estado_bots[bot]["remate_reason"] = reason

# Cerrar por WIN
def cerrar_por_win(bot: str, reason: str):
    global REAL_OWNER_LOCK
    # Limpieza total de “estado REAL” para evitar REAL fantasma
    try:
        estado_bots[bot]["token"] = "DEMO"
        estado_bots[bot]["trigger_real"] = False
        estado_bots[bot]["ciclo_actual"] = 1
        estado_bots[bot]["modo_real_anunciado"] = False
        estado_bots[bot]["fuente"] = None
        estado_bots[bot]["real_activado_en"] = 0.0
        estado_bots[bot]["ignore_cierres_hasta"] = 0.0


        # Flags IA/pending (si quedó algo colgado)
        estado_bots[bot]["ia_senal_pendiente"] = False
        estado_bots[bot]["ia_prob_senal"] = None

        # Remate limpio
        estado_bots[bot]["remate_active"] = False
        estado_bots[bot]["remate_start"] = None
        estado_bots[bot]["remate_reason"] = ""

    except Exception:
        pass

    # Liberar token global REAL
    REAL_OWNER_LOCK = None
    try:
        with file_lock():
            write_token_atomic(TOKEN_FILE, "REAL:none")
    except Exception:
        pass
    # Limpiar orden REAL para evitar re-entradas fantasma
    try:
        limpiar_orden_real(bot)
    except Exception:
        pass
    
    # Sync inmediato del HUD/token para evitar “REAL fantasma”
    try:
        _set_ui_token_holder(None)
    except Exception:
        pass
    
    # Resync de snapshots y panel
    try:
        REAL_ENTRY_BASELINE[bot] = 0
        SNAPSHOT_FILAS[bot] = contar_filas_csv(bot)
    except Exception:
        pass

    try:
        OCULTAR_HASTA_NUEVO[bot] = False
    except Exception:
        pass

    try:
        agregar_evento(f"✅ WIN: REAL liberado para {bot.upper()} ({reason})")
    except Exception:
        pass

    try:
        reinicio_forzado.set()
    except Exception:
        pass

# === FIN BLOQUE 7 ===

# === BLOQUE 8 — NORMALIZACIÓN Y PUNTAJE DE ESTRATEGIA ===
# Normalizar resultado
def normalizar_resultado(texto):
    if texto is None:
        return "INDEFINIDO"

    raw = str(texto)

    # 1) Detectar símbolos ANTES de normalización ASCII (ASCII los borra)
    if any(sym in raw for sym in ("✓", "✔", "✅", "🟢")):
        return "GANANCIA"
    if any(sym in raw for sym in ("✗", "❌", "🔴", "🟥")):
        return "PÉRDIDA"

    # 2) Normalización de texto (acentos/encoding raros)
    raw = raw.replace("Ã‰", "É").replace("PÃ‰RDIDA", "PÉRDIDA")
    t = normalize("NFKD", raw).encode("ASCII", "ignore").decode("ASCII").strip().upper()

    # Nota: después de ASCII, "PÉRDIDA" se vuelve "PERDIDA"
    if "PERD" in t or "LOSS" in t:
        return "PÉRDIDA"
    if "GAN" in t or "WIN" in t:
        return "GANANCIA"
    return "INDEFINIDO"
def normalizar_trade_status(ts):
    """
    Normaliza trade_status a canónico del Maestro:
      - "CERRADO"   (CERRADO/CLOSED/SETTLED/etc.)
      - "PRE_TRADE" (PRE_TRADE/PENDIENTE/PENDING/OPEN/ABIERTO/etc.)
      - otros: upper limpio (compat)
    """
    try:
        if ts is None:
            return ""
        s = str(ts).strip().upper()
        if s in ("", "NAN", "NONE"):
            return ""

        # --- CIERRES ---
        if s in (
            "CERRADO", "CERRADA",
            "CLOSED", "CLOSE",
            "SETTLED",
            "SOLD",
            "EXPIRED", "EXPIRE",
            "CANCELLED", "CANCELED",
            "VOID"
        ):
            return "CERRADO"

        # --- PRE / PENDIENTE / ABIERTO ---
        if s in (
            "PRE_TRADE", "PRETRADE",
            "PENDIENTE", "PENDING",
            "OPEN", "ABIERTO", "ABIERTA",
            "IN_PROGRESS", "INPROGRESS", "RUNNING"
        ):
            return "PRE_TRADE"

        return s
    except Exception:
        return ""

def canonicalizar_campos_bot_maestro(row_dict: dict | None):
    """
    Mapeo central BOT -> Maestro para mantener un único esquema canónico.

    Este normalizador NO inventa datos: solo renombra/duplica aliases conocidos
    hacia los nombres oficiales que consume la IA del maestro.
    """
    out = dict(row_dict or {})

    alias_map = {
        "direction": ("direccion",),
        "ciclo_martingala": ("ciclo",),
        "cruce_sma": ("cruce",),
        "payout_multiplier": ("payout_decimal_rounded",),
    }

    for canon, aliases in alias_map.items():
        if out.get(canon) in (None, ""):
            for a in aliases:
                if out.get(a) not in (None, ""):
                    out[canon] = out.get(a)
                    break

    return out

# ==========================================================
# Payout/ROI — Normalización consistente (SIN confundir %)
# Convención:
# - payout_total: total recibido (ej 1.95, 15.62)
# - payout (feature IA): ROI = (payout_total / monto) - 1  en [0.0, 1.5]
# ==========================================================
# Reutilizamos _safe_float (BLOQUE 6) para evitar duplicados
_safe_float_local = _safe_float
                                    
def _norm_01(x, lo=0.0, hi=3.5):
    """
    Normaliza x a [0..1] usando rango [lo..hi].
    Si no se puede convertir, devuelve 0.0
    """
    try:
        v = _safe_float_local(x)
        if v is None:
            return 0.0
        v = float(v)
        if not math.isfinite(v):
            return 0.0
        if hi <= lo:
            return 0.0
        t = (v - lo) / (hi - lo)
        return max(0.0, min(1.0, t))
    except Exception:
        return 0.0
      
def extraer_payout_multiplier(row_dict_full: dict):
    """
    payout_multiplier = payout_total / monto (ratio_total).

    Fuentes (en orden):
      1) payout_multiplier (nuevo BOT)
      2) payout_decimal_rounded (legacy ratio)
      3) payout legacy SOLO si parece RATIO (>=1.05 y <=3.50) o si con monto cuadra como total
      4) payout_total / monto

    Blindaje:
      - Si 'payout' parece ROI feature (0..1.5), se IGNORA como ratio/total.
      - Si 'payout' < 1.05, NO se acepta como ratio (en Deriv el ratio típico >1).
    """
    mult = _safe_float_local(row_dict_full.get("payout_multiplier"))
    if mult is not None and mult > 0:
        return mult

    mult = _safe_float_local(row_dict_full.get("payout_decimal_rounded"))
    if mult is not None and mult > 0:
        return mult

    monto = _safe_float_local(row_dict_full.get("monto"))
    p = _safe_float_local(row_dict_full.get("payout"))  # legacy (a veces ratio, a veces total, a veces ROI feature)

    # 🔒 Si payout parece ROI-feature (0..1.5), NO usarlo para ratio/total
    if p is not None and 0.0 <= p <= 1.5:
        p = None

    if p is not None and p > 0:
        if monto is not None and monto > 0:
            # Caso 1: p parece ratio típico (1.05..3.50)
            if 1.05 <= p <= 3.50:
                return p

            # Caso 2: p parece total (grande)
            if p > 3.50:
                return p / monto

            # Caso 3: p es "total pequeño" (monto<1) donde p/monto cae como ratio
            # Ej: monto=0.5, payout_total=0.975 -> ratio=1.95
            cand = p / monto
            if 1.05 <= cand <= 3.50:
                return cand

            return None
        else:
            # Sin monto: solo aceptar si parece ratio típico
            if 1.05 <= p <= 3.50:
                return p
            return None

    # Fallback por payout_total explícito
    pay_total = _safe_float_local(row_dict_full.get("payout_total"))
    if pay_total is None:
        pay_total = _safe_float_local(row_dict_full.get("payout"))  # legacy total a veces

    if pay_total is not None and monto is not None and monto > 0:
        return pay_total / monto

    return None


def extraer_payout_total(row_dict_full: dict):
    """
    payout_total = retorno total (stake + profit).

    Fuentes (en orden):
      1) payout_total (nuevo BOT)
      2) payout legacy si parece total (>3.5)
      3) monto * payout_multiplier (incluye payout_decimal_rounded o ratio legacy)
      4) si payout legacy parece total (monto<1), usarlo como total
    """
    pay_total = _safe_float_local(row_dict_full.get("payout_total"))
    if pay_total is not None and pay_total > 0:
        return pay_total

    monto = _safe_float_local(row_dict_full.get("monto"))
    p = _safe_float_local(row_dict_full.get("payout"))  # legacy

    if p is not None and p > 0:
        # Claramente total
        if p > 3.50:
            return p

        # Caso monto<1: payout_total puede ser 0.975, que cae <=3.5
        if monto is not None and monto > 0:
            # Si p/monto cae como ratio típico, p es total
            cand = p / monto
            if 0.90 <= cand <= 3.50:
                return p  # total

    mult = extraer_payout_multiplier(row_dict_full)
    if monto is not None and mult is not None and monto > 0 and mult > 0:
        return monto * mult

    return None


def calcular_roi_desde_total_y_monto(payout_total: float, monto: float):
    if payout_total is None or monto is None or monto <= 0:
        return None
    return (payout_total / monto) - 1.0


def calcular_payout_feature(row_dict_full: dict):
    """
    Feature IA 'payout' = ROI = (payout_multiplier - 1).
    Fallback: (payout_total / monto) - 1 si no hay multiplier.
    """
    mult = extraer_payout_multiplier(row_dict_full)

    # ✅ Rescate: algunos logs legacy guardan ratio en "payout" (ej 1.20, 1.35)
    # Esto NO confunde ROI real (ROI típico sería 0.20, no 1.20).
    if mult is None:
        try:
            p_leg = row_dict_full.get("payout", None)
            p_leg = float(p_leg) if p_leg is not None else None
            if p_leg is not None and math.isfinite(p_leg) and (1.05 <= p_leg <= 3.50):
                mult = float(p_leg)
        except Exception:
            pass

    if mult is not None:
        roi = float(mult) - 1.0
    else:
        payout_total = extraer_payout_total(row_dict_full)
        monto = _safe_float_local(row_dict_full.get("monto"))
        roi = calcular_roi_desde_total_y_monto(payout_total, monto)


    if roi is None:
        return None

    # clamps defensivos
    if roi < 0:
        roi = 0.0
    if roi > 1.5:
        roi = 1.5

    return roi

def normalizar_roi_0a1(roi):
    """Convierte ROI [0..1.5] a [0..1] cuando necesitas un 'factor'."""
    try:
        if roi is None:
            return 0.0
        roi = float(roi)
        if not math.isfinite(roi):
            return 0.0
        roi = max(0.0, min(roi, 1.5))
        return roi / 1.5
    except Exception:
        return 0.0
  
# Nueva: Clipping de features a rangos lógicos (para blindaje contra outliers)
def clip_feature_values(fila_dict, feature_names):
    ranges = {
        "rsi_9": (0, 100),
        "rsi_14": (0, 100),
        "cruce_sma": (-1, 1),
        "breakout": (0, 1),
        "rsi_reversion": (0, 1),
        "racha_actual": (-50, 50),
        "payout": (0, 1.5),
        "puntaje_estrategia": (0, 1),
        "volatilidad": (0, 1),
        "es_rebote": (0, 1),
        "hora_bucket": (0, 1),
        # sma_5 / sma_20: no clip, pero sí normalizar a float cuando se pueda
    }
    clipped = dict(fila_dict)

    for feat in feature_names:
        val = clipped.get(feat, None)

        # Normaliza a float si se puede
        try:
            if val is None or (isinstance(val, str) and val.strip() == ""):
                clipped[feat] = np.nan
                continue
            v = float(val)
            if not np.isfinite(v):
                clipped[feat] = np.nan
                continue
        except Exception:
            clipped[feat] = np.nan
            continue

        # Aplica clip solo donde hay rango definido
        if feat in ranges:
            lo, hi = ranges[feat]
            clipped[feat] = float(np.clip(v, lo, hi))
        else:
            clipped[feat] = float(v)

    return clipped
    
def calcular_volatilidad_simple(row_dict: dict) -> float:
    """
    Volatilidad 0–1 aproximada usando la separación relativa entre sma_5 y sma_20.
    Si no hay datos válidos, devuelve 0.0.
    """
    try:
        sma5 = float(row_dict.get("sma_5", 0.0) or 0.0)
        sma20 = float(row_dict.get("sma_20", 0.0) or 0.0)
    except Exception:
        return 0.0

    # Evitamos división por cero
    base = abs(sma20) if abs(sma20) > 1e-6 else 1.0
    spread_pct = abs(sma5 - sma20) / base

    # Forzamos a rango [0, 1]
    if spread_pct < 0.0:
        spread_pct = 0.0
    if spread_pct > 1.0:
        spread_pct = 1.0

    return spread_pct
    
# --- Helper: detectar rebote tras racha larga negativa --- 
def calcular_es_rebote(row_dict):
    """
    es_rebote = 1 si:
      - |racha_actual| >= 4  (racha larga, positiva o negativa)
      - y hay señal de giro (RSI reversión alta, breakout fuerte o cruce+fuerza a favor)
    En caso contrario, 0.
    """
    try:
        racha = float(row_dict.get("racha_actual", 0) or 0.0)
    except Exception:
        racha = 0.0

    # Solo consideramos rebote cuando la racha (ganadora o perdedora) es larga
    if abs(racha) < 4:
        return 0.0

    def _safe_num(key, default=0.0):
        try:
            return float(row_dict.get(key, default) or default)
        except Exception:
            return default

    rsi_rev  = _safe_num("rsi_reversion", 0.0)
    breakout = _safe_num("breakout", 0.0)
    cruce    = _safe_num("cruce_sma", 0.0)
    fuerza   = _safe_num("fuerza_vela", 0.0)

    # "Señal de giro": cualquiera de estos empuja a rebote
    giro_flag = (
        rsi_rev >= 0.60 or
        breakout >= 0.50 or
        (cruce > 0 and fuerza > 0)
    )

    return 1.0 if giro_flag else 0.0

def calcular_hora_bucket(row_dict):
    """
    Devuelve un valor 0–1 según la hora del día:
      0:  0–6   (madrugada)
      1:  6–12  (mañana)
      2:  12–18 (tarde)
      3:  18–24 (noche)

    Si no se puede parsear la hora, devuelve 0.5 (neutral).

    Claves soportadas (prioridad):
      - ts (ISO con TZ, ej: "2025-12-05T09:24:04.531328+00:00")  -> se convierte a America/Lima
      - epoch / timestamp (segundos o ms)                         -> se convierte a America/Lima
      - fecha (local, ej: "2025-12-05 04:24:04")                  -> se asume America/Lima (sin utc=True)
      - hora  (ej: "14:58" / "14:58:25")
    """
    def _missing(v):
        if v is None:
            return True
        try:
            if pd.isna(v):
                return True
        except Exception:
            pass
        return isinstance(v, str) and v.strip() == ""

    def _bucket(hour: int):
        if 0 <= hour < 6:
            b = 0
        elif 6 <= hour < 12:
            b = 1
        elif 12 <= hour < 18:
            b = 2
        else:
            b = 3
        return b / 3.0

    # 1) ts ISO con zona (mejor fuente)
    v = row_dict.get("ts", None)
    if not _missing(v) and isinstance(v, str):
        try:
            dt = pd.to_datetime(v, utc=True, errors="coerce")
            if dt is not None and pd.notna(dt):
                try:
                    dt = dt.tz_convert("America/Lima")
                except Exception:
                    pass
                return _bucket(int(dt.hour))
        except Exception:
            pass

    # 2) epoch/timestamp numérico o string numérico
    for k in ("epoch", "timestamp"):
        v = row_dict.get(k, None)
        if _missing(v):
            continue
        try:
            # aceptar "1764926644" (string)
            val = float(v)
            if val > 1e12:
                val = val / 1000.0
            if val > 1e9:
                dt = pd.to_datetime(val, unit="s", utc=True, errors="coerce")
                if dt is not None and pd.notna(dt):
                    try:
                        dt = dt.tz_convert("America/Lima")
                    except Exception:
                        pass
                    return _bucket(int(dt.hour))
        except Exception:
            pass

    # 3) fecha local (NO utc=True)
    v = row_dict.get("fecha", None)
    if not _missing(v) and isinstance(v, str):
        try:
            dt = pd.to_datetime(v, errors="coerce")  # naive
            if dt is not None and pd.notna(dt):
                # asumimos que 'fecha' ya está en hora local Lima
                return _bucket(int(dt.hour))
        except Exception:
            pass

    # 4) hora "HH:MM[:SS]"
    v = row_dict.get("hora", None)
    if not _missing(v) and isinstance(v, str):
        try:
            s = v.strip()
            if ":" in s:
                h = int(s.split(":")[0])
                if 0 <= h <= 23:
                    return _bucket(h)
        except Exception:
            pass

    return 0.5

# === Nuevo: cálculo enriquecido de puntaje_estrategia normalizado (0–1) ===
def calcular_puntaje_estrategia_normalizado(fila: dict) -> float:
    """
    Puntaje 0..1 usando señales + RSI + ROI (payout como ROI [0..1.5]).
    Robusto a valores 1.0/0.0 y strings.
    """
    def as01(x):
        try:
            if isinstance(x, str):
                x = x.strip().lower()
                if x in ("1", "true", "yes", "y"):
                    return 1.0
                if x in ("0", "false", "no", "n", ""):
                    return 0.0
            v = float(x)
            return 1.0 if v >= 0.5 else 0.0
        except Exception:
            return 0.0

    score = 0.0

    breakout      = as01(fila.get("breakout", 0))
    cruce_sma     = as01(fila.get("cruce_sma", 0))
    rsi_reversion = as01(fila.get("rsi_reversion", 0))
    es_rebote     = as01(fila.get("es_rebote", 0))

    score += breakout * 0.30
    score += cruce_sma * 0.25
    score += rsi_reversion * 0.20
    score += es_rebote * 0.15

    # RSI zona caliente (bonus pequeño)
    try:
        rsi_9 = float(fila.get("rsi_9", 50.0) or 50.0)
        rsi_14 = float(fila.get("rsi_14", 50.0) or 50.0)
        if rsi_9 >= 70 or rsi_14 >= 70:
            score += 0.05
        if rsi_9 <= 30 or rsi_14 <= 30:
            score += 0.05
    except Exception:
        pass

    # ROI (feature payout) en 0..1
    # Si ya viene como ROI (0..1.5) úsalo directo; si no, recién calcula.
    roi = fila.get("payout", None)
    try:
        roi = float(roi)
        # Si parece ROI válido
        if (not math.isfinite(roi)) or roi < 0 or roi > 1.5:
            roi = None
    except Exception:
        roi = None

    if roi is None:
        roi = calcular_payout_feature(fila)  # fallback

    roi01 = 0.0
    try:
        roi01 = max(0.0, min(float(roi or 0.0), 1.5)) / 1.5
    except Exception:
        roi01 = 0.0

    score += roi01 * 0.05

    if not math.isfinite(score):
        score = 0.0
    return max(0.0, min(score, 1.0))

# Leer última fila válida
def leer_ultima_fila_con_resultado(bot: str) -> tuple[dict | None, int | None]:
    """
    Devuelve (fila_dict_features_pretrade, label) emparejando:
      - LABEL: desde el último trade CERRADO (GANANCIA/PÉRDIDA)
      - FEATURES: desde el PRE_TRADE/PENDIENTE del mismo epoch (o el más cercano ANTES)

    FIX CLAVE:
      - racha_actual se RECALCULA desde el historial de CIERRES ANTERIORES,
        excluyendo SIEMPRE el cierre actual (anti-contaminación).
      - payout(feature) prioriza ratio cotizado (mult/decimal_rounded).
        Si existe payout_total en PRE_TRADE, se usa como fallback seguro (más rango real).
    """
    try:
        ruta = f"registro_enriquecido_{bot}.csv"
        if not os.path.exists(ruta):
            return None, None

        df = None
        for enc in ("utf-8", "latin-1", "windows-1252"):
            try:
                df = pd.read_csv(ruta, sep=",", encoding=enc, engine="python", on_bad_lines="skip")
                break
            except Exception:
                continue

        if df is None or df.empty:
            return None, None
        if "resultado" not in df.columns:
            return None, None

        # Normalizar resultado robusto
        df["resultado_norm"] = df["resultado"].apply(normalizar_resultado)

        # Normalizar trade_status de forma canónica (evita CLOSED/SETTLED vs CERRADO)
        # IMPORTANTÍSIMO:
        # - No basta con que exista la columna: debe haber valores reales.
        # - normalizar_trade_status() canoniza a: "CERRADO" o "PRE_TRADE" (o "")
        ts_source_col = None
        if "trade_status_norm" in df.columns:
            ts_source_col = "trade_status_norm"
        elif "trade_status" in df.columns:
            ts_source_col = "trade_status"

        if ts_source_col:
            try:
                df["trade_status_norm"] = df[ts_source_col].apply(normalizar_trade_status)
            except Exception:
                df["trade_status_norm"] = ""
        else:
            df["trade_status_norm"] = ""

        # has_trade_status = hay valores reales (no solo columna vacía)
        try:
            has_trade_status = df["trade_status_norm"].astype(str).str.strip().ne("").any()
        except Exception:
            has_trade_status = False

        def _calc_racha_pretrade(_df: pd.DataFrame, _idx_close: int) -> float:
            """
            racha_actual PRE-TRADE:
              - Usa SOLO cierres (GANANCIA/PÉRDIDA)
              - Usa SOLO filas con índice < idx_close (excluye el cierre actual)
              - Si hay trade_status usable, acepta:
                    * CERRADO
                    * "" (legacy sin status) PERO solo si es cierre real por resultado_norm
              - Devuelve racha firmada: +N (wins), -N (losses), 0 si no hay historial
            """
            try:
                try:
                    d = _df.loc[_df.index < _idx_close].copy()
                except Exception:
                    d = _df.copy()

                d = d[d["resultado_norm"].isin(["GANANCIA", "PÉRDIDA"])].copy()
                if d.empty:
                    return 0.0

                if has_trade_status and "trade_status_norm" in d.columns:
                    try:
                        ts = d["trade_status_norm"].astype(str).str.strip()
                        d = d[(ts.eq("CERRADO")) | (ts.eq(""))].copy()
                    except Exception:
                        d = d[d["trade_status_norm"].eq("CERRADO")].copy()

                if d.empty:
                    return 0.0

                seq = d["resultado_norm"].tolist()
                last = seq[-1]
                streak = 0
                for r in reversed(seq):
                    if r == last:
                        streak += 1
                    else:
                        break

                val = float(streak if last == "GANANCIA" else -streak)
                if val > 50:
                    val = 50.0
                if val < -50:
                    val = -50.0
                return val
            except Exception:
                return 0.0

        # 1) Último cierre válido (label)
        df_cerr = df[df["resultado_norm"].isin(["GANANCIA", "PÉRDIDA"])].copy()

        # Si hay trade_status usable, filtramos CERRADO o "" (legacy)
        if has_trade_status:
            try:
                ts = df_cerr["trade_status_norm"].astype(str).str.strip()
                df_cerr = df_cerr[(ts.eq("CERRADO")) | (ts.eq(""))].copy()
            except Exception:
                df_cerr = df_cerr[df_cerr["trade_status_norm"].eq("CERRADO")].copy()

        if df_cerr.empty:
            return None, None

        idx_close = int(df_cerr.index[-1])
        r_close = df.loc[idx_close].to_dict()
        epoch_close = r_close.get("epoch", None)

        res_norm = r_close.get("resultado_norm", None)
        if res_norm not in ("GANANCIA", "PÉRDIDA"):
            res_norm = normalizar_resultado(r_close.get("resultado"))
        label = 1 if res_norm == "GANANCIA" else 0

        # 2) Buscar PRE_TRADE correspondiente (SIN futuro)
        pre_row = None
        pre_idx = None

        if has_trade_status:
            # ✅ OJO: después del normalizador solo existe PRE_TRADE (no PENDIENTE/OPEN/ABIERTO)
            df_pending = df[df["trade_status_norm"].eq("PRE_TRADE")].copy()

            # evitar futuro: solo filas con índice <= cierre
            try:
                df_pending = df_pending.loc[df_pending.index <= idx_close]
            except Exception:
                pass

            # prioridad: mismo epoch
            if epoch_close is not None and "epoch" in df_pending.columns:
                try:
                    ep = pd.to_numeric(df_pending["epoch"], errors="coerce")
                    ec = float(epoch_close)
                    same_ep = df_pending[ep.notna() & (ep == ec)].copy()
                except Exception:
                    same_ep = df_pending[df_pending["epoch"] == epoch_close].copy()

                if not same_ep.empty:
                    pre_idx = int(same_ep.index[-1])
                    pre_row = df.loc[pre_idx].to_dict()

            # fallback: último PRE_TRADE antes del cierre
            if pre_row is None and not df_pending.empty:
                pre_idx = int(df_pending.index[-1])
                pre_row = df.loc[pre_idx].to_dict()

        # ✅ Fallback UNIVERSAL:
        # Si no hubo PRE_TRADE (o trade_status era “usable” pero no encontró), usamos la última fila NO cierre antes del cierre.
        if pre_row is None:
            try:
                df_before = df.loc[df.index <= idx_close].copy()
            except Exception:
                df_before = df.copy()

            cand = df_before[~df_before["resultado_norm"].isin(["GANANCIA", "PÉRDIDA"])].copy()
            if not cand.empty:
                pre_idx = int(cand.index[-1])
                pre_row = df.loc[pre_idx].to_dict()
            else:
                pre_idx = int(df_before.index[-1])
                pre_row = df.loc[pre_idx].to_dict()

        if pre_row is None:
            return None, None

        row_dict_full = canonicalizar_campos_bot_maestro(pre_row)

        # 3) Asegurar monto (stake) desde PRE; si falta, tomar del cierre (monto NO filtra label)
        if ("monto" not in row_dict_full) or (row_dict_full.get("monto") in (None, "", 0, 0.0)):
            if "monto" in r_close:
                row_dict_full["monto"] = r_close.get("monto")

        # 4) Copiar ratio cotizado desde cierre SOLO si viene como multiplier/decimal_rounded (ratio seguro)
        for k in ("payout_multiplier", "payout_decimal_rounded"):
            if (k not in row_dict_full) or (row_dict_full.get(k) in (None, "", 0, 0.0)):
                if k in r_close:
                    row_dict_full[k] = r_close.get(k)

        # 4.9) Anti-leakage duro:
        # - Nunca permitir campos que puedan oler al cierre (resultado/profit).
        for k in ("ganancia_perdida", "profit", "resultado", "resultado_norm"):
            try:
                row_dict_full.pop(k, None)
            except Exception:
                pass

        # payout_total: SOLO lo permitimos si viene del PRE_TRADE/PENDIENTE (no cierre).
        # Importante: detectar CERRADO también si viene como CLOSED/SETTLED.
        try:
            ts_pre_norm = normalizar_trade_status(
                pre_row.get("trade_status_norm", None) or pre_row.get("trade_status", None)
            )
        except Exception:
            ts_pre_norm = ""

        if ts_pre_norm == "CERRADO":
            try:
                row_dict_full.pop("payout_total", None)
            except Exception:
                pass
        else:
            try:
                pt = _safe_float_local(row_dict_full.get("payout_total"))
                if pt is not None and pt <= 0:
                    row_dict_full.pop("payout_total", None)
            except Exception:
                try:
                    row_dict_full.pop("payout_total", None)
                except Exception:
                    pass


        # 5) FIX CONTAMINACIÓN: recalcular racha_actual PRE-TRADE desde historia real
        racha_safe = _calc_racha_pretrade(df, idx_close)
        try:
            old_racha = _safe_float_local(row_dict_full.get("racha_actual"))
        except Exception:
            old_racha = None

        row_dict_full["racha_actual"] = float(racha_safe)

        # opcional: avisar SOLO si había valor y cambia fuerte (útil para ver contaminación)
        try:
            if old_racha is not None and math.isfinite(float(old_racha)):
                if abs(float(old_racha) - float(racha_safe)) >= 1.0:
                    fn_evt = globals().get("agregar_evento", None)
                    if callable(fn_evt):
                        fn_evt(f"🧼 racha_actual corregida {bot}: {old_racha:.0f} → {racha_safe:.0f} (anti-contaminación)")
        except Exception:
            pass

        # Blindaje: si 'payout' existe, puede ser ROI-feature (0..1.5) o total legacy.
        # - ROI-feature: ignorar SIEMPRE (no sirve para ratio)
        # - total grande o ratio inválido: ignorar
        try:
            _p = _safe_float_local(row_dict_full.get("payout"))
            if _p is not None:
                if 0.0 <= _p <= 1.5:
                    row_dict_full["payout"] = None
                elif _p > 3.50:
                    row_dict_full["payout"] = None
                elif _p < 1.05:
                    row_dict_full["payout"] = None
        except Exception:
            pass

        # 6) payout feature (ROI) usando ratio cotizado / payout_total PRETRADE como fallback seguro
        mult = extraer_payout_multiplier(row_dict_full)

        # Si no hay ratio, inferimos por moda histórica del ratio (SOLO hasta el cierre)
        if mult is None:
            mult_moda = None
            try:
                df_hist = df.loc[df.index <= idx_close].copy()

                cand_cols = []
                if "payout_multiplier" in df_hist.columns:
                    cand_cols.append("payout_multiplier")
                if "payout_decimal_rounded" in df_hist.columns:
                    cand_cols.append("payout_decimal_rounded")

                for col in cand_cols:
                    s = pd.to_numeric(df_hist[col], errors="coerce").dropna()
                    if len(s) > 0:
                        moda = float(s.value_counts().idxmax())
                        if 1.05 < moda < 3.50:
                            mult_moda = moda
                            break
            except Exception:
                mult_moda = None

            if mult_moda is None:
                return None, None
            mult = float(mult_moda)

        try:
            pay_ok = float(mult) - 1.0
        except Exception:
            return None, None

        if not math.isfinite(pay_ok):
            return None, None

        pay_ok = max(0.0, min(pay_ok, 1.5))
        row_dict_full["payout"] = float(pay_ok)

        # 7) Completar derivados si faltan
        vol = _safe_float_local(row_dict_full.get("volatilidad"))
        if vol is None:
            vol = calcular_volatilidad_simple(row_dict_full)
        if vol is None or not math.isfinite(float(vol)):
            return None, None
        row_dict_full["volatilidad"] = float(vol)

        hb = _safe_float_local(row_dict_full.get("hora_bucket"))
        if hb is None:
            hb = calcular_hora_bucket(row_dict_full)
        if hb is None or not math.isfinite(float(hb)):
            return None, None
        row_dict_full["hora_bucket"] = float(hb)

        er = _safe_float_local(row_dict_full.get("es_rebote"))
        if er is None:
            er = calcular_es_rebote(row_dict_full)
        if er is None or not math.isfinite(float(er)):
            return None, None
        row_dict_full["es_rebote"] = float(er)

        pe = None
        try:
            pe = calcular_puntaje_estrategia_normalizado(row_dict_full)
        except Exception:
            pe = None

        if pe is None:
            pe_raw = _safe_float_local(row_dict_full.get("puntaje_estrategia"))
            if pe_raw is None:
                return None, None
            pe = _norm_01(pe_raw)

        pe = float(pe)
        if not math.isfinite(pe):
            return None, None
        pe = max(0.0, min(pe, 1.0))
        row_dict_full["puntaje_estrategia"] = pe

        # 8) Features requeridas (13 core, estricto)
        required = [
            "rsi_9","rsi_14","sma_5","sma_20","cruce_sma","breakout",
            "rsi_reversion","racha_actual","payout","puntaje_estrategia",
            "volatilidad","es_rebote","hora_bucket",
        ]

        fila_dict = {}
        for k in required:
            fv = _safe_float_local(row_dict_full.get(k))
            if fv is None or not math.isfinite(float(fv)):
                return None, None
            fila_dict[k] = float(fv)

        return fila_dict, int(label)

    except Exception as e:
        print(f"[WARN] leer_ultima_fila_con_resultado({bot}) fallo: {e}")
        return None, None

# ==========================================================
# === BLOQUE 10A — AUDITORÍA DE SEÑALES IA (solo logging; NO toca trading) ===
# Objetivo:
# - Registrar señales IA (bot, epoch, prob, thr, modo) en ia_signals_log.csv
# - Cerrar señales cuando aparezca el CIERRE real (GANANCIA/PÉRDIDA) para ese epoch
# - Calcular métricas simples (Brier/AUC/Acc) con señales cerradas
# ==========================================================

IA_SIGNALS_LOG = "ia_signals_log.csv"

# Blindaje: evita crash si threading aún no estaba importado (aunque tú sí lo tienes)
try:
    import threading as _audit_threading
    IA_SIGNALS_LOCK = _audit_threading.Lock()
except Exception:
    class _DummyLock:
        def __enter__(self): return self
        def __exit__(self, exc_type, exc, tb): return False
    IA_SIGNALS_LOCK = _DummyLock()

def _col_as_str_series(df: pd.DataFrame, col: str) -> pd.Series:
    """Devuelve df[col] como Series(str) y trata NaN como vacío (""). Si no existe, Series vacía del tamaño del df."""
    try:
        if col in df.columns:
            s = df[col]
            try:
                s = s.fillna("")
            except Exception:
                pass
            return s.astype(str)
        return pd.Series([""] * len(df), index=df.index, dtype="object")
    except Exception:
        return pd.Series([""] * len(df), index=df.index, dtype="object")

def _ag_evt(msg: str):
    try:
        fn = globals().get("agregar_evento", None)
        if callable(fn):
            fn(msg)
        else:
            print(msg)
    except Exception:
        pass

def _safe_read_csv_any_encoding(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    for enc in ("utf-8", "latin-1", "windows-1252"):
        try:
            return pd.read_csv(path, sep=",", encoding=enc, engine="python", on_bad_lines="skip")
        except Exception:
            continue
    return None

def _ensure_ia_signals_log():
    """Crea el archivo con header si no existe."""
    if os.path.exists(IA_SIGNALS_LOG):
        return
    try:
        with open(IA_SIGNALS_LOG, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["ts", "bot", "epoch", "prob", "thr", "modo", "y"])
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        pass

def _atomic_write_text(path: str, text: str) -> bool:
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        return True
    except Exception:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        return False

def _to_int_epoch(v) -> int | None:
    try:
        if v is None:
            return None
        if isinstance(v, str):
            v = v.strip()
            if v == "":
                return None
        x = float(v)
        if not math.isfinite(x):
            return None
        # epoch ms -> s
        if x > 1e12:
            x = x / 1000.0
        if x < 1e9:
            return None
        return int(x)
    except Exception:
        return None

def _tail_rows_dict(path: str, max_lines: int = 1200) -> list[dict]:
    """
    Lee SOLO el header + últimas N líneas para no reventar rendimiento con CSV enormes.
    Devuelve lista de dicts (puede ser vacía).
    """
    if not os.path.exists(path):
        return []
    for enc in ("utf-8", "latin-1", "windows-1252"):
        try:
            from collections import deque as _dq
            with open(path, "r", encoding=enc, errors="replace", newline="") as f:
                header = f.readline()
                if not header:
                    return []
                dq = _dq(f, maxlen=int(max_lines))
            lines = [header] + list(dq)
            reader = csv.DictReader(lines)
            out = []
            for r in reader:
                if isinstance(r, dict) and r:
                    out.append(r)
            return out
        except Exception:
            continue
    return []

def ia_audit_get_last_pre_epoch(bot: str) -> int | None:
    """
    Intenta obtener epoch del último PRE_TRADE (incluye PENDING/OPEN/ABIERTO por normalizador).
    Si no existe trade_status, cae a la última fila NO cierre.
    """
    ruta = f"registro_enriquecido_{bot}.csv"
    rows = _tail_rows_dict(ruta, max_lines=1400)
    if not rows:
        return None

    norm_fn = globals().get("normalizar_resultado", None)

    def _norm(x):
        try:
            if callable(norm_fn):
                return norm_fn(x)
        except Exception:
            pass
        s = str(x or "").upper()
        if "GAN" in s or "WIN" in s or "✓" in s:
            return "GANANCIA"
        if "PERD" in s or "LOSS" in s or "✗" in s:
            return "PÉRDIDA"
        return "INDEFINIDO"

    # ¿Existe trade_status o trade_status_norm con contenido?
    def _has_key_nonempty(r, k):
        try:
            if k not in r:
                return False
            v = r.get(k, None)
            if v is None:
                return False
            return str(v).strip() != ""
        except Exception:
            return False

    has_ts = any((_has_key_nonempty(r, "trade_status") or _has_key_nonempty(r, "trade_status_norm")) for r in rows)

    # 1) Con trade_status: preferir PRE_TRADE real
    if has_ts:
        for r in reversed(rows):
            ts = normalizar_trade_status(r.get("trade_status_norm", None) or r.get("trade_status", None))
            if ts != "PRE_TRADE":
                continue

            # NO usar cierres como "pre"
            res = _norm(r.get("resultado", None))
            if res in ("GANANCIA", "PÉRDIDA"):
                continue

            ep = _to_int_epoch(r.get("epoch", None))
            if ep is not None:
                return ep

    # 2) Fallback: última fila que NO sea cierre
    for r in reversed(rows):
        res = _norm(r.get("resultado", None))
        if res not in ("GANANCIA", "PÉRDIDA"):
            ep = _to_int_epoch(r.get("epoch", None))
            if ep is not None:
                return ep

    return None
def ia_audit_get_last_close(bot: str) -> tuple[int | None, int | None]:
    """
    Devuelve (epoch_close, y) donde y=1 GANANCIA, y=0 PÉRDIDA.
    Acepta CERRADO y también CLOSED/SETTLED/etc. vía normalizar_trade_status().
    """
    ruta = f"registro_enriquecido_{bot}.csv"
    rows = _tail_rows_dict(ruta, max_lines=1800)
    if not rows:
        return None, None

    norm_fn = globals().get("normalizar_resultado", None)
    def _norm(x):
        try:
            if callable(norm_fn):
                return norm_fn(x)
        except Exception:
            pass
        s = str(x or "").upper()
        if "GAN" in s or "WIN" in s or "✓" in s:
            return "GANANCIA"
        if "PERD" in s or "LOSS" in s or "✗" in s:
            return "PÉRDIDA"
        return "INDEFINIDO"

    has_ts = any(("trade_status" in r) or ("trade_status_norm" in r) for r in rows)

    for r in reversed(rows):
        res = _norm(r.get("resultado", None))
        if res not in ("GANANCIA", "PÉRDIDA"):
            continue

        if has_ts:
            ts_raw = r.get("trade_status_norm", None) or r.get("trade_status", None)
            ts = normalizar_trade_status(ts_raw)
            if ts and ts != "CERRADO":
                continue

        ep = _to_int_epoch(r.get("epoch", None))
        if ep is None:
            continue

        y = 1 if res == "GANANCIA" else 0
        return ep, y

    return None, None


def log_ia_open(bot: str, epoch: int, prob: float, thr: float, modo: str):
    """
    Registra una señal IA ABIERTA (y="") asociada al epoch PRE_TRADE.
    Blindaje: normaliza columna y para que quede SIEMPRE como "", "0" o "1" (no 0.0/1.0).
    Además, normaliza epoch para evitar duplicados por "123" vs "123.0".
    """
    try:
        _ensure_ia_signals_log()
        with IA_SIGNALS_LOCK:
            df = _safe_read_csv_any_encoding(IA_SIGNALS_LOG)
            if df is None or df.empty:
                df = pd.DataFrame(columns=["ts", "bot", "epoch", "prob", "thr", "modo", "y"])

            # Asegurar columnas base
            for c in ["ts", "bot", "epoch", "prob", "thr", "modo", "y"]:
                if c not in df.columns:
                    df[c] = ""

            def _norm_y(v):
                if v is None:
                    return ""
                s = str(v).strip()
                if s == "" or s.lower() == "nan":
                    return ""
                try:
                    return "1" if float(s) >= 0.5 else "0"
                except Exception:
                    su = s.upper()
                    if ("GAN" in su) or ("WIN" in su) or ("✓" in su):
                        return "1"
                    if ("PERD" in su) or ("PÉRD" in su) or ("LOSS" in su) or ("✗" in su):
                        return "0"
                    return ""

            df["y"] = df["y"].map(_norm_y)

            bot_s = _col_as_str_series(df, "bot").str.strip()
            y_s = _col_as_str_series(df, "y").str.strip()
            epoch_num = pd.to_numeric(_col_as_str_series(df, "epoch"), errors="coerce")

            # Evitar duplicado exacto (misma señal abierta)
            try:
                ep = float(int(epoch))
            except Exception:
                ep = None

            if ep is not None:
                m_same_open = (bot_s == str(bot)) & (epoch_num == ep) & (y_s == "")
                if m_same_open.any():
                    return False

            row = {
                "ts": f"{time.time():.6f}",
                "bot": str(bot),
                "epoch": str(int(epoch)),
                "prob": float(prob),
                "thr": float(thr),
                "modo": str(modo or ""),
                "y": ""
            }

            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            _atomic_write_text(IA_SIGNALS_LOG, df.to_csv(index=False, lineterminator="\n"))
            return True
    except Exception:
        return False


def log_ia_close(
    bot: str,
    epoch: int,
    y: int,
    prob_override: float = None,
    thr_override: float = None,
    modo_override: str = None
):
    """
    Cierra la señal IA más cercana (ABIERTA) para ese bot.
    Blindaje: reconoce cierres aunque el CSV tenga y como 0.0/1.0 o texto, y re-normaliza a "0"/"1".
    """
    try:
        _ensure_ia_signals_log()
        with IA_SIGNALS_LOCK:
            df = _safe_read_csv_any_encoding(IA_SIGNALS_LOG)
            if df is None or df.empty:
                return False

            for c in ["ts", "bot", "epoch", "prob", "thr", "modo", "y"]:
                if c not in df.columns:
                    df[c] = ""

            def _norm_y(v):
                if v is None:
                    return ""
                s = str(v).strip()
                if s == "" or s.lower() == "nan":
                    return ""
                try:
                    return "1" if float(s) >= 0.5 else "0"
                except Exception:
                    su = s.upper()
                    if ("GAN" in su) or ("WIN" in su) or ("✓" in su):
                        return "1"
                    if ("PERD" in su) or ("PÉRD" in su) or ("LOSS" in su) or ("✗" in su):
                        return "0"
                    return ""

            df["y"] = df["y"].map(_norm_y)

            bot_s = _col_as_str_series(df, "bot").str.strip()
            y_s = _col_as_str_series(df, "y").str.strip()
            epoch_num = pd.to_numeric(_col_as_str_series(df, "epoch"), errors="coerce")

            is_open = (y_s == "")
            yv = 1 if int(y) == 1 else 0
            close_n = int(epoch)

            # 1) Match exacto a señal abierta por epoch
            m_exact_open = (bot_s == str(bot)) & is_open & (epoch_num == float(close_n))
            if m_exact_open.any():
                for idx in df.index[m_exact_open]:
                    df.at[idx, "y"] = str(yv)
                    if prob_override is not None:
                        df.at[idx, "prob"] = float(prob_override)
                    if thr_override is not None:
                        df.at[idx, "thr"] = float(thr_override)
                    if modo_override is not None:
                        df.at[idx, "modo"] = str(modo_override)

                _atomic_write_text(IA_SIGNALS_LOG, df.to_csv(index=False, lineterminator="\n"))
                return True

            # 2) Si no hay exact match, cerrar la ABIERTA más reciente con epoch <= epoch_close
            cand = df[(bot_s == str(bot)) & is_open].copy()
            if cand.empty:
                return False

            cand_epoch = pd.to_numeric(cand["epoch"], errors="coerce")
            cand = cand[cand_epoch.notna()].copy()
            cand["epoch_num"] = cand_epoch[cand_epoch.notna()].astype(float)

            cand = cand[cand["epoch_num"] <= float(close_n)]
            if cand.empty:
                return False

            pick_idx = cand["epoch_num"].idxmax()
            df.at[pick_idx, "y"] = str(yv)
            if prob_override is not None:
                df.at[pick_idx, "prob"] = float(prob_override)
            if thr_override is not None:
                df.at[pick_idx, "thr"] = float(thr_override)
            if modo_override is not None:
                df.at[pick_idx, "modo"] = str(modo_override)

            _atomic_write_text(IA_SIGNALS_LOG, df.to_csv(index=False, lineterminator="\n"))
            return True
    except Exception:
        return False

IA_AUDIT_LAST_CLOSE_EPOCH = {b: None for b in BOT_NAMES}

def ia_audit_scan_close(bot: str, tail_lines: int = 2000, max_events: int = 6):
    """
    Detecta CIERRES reales nuevos en registro_enriquecido_{bot}.csv y cierra señales en ia_signals_log.csv.
    - Procesa hasta `max_events` cierres por tick para no sobrecargar.
    - Procesa en ORDEN ASCENDENTE para NO saltarse cierres antiguos (catch-up real).
    - Usa IA_AUDIT_LAST_CLOSE_EPOCH para avanzar incrementalmente.
    """
    try:
        last = IA_AUDIT_LAST_CLOSE_EPOCH.get(bot)
    except Exception:
        last = None

    ruta = f"registro_enriquecido_{bot}.csv"
    rows = _tail_rows_dict(ruta, max_lines=int(tail_lines))
    if not rows:
        return

    # ¿Existe trade_status (raw o norm) en alguna fila?
    try:
        has_ts = any(("trade_status" in (r or {})) or ("trade_status_norm" in (r or {})) for r in rows)
    except Exception:
        has_ts = False

    # Recolectar cierres > last
    cierres = []
    seen = set()

    for r in rows:
        try:
            rr = (r or {})
            ep = None
            for k in ("epoch", "fecha", "timestamp", "ts"):
                ep = _to_int_epoch(rr.get(k))
                if ep is not None:
                    break
        except Exception:
            ep = None

        if ep is None:
            continue


        if last is not None:
            try:
                if int(ep) <= int(last):
                    continue
            except Exception:
                continue

        if has_ts:
            try:
                ts_raw = (r or {}).get("trade_status_norm") or (r or {}).get("trade_status")
                tsn = normalizar_trade_status(ts_raw)
                if tsn and tsn != "CERRADO":
                    continue
            except Exception:
                pass

        try:
            resn = normalizar_resultado((r or {}).get("resultado"))
        except Exception:
            resn = ""
        if resn not in ("GANANCIA", "PÉRDIDA"):
            continue

        y = 1 if resn == "GANANCIA" else 0

        # Dedup por epoch en este tick (nos quedamos con el último y del epoch)
        if int(ep) in seen:
            for i in range(len(cierres) - 1, -1, -1):
                if cierres[i][0] == int(ep):
                    cierres[i] = (int(ep), int(y))
                    break
        else:
            cierres.append((int(ep), int(y)))
            seen.add(int(ep))

    if not cierres:
        return

    cierres.sort(key=lambda t: t[0])  # ASC

    # Limitar: tomamos LOS PRIMEROS para catch-up real (no saltar viejos)
    if max_events and len(cierres) > int(max_events):
        cierres = cierres[:int(max_events)]

    for ep, y in cierres:
        try:
            _ = bool(log_ia_close(bot, ep, y))
        except Exception:
            pass
        # Avanzamos el puntero siempre: si no había señal abierta (trade sin señal IA), no queremos trabarnos.
        IA_AUDIT_LAST_CLOSE_EPOCH[bot] = int(ep)

def semaforo_calibracion(n: int, infl_pp: float | None):
    """Devuelve (emoji, etiqueta, detalle) para lectura rápida de calibración."""
    try:
        n = int(n or 0)
    except Exception:
        n = 0

    try:
        infl = abs(float(infl_pp)) if infl_pp is not None else None
    except Exception:
        infl = None

    if n < SEM_CAL_N_ROJO:
        return "🔴", "CRÍTICO", f"n={n}<{SEM_CAL_N_ROJO}"

    if infl is None:
        if n < SEM_CAL_N_AMARILLO:
            return "🟡", "PRECAUCIÓN", f"n={n}<{SEM_CAL_N_AMARILLO}"
        return "🟢", "CONFIABLE", f"n={n} (sin inflación calculable)"

    if infl > SEM_CAL_INFL_WARN_PP:
        return "🔴", "CRÍTICO", f"|infl|={infl:.1f}pp>{SEM_CAL_INFL_WARN_PP:.0f}pp"

    if (n < SEM_CAL_N_AMARILLO) or (infl > SEM_CAL_INFL_OK_PP):
        return "🟡", "PRECAUCIÓN", f"n={n}, |infl|={infl:.1f}pp"

    return "🟢", "CONFIABLE", f"n={n}, |infl|={infl:.1f}pp"

def diagnostico_calibracion(n: int, pred_mean: float, win_rate: float, infl_pp: float | None):
    """Mensaje corto para saber si la calibración va por buen camino."""
    try:
        n = int(n or 0)
    except Exception:
        n = 0

    try:
        infl_abs = abs(float(infl_pp)) if infl_pp is not None else None
    except Exception:
        infl_abs = None

    if n < SEM_CAL_N_ROJO:
        return "Todavía no se puede concluir (muestra muy chica): sigue juntando cierres reales."

    if infl_abs is None:
        return "Hay cierres, pero aún no alcanza para medir la brecha Pred vs Real con confianza."

    if infl_abs <= SEM_CAL_INFL_OK_PP:
        return "Vas por buen camino: la probabilidad predicha está cerca del resultado real."

    if infl_abs <= SEM_CAL_INFL_WARN_PP:
        sesgo = "sobreestima" if pred_mean >= win_rate else "subestima"
        return f"Hay avance, pero la IA aún {sesgo} el resultado real; conviene más muestra."

    sesgo = "sobreestimando" if pred_mean >= win_rate else "subestimando"
    return f"Se detecta descalibración fuerte ({sesgo}); no usar la probabilidad sola para decidir."

def auditar_calibracion_seniales_reales(min_prob: float = 0.70, max_rows: int = 20000, n_bins: int = 10):
    """
    Auditoría REAL vs PRED (señales cerradas en ia_signals_log.csv).

    Devuelve:
      - n
      - win_rate (real)
      - avg_pred (promedio de prob del modelo)
      - inflacion_pp = (avg_pred - win_rate)*100
      - factor = win_rate/avg_pred (clamp) si hay data suficiente
      - brier = mean((prob - y)^2)
      - ece = Expected Calibration Error (bins 0..1)
      - por_bot: métricas por bot
    """
    try:
        _ensure_ia_signals_log()
        df = _safe_read_csv_any_encoding(IA_SIGNALS_LOG)
        if df is None or df.empty:
            return None

        if ("prob" not in df.columns) or ("y" not in df.columns) or ("bot" not in df.columns):
            return None

        d = df.copy()

        # y debe ser 0/1 para considerar "cerrada"
        # (soporta 0/1, 0.0/1.0, "0"/"1", y texto tipo GANANCIA/PÉRDIDA/✓/✗)
        y_raw = d["y"]

        y_num = pd.to_numeric(y_raw, errors="coerce")
        y_txt = y_raw.astype(str).str.strip().str.upper()

        # Fallback por texto si no pudo convertirse a número
        y_num = y_num.where(
            ~y_num.isna(),
            np.where(
                y_txt.str.contains(r"GAN|WIN|✓"),
                1.0,
                np.where(y_txt.str.contains(r"PERD|PÉRD|LOSS|✗"), 0.0, np.nan)
            )
        )

        # Normalizar a 0/1 (todo lo >=0.5 se considera 1)
        d["y"] = np.where(pd.isna(y_num), np.nan, np.where(y_num >= 0.5, 1, 0))
        d = d[d["y"].isin([0, 1])].copy()
        if d.empty:
            return None

        # prob numérica (defensivo)
        d["prob"] = pd.to_numeric(d["prob"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        d["y"] = d["y"].astype(int)

        n_total_closed = int(len(d))

        # filtro por umbral
        d = d[d["prob"] >= float(min_prob)].copy()
        if d.empty:
            return {
                "n": 0,
                "n_total_closed": n_total_closed,
                "n_after_threshold": 0,
                "min_prob": float(min_prob),
                "win_rate": None,
                "avg_pred": None,
                "inflacion_pp": None,
                "factor": 1.0,
                "brier": None,
                "ece": None,
                "por_bot": {},
            }

        # limitar tamaño para no castigar IO/cpu
        try:
            if max_rows and len(d) > int(max_rows):
                d = d.iloc[-int(max_rows):].copy()
        except Exception:
            pass

        y = d["y"].to_numpy(dtype=float)
        p = d["prob"].to_numpy(dtype=float)

        def _ece(_y, _p, bins: int = 10) -> float:
            _y = np.asarray(_y, dtype=float)
            _p = np.asarray(_p, dtype=float)
            if _p.size == 0:
                return 0.0
            edges = np.linspace(0.0, 1.0, int(bins) + 1)
            ece = 0.0
            n = float(_p.size)
            for i in range(len(edges) - 1):
                lo, hi = edges[i], edges[i + 1]
                if i < len(edges) - 2:
                    m = (_p >= lo) & (_p < hi)
                else:
                    m = (_p >= lo) & (_p <= hi)
                cnt = int(m.sum())
                if cnt <= 0:
                    continue
                avg_p = float(_p[m].mean())
                avg_y = float(_y[m].mean())
                ece += abs(avg_p - avg_y) * (cnt / n)
            return float(ece)

        win_rate = float(np.mean(y))
        avg_pred = float(np.mean(p))
        infl_pp = (avg_pred - win_rate) * 100.0

        factor = 1.0
        if avg_pred > 1e-6 and len(d) >= 30:
            factor = win_rate / avg_pred
            factor = max(0.60, min(1.30, factor))  # clamp defensivo

        brier = float(np.mean((p - y) ** 2))
        ece = _ece(y, p, bins=n_bins)

        por_bot = {}
        try:
            for b, g in d.groupby("bot"):
                n = int(len(g))
                yb = g["y"].to_numpy(dtype=float)
                pb = g["prob"].to_numpy(dtype=float)

                wr = float(np.mean(yb)) if n else None
                ap = float(np.mean(pb)) if n else None
                inf = ((ap - wr) * 100.0) if (wr is not None and ap is not None) else None

                fb = 1.0
                if ap and ap > 1e-6 and n >= 20:
                    fb = wr / ap
                    fb = max(0.60, min(1.30, fb))

                por_bot[str(b)] = {
                    "n": n,
                    "win_rate": wr,
                    "avg_pred": ap,
                    "inflacion_pp": inf,
                    "factor": fb,
                    "brier": float(np.mean((pb - yb) ** 2)) if n else None,
                    "ece": _ece(yb, pb, bins=n_bins) if n else None,
                }
        except Exception:
            por_bot = {}

        stable_sample = bool(len(d) >= int(IA_CALIB_MIN_CLOSED))

        return {
            "n": int(len(d)),
            "n_total_closed": n_total_closed,
            "n_after_threshold": int(len(d)),
            "min_prob": float(min_prob),
            "win_rate": win_rate,
            "avg_pred": avg_pred,
            "inflacion_pp": infl_pp,
            "factor": factor,
            "brier": brier,
            "ece": ece,
            "stable_sample": stable_sample,
            "min_recommended_n": int(IA_CALIB_MIN_CLOSED),
            "por_bot": por_bot,
        }
    except Exception:
        return None

# === FIN BLOQUE 10A ===
# ==========================================================
        
# ==========================================================
# ✅ HOTFIX IA: Prob IA REAL (no forzar a 0 si no hay señal)
# - Predice sobre la última fila PRE_TRADE/PENDIENTE si existe
# - Si no existe, cae a la última fila “no cierre” antes del último cierre
# - Alinea features con features.pkl (si existe) o FEATURE_NAMES_DEFAULT
# - Si falla, deja ia_ready=False y ia_last_err con el motivo
# ==========================================================

_IA_ASSETS_CACHE = {"loaded": False, "model": None, "scaler": None, "features": None, "meta": None}

def _find_first_pickle(regex_list, exts=(".pkl", ".joblib")):
    try:
        files = os.listdir(".")
        for fn in files:
            low = fn.lower()
            if not low.endswith(exts):
                continue
            for rx in regex_list:
                try:
                    if re.search(rx, low):
                        return fn
                except Exception:
                    continue
    except Exception:
        pass
    return None

def _load_ia_assets_once(force: bool = False):
    """
    Carga assets desde globals si existen; si no, autodetecta en disco.
    IMPORTANTE: si al arrancar no había modelo/scaler y luego aparecen (por entrenamiento),
    con force=True se recarga y no se queda “pegado”.
    """
    # Si ya se cargó y no pedimos fuerza, salimos
    if _IA_ASSETS_CACHE.get("loaded", False) and (not force):
        return

    g = globals()

    # 1) Preferir objetos en memoria (después de entrenar)
    model  = g.get("modelo_ia") or g.get("IA_MODELO") or g.get("modelo_oracle") or g.get("oracle_model") or None
    scaler = g.get("scaler_ia") or g.get("IA_SCALER") or g.get("oracle_scaler") or None
    feats  = g.get("feature_names_ia") or g.get("FEATURE_NAMES_USADAS") or g.get("FEATURE_NAMES_MODEL") or None
    meta   = g.get("meta_ia") or g.get("IA_META") or g.get("oracle_meta") or None

    # 2) Fallback: disco (tus “4 artefactos”)
    if model is None:
        mfile = _find_first_pickle([r"modelo", r"model", r"xgb"])
        if mfile:
            try:
                model = joblib.load(mfile)
            except Exception:
                model = None

    if scaler is None:
        sfile = _find_first_pickle([r"scaler"])
        if sfile:
            try:
                scaler = joblib.load(sfile)
            except Exception:
                scaler = None

    if feats is None:
        ffile = _find_first_pickle([r"features", r"feature_names"])
        if ffile:
            try:
                feats = joblib.load(ffile)
            except Exception:
                feats = None

    if meta is None:
        metafile = _find_first_pickle([r"meta"])
        if metafile:
            try:
                meta = joblib.load(metafile)
            except Exception:
                meta = None

    _IA_ASSETS_CACHE.update({
        "loaded": True,
        "model": model,
        "scaler": scaler,
        "features": feats,
        "meta": meta
    })

def _features_model_list():
    _load_ia_assets_once()
    feats = _IA_ASSETS_CACHE.get("features")
    # features.pkl puede venir como list o dict
    if isinstance(feats, list) and feats:
        return list(feats)
    if isinstance(feats, dict) and feats.get("features"):
        try:
            return list(feats["features"])
        except Exception:
            pass
    # fallback canónico
    return list(FEATURE_NAMES_DEFAULT)

def _add_derived_for_model(d: dict):
    """Si el modelo espera features derivadas, créalas aquí."""
    try:
        racha = float(d.get("racha_actual", 0.0) or 0.0)
    except Exception:
        racha = 0.0

    d["racha_signo"] = 1.0 if racha > 0 else (-1.0 if racha < 0 else 0.0)
    d["racha_abs"] = abs(racha)
    d["rebote_fuerte"] = 1.0 if abs(racha) >= 6 else 0.0

    # Interacciones (si no existen, se arman igual)
    try:
        payout = float(d.get("payout", 0.0) or 0.0)
    except Exception:
        payout = 0.0
    try:
        pe = float(d.get("puntaje_estrategia", 0.0) or 0.0)
    except Exception:
        pe = 0.0
    try:
        vol = float(d.get("volatilidad", 0.0) or 0.0)
    except Exception:
        vol = 0.0
    try:
        brk = float(d.get("breakout", 0.0) or 0.0)
    except Exception:
        brk = 0.0
    try:
        hb = float(d.get("hora_bucket", 0.5) or 0.5)
    except Exception:
        hb = 0.5
    try:
        er = float(d.get("es_rebote", 0.0) or 0.0)
    except Exception:
        er = 0.0

    d["pay_x_puntaje"] = payout * pe
    d["vol_x_breakout"] = vol * brk
    d["hora_x_rebote"] = hb * er

    return d

def leer_ultima_fila_features_para_pred(bot: str) -> dict | None:
    """
    Lee features para PREDICCIÓN (sin label):
    - Prefiere trade_status PRE_TRADE/PENDIENTE/OPEN/ABIERTO
    - Fallback: última fila “no cierre” antes del último cierre
    - Anti-leakage: elimina payout_total/ganancia_perdida/resultado del dict final
    """
    ruta = f"registro_enriquecido_{bot}.csv"
    if not os.path.exists(ruta):
        return None

    df = None
    for enc in ("utf-8", "latin-1", "windows-1252"):
        try:
            df = pd.read_csv(ruta, sep=",", encoding=enc, engine="python", on_bad_lines="skip")
            break
        except Exception:
            continue
    if df is None or df.empty:
        return None

    # Normalizar resultado si existe
    if "resultado" in df.columns:
        df["resultado_norm"] = df["resultado"].apply(normalizar_resultado)
    else:
        df["resultado_norm"] = "INDEFINIDO"

    # Normalizar trade_status de forma canónica (evita CLOSED/SETTLED vs CERRADO)
    has_trade_status = (("trade_status" in df.columns) or ("trade_status_norm" in df.columns))

    if has_trade_status:
        if "trade_status_norm" in df.columns:
            df["trade_status_norm"] = df["trade_status_norm"].apply(normalizar_trade_status)
        else:
            df["trade_status_norm"] = df["trade_status"].apply(normalizar_trade_status)
    else:
        df["trade_status_norm"] = ""

    # 1) preferir PRE_TRADE/PENDIENTE
    pre = None
    if has_trade_status:
        df_pre = df[df["trade_status_norm"].isin(["PENDIENTE", "PRE_TRADE", "OPEN", "ABIERTO"])].copy()
        # evitar “filas basura”: no usar cierres
        if "resultado_norm" in df_pre.columns:
            df_pre = df_pre[~df_pre["resultado_norm"].isin(["GANANCIA", "PÉRDIDA"])].copy()
        if not df_pre.empty:
            pre = df_pre.iloc[-1].to_dict()

    # 2) fallback: tomar última fila antes del último cierre que no sea cierre
    if pre is None:
        df_cerr = df[df["resultado_norm"].isin(["GANANCIA", "PÉRDIDA"])].copy()
        if has_trade_status:
            df_cerr = df_cerr[df_cerr["trade_status_norm"].eq("CERRADO")].copy()
        if not df_cerr.empty:
            idx_close = df_cerr.index[-1]
            try:
                df_before = df.loc[df.index <= idx_close].copy()
            except Exception:
                df_before = df.copy()
        else:
            df_before = df.copy()

        cand = df_before[~df_before["resultado_norm"].isin(["GANANCIA", "PÉRDIDA"])].copy()
        if not cand.empty:
            pre = cand.iloc[-1].to_dict()
        else:
            pre = df_before.iloc[-1].to_dict()

    if pre is None:
        return None

    row = dict(pre)

    # 🔒 Anti-leakage duro
    for k in ("payout_total", "ganancia_perdida", "profit", "resultado", "resultado_norm"):
        try:
            row.pop(k, None)
        except Exception:
            pass

    # Completar y normalizar base 13 features como haces en leer_ultima_fila_con_resultado
    # payout feature (ROI 0..1.5)
    try:
        roi = calcular_payout_feature(row)
    except Exception:
        roi = None
    if roi is None:
        return None
    row["payout"] = float(max(0.0, min(float(roi), 1.5)))

    # volatilidad / hora_bucket / es_rebote / puntaje_estrategia
    try:
        vol = _safe_float(row.get("volatilidad"))
        if vol is None:
            vol = calcular_volatilidad_simple(row)
        row["volatilidad"] = float(max(0.0, min(float(vol), 1.0)))
    except Exception:
        return None

    try:
        hb = _safe_float(row.get("hora_bucket"))
        if hb is None:
            hb = calcular_hora_bucket(row)
        row["hora_bucket"] = float(max(0.0, min(float(hb), 1.0)))
    except Exception:
        return None

    try:
        er = _safe_float(row.get("es_rebote"))
        if er is None:
            er = calcular_es_rebote(row)
        row["es_rebote"] = float(max(0.0, min(float(er), 1.0)))
    except Exception:
        return None

    try:
        pe = calcular_puntaje_estrategia_normalizado(row)
        row["puntaje_estrategia"] = float(max(0.0, min(float(pe), 1.0)))
    except Exception:
        # fallback a lo que exista
        pe_raw = _safe_float(row.get("puntaje_estrategia"))
        if pe_raw is None:
            return None
        row["puntaje_estrategia"] = float(max(0.0, min(float(pe_raw), 1.0)))

    # Derived extras (por si el modelo los espera)
    row = _add_derived_for_model(row)

    # Clip defensivo si existe tu helper
    try:
        # usa el set esperado por el modelo (solo recorta lo que exista)
        fnames = _features_model_list()
        row = clip_feature_values(row, fnames)
    except Exception:
        pass

    return row

def _coerce_float_default(v, default=0.0) -> float:
    try:
        if v is None:
            return float(default)
        if isinstance(v, str):
            v = v.strip()
            if v == "":
                return float(default)
        x = float(v)
        if not np.isfinite(x):
            return float(default)
        return float(x)
    except Exception:
        return float(default)

def predecir_prob_ia_bot(bot: str) -> tuple[float | None, str | None]:
    """
    Retorna (prob, err). prob en 0..1.
    NO devuelve 0 "por defecto": si falla, devuelve (None, CODIGO_ERROR).
    """
    try:
        # 1) Cargar assets (y recargar si aparecieron luego del boot)
        _load_ia_assets_once()
        model = _IA_ASSETS_CACHE.get("model")
        scaler = _IA_ASSETS_CACHE.get("scaler")

        if model is None:
            _load_ia_assets_once(force=True)
            model = _IA_ASSETS_CACHE.get("model")
            scaler = _IA_ASSETS_CACHE.get("scaler")

        if model is None:
            return None, "NO_MODELO"

        # 2) Leer fila de features para pred (sin label)
        row = leer_ultima_fila_features_para_pred(bot)
        if row is None:
            return None, "NO_FEATURE_ROW"

        # 3) Lista de features esperadas por el modelo (features.pkl si existe)
        feats = _features_model_list()
        if not feats:
            return None, "NO_FEATS"

        # 4) Armar X con orden exacto + rellenar faltantes con 0.0 + NaN→0.0
        values = []
        for k in feats:
            v = row.get(k, None)
            # si clip_feature_values metió NaN, lo volvemos 0.0 para no romper scaler
            if v is not None:
                try:
                    if isinstance(v, float) and np.isnan(v):
                        v = 0.0
                except Exception:
                    pass
            values.append(_coerce_float_default(v, default=0.0))

        X = pd.DataFrame([values], columns=list(feats))

        # 5) Escalado (si existe)
        X_in = X
        if scaler is not None:
            try:
                # Si el scaler fue fit con nombres, intentamos alinear
                if hasattr(scaler, "feature_names_in_") and scaler.feature_names_in_ is not None:
                    need = list(scaler.feature_names_in_)
                    # reindexa (si falta algo -> 0.0)
                    X_in = X.reindex(columns=need, fill_value=0.0)
                else:
                    X_in = X

                # StandardScaler no acepta NaN: blindaje final
                X_in = X_in.replace([np.inf, -np.inf], 0.0).fillna(0.0)

                X_scaled = scaler.transform(X_in)
            except Exception as e:
                return None, f"SCALER_FAIL:{type(e).__name__}"

        else:
            # sin scaler, usamos valores crudos
            X_scaled = X_in.replace([np.inf, -np.inf], 0.0).fillna(0.0).values

        # 6) Predict proba
        try:
            proba = model.predict_proba(X_scaled)
            p = float(proba[0][1])
        except Exception as e:
            return None, f"PRED_FAIL:{type(e).__name__}"

        if not np.isfinite(p):
            return None, "PROB_NAN"

        # clamp
        p = max(0.0, min(1.0, p))
        return p, None

    except Exception as e:
        return None, f"IA_ERR:{type(e).__name__}"

# --- Updater: NO fuerces prob_ia=0 cuando falla ---
IA_PRED_TTL_S = 180.0          # si falla por mucho tiempo, recién se limpia a None
IA_PRED_MIN_INTERVAL_S = 2.0   # anti-spam de predicción
_last_pred_ts = {b: 0.0 for b in BOT_NAMES}

def actualizar_prob_ia_bot(bot: str):
    """
    Actualiza estado_bots[bot]['prob_ia'] de forma segura:
    - Si hay prob válida: la escribe, define modo_ia y marca ia_ready=True.
    - Si falla: NO pisa prob_ia a 0. Conserva último valor por TTL para no vaciar el HUD.
    """
    try:
        now = time.time()
        last = float(_last_pred_ts.get(bot, 0.0) or 0.0)
        if (now - last) < IA_PRED_MIN_INTERVAL_S:
            return
        _last_pred_ts[bot] = now

        p, err = predecir_prob_ia_bot(bot)

        if p is not None:
            estado_bots[bot]["prob_ia"] = float(p)
            estado_bots[bot]["ia_ready"] = True
            estado_bots[bot]["ia_last_err"] = None
            estado_bots[bot]["ia_last_prob_ts"] = now

            # FIX UI/AUTO: garantizar modo_ia distinto de OFF cuando hay predicción.
            try:
                meta_local = _ORACLE_CACHE.get("meta") or leer_model_meta() or {}
                reliable = bool(meta_local.get("reliable", False))
                n_samples = int(meta_local.get("n_samples", meta_local.get("n", 0)) or 0)
                if reliable:
                    modo = "confiable"
                elif n_samples >= int(MIN_FIT_ROWS_LOW):
                    modo = "modelo"
                else:
                    modo = "low_data"
                estado_bots[bot]["modo_ia"] = modo
            except Exception:
                estado_bots[bot]["modo_ia"] = "modelo"
            return

        # fallo: no mates la última prob, solo marca error
        estado_bots[bot]["ia_last_err"] = err or "ERR"

        # si hace demasiado que no hay prob válida, limpia a None
        last_ok = float(estado_bots[bot].get("ia_last_prob_ts", 0.0) or 0.0)
        age = (now - last_ok) if last_ok > 0 else 10**9

        if age <= IA_PRED_TTL_S and estado_bots[bot].get("prob_ia") is not None:
            # Mantener último dato útil para que la UI no quede en '--'.
            estado_bots[bot]["ia_ready"] = True
            if str(estado_bots[bot].get("modo_ia", "")).strip().lower() in ("", "off"):
                estado_bots[bot]["modo_ia"] = "stale"
        else:
            estado_bots[bot]["ia_ready"] = False
            estado_bots[bot]["prob_ia"] = None
            estado_bots[bot]["modo_ia"] = "off"

    except Exception:
        # ultra defensivo: no romper loop
        try:
            estado_bots[bot]["ia_ready"] = False
            estado_bots[bot]["ia_last_err"] = "UPD_ERR"
        except Exception:
            pass

def actualizar_prob_ia_todos():
    """
    Tick único para el panel:
      1) Cierra señales en IA_SIGNALS_LOG (Real vs Ficción) usando los cierres del CSV del bot.
      2) Actualiza Prob IA por bot (sin tocar la lógica de trading).
    Nota: esto arregla el caso clásico "Aún no hay cierres suficientes" cuando el cierre nunca se ejecuta.
    """
    for b in BOT_NAMES:
        # 1) Backfill / cierre de señales (más profundo solo en el primer tick tras arrancar)
        try:
            last = IA_AUDIT_LAST_CLOSE_EPOCH.get(b, None)
            tail_lines = 25000 if last is None else 6000
            max_events = 60 if last is None else 15
            ia_audit_scan_close(b, tail_lines=tail_lines, max_events=max_events)
        except Exception:
            pass

        # 2) Predicción / estado IA del bot
        try:
            actualizar_prob_ia_bot(b)
        except Exception:
            pass
def actualizar_prob_ia_bots_tick():
    """
    Actualiza estado_bots[*].prob_ia con la prob REAL (aunque sea < 0.70).
    La regla ≥0.70 se usa SOLO para “señal”, no para pintar prob.
    """
    now = time.time()

    for bot in BOT_NAMES:
        # 10A: cierre automático de auditoría si hubo trade cerrado nuevo
        try:
            ia_audit_scan_close(bot)
        except Exception:
            pass
        prob, err = predecir_prob_ia_bot(bot)

        if prob is None:
            estado_bots[bot]["ia_ready"] = False
            estado_bots[bot]["ia_last_err"] = err
            # NO fuerces a 0 por estética: deja el último prob si había, o queda 0 si nunca hubo
            # (pero el HUD lo mostrará como -- con el parche #3)
            # 🔒 evita “señal fantasma” si antes estaba en True
            estado_bots[bot]["ia_senal_pendiente"] = False
            estado_bots[bot]["ia_prob_senal"] = None
        else:
            estado_bots[bot]["ia_ready"] = True
            estado_bots[bot]["ia_last_err"] = None

            # Modo IA coherente con meta
            meta_local = _ORACLE_CACHE.get("meta") or leer_model_meta()
            estado_bots[bot]["modo_ia"] = "CONFIABLE" if bool((meta_local or {}).get("reliable", False)) else "MODELO"

            # Guardar raw (modelo calibrado) y métricas SOLO como diagnóstico (NO ajustar prob_ia)
            prob_raw = float(prob)
            estado_bots[bot]["prob_ia_raw"] = prob_raw

            # Auditoría/calibración: calcular UNA sola vez por tick (evita leer CSV 6 veces)
            if "_stats_cal_global" not in locals():
                _stats_cal_global = auditar_calibracion_seniales_reales(min_prob=0.70) or {}

            por_bot = _stats_cal_global.get("por_bot", {}) if isinstance(_stats_cal_global, dict) else {}
            stats_cal = por_bot.get(bot, {}) if isinstance(por_bot, dict) else {}

            estado_bots[bot]["cal_n"] = int(stats_cal.get("n", 0) or 0)
            estado_bots[bot]["cal_win_rate"] = stats_cal.get("win_rate", None)
            estado_bots[bot]["cal_avg_pred"] = stats_cal.get("avg_pred", None)
            estado_bots[bot]["cal_infl_pp"] = stats_cal.get("inflacion_pp", None)

            # Guardamos el factor sugerido SOLO para diagnóstico (no se aplica a prob)
            estado_bots[bot]["cal_factor_sugerido"] = float(stats_cal.get("factor", 1.0) or 1.0)
            estado_bots[bot]["cal_brier"] = stats_cal.get("brier", None)
            estado_bots[bot]["cal_ece"] = stats_cal.get("ece", None)


            # FIX CLAVE: prob_ia ES la prob real del modelo (calibrada), sin multiplicadores
            estado_bots[bot]["cal_factor"] = 1.0
            estado_bots[bot]["prob_ia"] = prob_raw
            estado_bots[bot]["ia_last_prob_ts"] = now

        # Señal SOLO si supera el umbral verde
        # IMPORTANTE: aquí SOLO marcamos "pendiente" (HUD / elegibilidad).
        # El LOG de auditoría (ia_signals_log) se escribe cuando HAY orden/operación real.
        try:
            if prob is not None and float(prob) >= float(IA_VERDE_THR):
                estado_bots[bot]["ia_senal_pendiente"] = True
                estado_bots[bot]["ia_prob_senal"] = float(prob)
            else:
                estado_bots[bot]["ia_senal_pendiente"] = False
                estado_bots[bot]["ia_prob_senal"] = None
        except Exception:
            estado_bots[bot]["ia_senal_pendiente"] = False
            estado_bots[bot]["ia_prob_senal"] = None

def ia_prob_valida(bot: str, max_age_s: float = 10.0) -> bool:
    """
    True si:
    - ia_ready=True
    - prob_ia existe y es finita en [0..1]
    - timestamp reciente (<= max_age_s)
    """
    try:
        if bot not in estado_bots:
            return False

        if not bool(estado_bots[bot].get("ia_ready", False)):
            return False

        ts = float(estado_bots[bot].get("ia_last_prob_ts", 0.0) or 0.0)
        if ts <= 0:
            return False

        if (time.time() - ts) > float(max_age_s):
            return False

        p = estado_bots[bot].get("prob_ia", None)
        if p is None:
            return False

        p = float(p)
        if not np.isfinite(p):
            return False

        return (0.0 <= p <= 1.0)

    except Exception:
        return False
                                              
def detectar_cierre_martingala(bot, min_fila=None, require_closed=True, require_real_token=False, expected_ciclo=None):
    """
    Devuelve: (resultado_norm, monto, ciclo, payout_total)
    - min_fila: solo acepta filas con número > min_fila (evita cierres viejos)
              Nota: min_fila se interpreta como "cantidad de filas de datos" (sin header),
              y cuadra con contar_filas_csv().
    - require_closed: si existe trade_status, exige CERRADO/CLOSED.
    - require_real_token: si hay columna de token/cuenta, ignora cierres DEMO.
    - expected_ciclo: si existe columna de ciclo, exige coincidencia con ese ciclo.
    """
    path = f"registro_enriquecido_{bot}.csv"
    if not os.path.exists(path):
        return None

    rows = None
    header = None

    for enc in ("utf-8", "latin-1", "windows-1252"):
        try:
            with open(path, "r", encoding=enc, errors="replace", newline="") as f:
                rows = list(csv.reader(f))
            if rows and len(rows) >= 2:
                header = rows[0]
                break
        except Exception:
            continue

    if not rows or not header or len(rows) < 2:
        return None

    # Mapa de columnas case-insensitive y con strip
    hmap = {}
    for i, h in enumerate(header):
        try:
            key = str(h).strip().lower()
            if key and key not in hmap:
                hmap[key] = i
        except Exception:
            pass

    def _col(*names):
        for n in names:
            k = str(n).strip().lower()
            if k in hmap:
                return hmap[k]
        return None

    i_res = _col("resultado", "result", "outcome")
    i_status = _col("trade_status", "status")
    i_monto = _col("monto", "stake", "buy_price", "amount")
    i_ciclo = _col("ciclo", "ciclo_martingala", "ciclo_actual", "marti_ciclo", "martingale_step")
    i_token = _col("token", "account", "account_type", "cuenta", "modo", "mode")
    # payout_total puede venir explícito o calculable
    # (extraer_payout_total ya se encarga, pero igual ayudamos con nombres)
    i_payout_total = _col("payout_total")
    i_payout_mult = _col("payout_multiplier")
    i_payout_dec = _col("payout_decimal_rounded")
    i_payout_legacy = _col("payout")  # legacy (ojo: a veces ROI feature, a veces ratio/total)

    if i_res is None:
        return None

    # Recorremos desde el final (último evento primero)
    for ridx in range(len(rows) - 1, 0, -1):
        row = rows[ridx]
        if not row:
            continue

        # ridx equivale a "número de fila de datos" (header está en 0)
        fila_num = ridx  # 1..N
        if min_fila is not None:
            try:
                if int(fila_num) <= int(min_fila):
                    break  # todo lo que sigue es más viejo todavía
            except Exception:
                pass

        # trade_status si aplica
        if require_closed and (i_status is not None) and (i_status < len(row)):
            st = str(row[i_status]).strip().upper()
            if st not in ("CERRADO", "CLOSED"):
                continue

        # Si el CSV informa token/cuenta, en REAL ignoramos cierres explícitos de DEMO.
        if require_real_token and (i_token is not None) and (i_token < len(row)):
            tok_raw = str(row[i_token] or "").strip().upper()
            if tok_raw:
                # Heurística robusta: DEMO en Deriv suele venir como VRTC*
                es_demo = ("DEMO" in tok_raw) or tok_raw.startswith("VRTC")
                es_real = ("REAL" in tok_raw) or tok_raw.startswith("CR")
                if es_demo and not es_real:
                    continue

        # Si el CSV informa token/cuenta, en REAL ignoramos cierres explícitos de DEMO.
        if require_real_token and (i_token is not None) and (i_token < len(row)):
            tok_raw = str(row[i_token] or "").strip().upper()
            if tok_raw:
                # Heurística robusta: DEMO en Deriv suele venir como VRTC*
                es_demo = ("DEMO" in tok_raw) or tok_raw.startswith("VRTC")
                es_real = ("REAL" in tok_raw) or tok_raw.startswith("CR")
                if es_demo and not es_real:
                    continue

        # Si el CSV informa token/cuenta, en REAL ignoramos cierres explícitos de DEMO.
        if require_real_token and (i_token is not None) and (i_token < len(row)):
            tok_raw = str(row[i_token] or "").strip().upper()
            if tok_raw:
                # Heurística robusta: DEMO en Deriv suele venir como VRTC*
                es_demo = ("DEMO" in tok_raw) or tok_raw.startswith("VRTC")
                es_real = ("REAL" in tok_raw) or tok_raw.startswith("CR")
                if es_demo and not es_real:
                    continue

        # Si el CSV informa token/cuenta, en REAL ignoramos cierres explícitos de DEMO.
        if require_real_token and (i_token is not None) and (i_token < len(row)):
            tok_raw = str(row[i_token] or "").strip().upper()
            if tok_raw:
                # Heurística robusta: DEMO en Deriv suele venir como VRTC*
                es_demo = ("DEMO" in tok_raw) or tok_raw.startswith("VRTC")
                es_real = ("REAL" in tok_raw) or tok_raw.startswith("CR")
                if es_demo and not es_real:
                    continue

        # resultado
        try:
            raw_res = row[i_res] if i_res < len(row) else ""
        except Exception:
            raw_res = ""
        res_norm = normalizar_resultado(raw_res)
        if res_norm not in ("GANANCIA", "PÉRDIDA"):
            continue

        # Armamos dict de fila para reutilizar tus extractores robustos
        row_dict_full = {}
        try:
            for j, h in enumerate(header):
                if j < len(row):
                    row_dict_full[str(h).strip()] = row[j]
        except Exception:
            row_dict_full = {}

        # Monto
        monto = None
        try:
            if i_monto is not None and i_monto < len(row):
                monto = _safe_float_local(row[i_monto])
        except Exception:
            monto = None

        # Ciclo
        ciclo = None
        try:
            if i_ciclo is not None and i_ciclo < len(row):
                ciclo = _safe_float_local(row[i_ciclo])
                ciclo = int(float(ciclo)) if ciclo is not None else None
        except Exception:
            ciclo = None

        # Si esperamos un ciclo concreto, descarta cierres de otro ciclo.
        if expected_ciclo is not None and ciclo is not None:
            try:
                if int(ciclo) != int(expected_ciclo):
                    continue
            except Exception:
                pass

        # Si esperamos un ciclo concreto, descarta cierres de otro ciclo.
        if expected_ciclo is not None and ciclo is not None:
            try:
                if int(ciclo) != int(expected_ciclo):
                    continue
            except Exception:
                pass

        # Si esperamos un ciclo concreto, descarta cierres de otro ciclo.
        if expected_ciclo is not None and ciclo is not None:
            try:
                if int(ciclo) != int(expected_ciclo):
                    continue
            except Exception:
                pass

        # Si esperamos un ciclo concreto, descarta cierres de otro ciclo.
        if expected_ciclo is not None and ciclo is not None:
            try:
                if int(ciclo) != int(expected_ciclo):
                    continue
            except Exception:
                pass

        # payout_total: preferimos extractor (maneja legacy y ratio)
        payout_total = None
        try:
            # Si el CSV trae explícito, dale prioridad
            if i_payout_total is not None and i_payout_total < len(row):
                payout_total = _safe_float_local(row[i_payout_total])
        except Exception:
            payout_total = None

        if payout_total is None:
            # Asegurar que el dict tenga keys útiles si existen en el CSV
            try:
                if i_payout_mult is not None and i_payout_mult < len(row):
                    row_dict_full["payout_multiplier"] = row[i_payout_mult]
                if i_payout_dec is not None and i_payout_dec < len(row):
                    row_dict_full["payout_decimal_rounded"] = row[i_payout_dec]
                if i_payout_legacy is not None and i_payout_legacy < len(row):
                    row_dict_full["payout"] = row[i_payout_legacy]
                if monto is not None:
                    row_dict_full["monto"] = monto
            except Exception:
                pass

            try:
                payout_total = extraer_payout_total(row_dict_full)
            except Exception:
                payout_total = None

        # Devolver lo encontrado (payout_total puede ser None si no hay info suficiente)
        try:
            monto_out = float(monto) if monto is not None else None
        except Exception:
            monto_out = None

        return (res_norm, monto_out, ciclo, payout_total)

    return None

def detectar_martingala_perdida_completa(bot):
    """
    Detecta si se perdió una Martingala completa:
    últimos MAX_CICLOS resultados definitivos son todos PÉRDIDA
    (con normalización robusta del resultado).
    """
    path = f"registro_enriquecido_{bot}.csv"
    if not os.path.exists(path):
        return False

    rows = None
    header = None
    for enc in ("utf-8", "latin-1", "windows-1252"):
        try:
            with open(path, "r", encoding=enc, errors="replace", newline="") as f:
                rows = list(csv.reader(f))
            if rows and len(rows) >= 2:
                header = rows[0]
                break
        except Exception:
            continue

    if not rows or not header or len(rows) < 2:
        return False

    def idx(col):
        return header.index(col) if col in header else None

    res_idx = idx("resultado")
    trade_idx = idx("trade_status")  # <- ahora sí existe siempre

    if res_idx is None:
        return False

    ult = []
    for row in reversed(rows[1:]):
        if not row or len(row) <= res_idx:
            continue

        # Si existe trade_status, exigir CERRADO para contar en rachas
        if trade_idx is not None:
            if len(row) <= trade_idx:
                continue
            ts = (row[trade_idx] or "").strip().upper()
            if ts != "CERRADO":
                continue

        res_norm = normalizar_resultado((row[res_idx] or "").strip())
        if res_norm not in ("GANANCIA", "PÉRDIDA"):
            continue

        ult.append(res_norm)
        if len(ult) >= MAX_CICLOS:
            break

    if len(ult) < MAX_CICLOS:
        return False

    return all(x == "PÉRDIDA" for x in ult)

# Reinicio completo - Corregido para no resetear métricas en modo suave
def reiniciar_completo(borrar_csv=False, limpiar_visual_segundos=15, modo_suave=True):
    global LIMPIEZA_PANEL_HASTA, marti_paso, marti_activa, marti_ciclos_perdidos
    with file_lock():
        write_token_atomic(TOKEN_FILE, "REAL:none")
    
    if borrar_csv and os.path.exists("dataset_incremental.csv"):
        os.remove("dataset_incremental.csv")

    for bot in BOT_NAMES:
        if borrar_csv:
            archivo = f"registro_enriquecido_{bot}.csv"
            if os.path.exists(archivo):
                os.remove(archivo)
            estado_bots[bot]["resultados"] = []
            huellas_usadas[bot] = set()
            estado_bots[bot]["ganancias"] = 0
            estado_bots[bot]["perdidas"] = 0
            estado_bots[bot]["porcentaje_exito"] = None
            estado_bots[bot]["tamano_muestra"] = 0
        elif not modo_suave:
            estado_bots[bot]["resultados"] = []
            estado_bots[bot]["ganancias"] = 0
            estado_bots[bot]["perdidas"] = 0
            estado_bots[bot]["porcentaje_exito"] = None
            estado_bots[bot]["tamano_muestra"] = 0
        estado_bots[bot].update({
            "token": "DEMO",
            "trigger_real": False,
            "prob_ia": 0.0,
            "ia_ready": False,
            "ciclo_actual": 1,
            "modo_real_anunciado": False,
            "ultimo_resultado": None,
            "reintentar_ciclo": False,
            "remate_active": False,
            "remate_start": None,
            "remate_reason": "",
            "fuente": None,
            "real_activado_en": 0.0,  
            "ignore_cierres_hasta": 0.0,
            "modo_ia": "off",
            "ia_seniales": 0,
            "ia_aciertos": 0,
            "ia_fallos": 0,
            "ia_senal_pendiente": False,
            "ia_prob_senal": None
        })
        SNAPSHOT_FILAS[bot] = contar_filas_csv(bot)
        OCULTAR_HASTA_NUEVO[bot] = False  # Cambiado para no ocultar
        IA53_TRIGGERED[bot] = False
        IA90_stats[bot] = {"n": 0, "ok": 0, "pct": 0.0}
        if not isinstance(huellas_usadas.get(bot), set):
            huellas_usadas[bot] = set()
    eventos_recentes.clear()
    for b in BOT_NAMES:
        LAST_REAL_CLOSE_SIG[b] = None
    marti_paso = 0
    marti_activa = False
    marti_ciclos_perdidos = 0
    LIMPIEZA_PANEL_HASTA = time.time() + limpiar_visual_segundos

# Reinicio de bot individual - Corregido similar
def reiniciar_bot(bot, borrar_csv=False):
    # Nunca reiniciar duro al owner REAL activo: durante una operación puede no
    # escribir filas por varios segundos y eso NO significa que deba volver a DEMO.
    owner_activo = REAL_OWNER_LOCK if REAL_OWNER_LOCK in BOT_NAMES else None
    if owner_activo == bot or estado_bots.get(bot, {}).get("token") == "REAL":
        try:
            agregar_evento(f"🛡️ Reinicio omitido para {bot.upper()}: operación REAL en curso.")
        except Exception:
            pass
        return

    if borrar_csv:
        archivo = f"registro_enriquecido_{bot}.csv"
        if os.path.exists(archivo):
            os.remove(archivo)
        estado_bots[bot]["resultados"] = []
        huellas_usadas[bot] = set()
        estado_bots[bot]["ganancias"] = 0
        estado_bots[bot]["perdidas"] = 0
        estado_bots[bot]["porcentaje_exito"] = None
        estado_bots[bot]["tamano_muestra"] = 0
    estado_bots[bot].update({
        "token": "DEMO", 
        "trigger_real": False,
        "prob_ia": 0.0,
        "ia_ready": False,
        "ciclo_actual": 1,
        "modo_real_anunciado": False,
        "ultimo_resultado": None,
        "reintentar_ciclo": False,
        "remate_active": False,
        "remate_start": None,
        "remate_reason": "",
        "fuente": None,
        "real_activado_en": 0.0,  
        "ignore_cierres_hasta": 0.0,
        "modo_ia": "off",
        "ia_seniales": 0,
        "ia_aciertos": 0,
        "ia_fallos": 0,
        "ia_senal_pendiente": False,
        "ia_prob_senal": None
    })
    SNAPSHOT_FILAS[bot] = contar_filas_csv(bot)
    OCULTAR_HASTA_NUEVO[bot] = False  # Cambiado para no ocultar
    IA90_stats[bot] = {"n": 0, "ok": 0, "pct": 0.0}
    LAST_REAL_CLOSE_SIG[bot] = None
    if not isinstance(huellas_usadas.get(bot), set):
        huellas_usadas[bot] = set()

def cerrar_por_fin_de_ciclo(bot: str, reason: str):
    global REAL_OWNER_LOCK
    # Limpieza total de “estado REAL” para evitar HUD/estado fantasma
    try:
        estado_bots[bot]["token"] = "DEMO"
        estado_bots[bot]["trigger_real"] = False
        estado_bots[bot]["ciclo_actual"] = 1
        estado_bots[bot]["modo_real_anunciado"] = False
        estado_bots[bot]["fuente"] = None

        # ✅ Extra blindaje (evita “escudos” de cierre pegados en DEMO)
        estado_bots[bot]["real_activado_en"] = 0.0
        estado_bots[bot]["ignore_cierres_hasta"] = 0.0

        # Flags IA/pending (si quedó algo colgado)
        estado_bots[bot]["ia_senal_pendiente"] = False
        estado_bots[bot]["ia_prob_senal"] = None

        # Remate limpio
        estado_bots[bot]["remate_active"] = False
        estado_bots[bot]["remate_start"] = None
        estado_bots[bot]["remate_reason"] = ""

    except Exception:
        pass

    # Liberar token global REAL
    REAL_OWNER_LOCK = None
    try:
        with file_lock():
            write_token_atomic(TOKEN_FILE, "REAL:none")
    except Exception:
        pass

    # Limpiar orden REAL para evitar re-entradas fantasma (igual que cerrar_por_win)
    try:
        limpiar_orden_real(bot)
    except Exception:
        pass

    # Sync inmediato del HUD/token para evitar “REAL fantasma”
    try:
        _set_ui_token_holder(None)
    except Exception:
        pass

   
    # Actualizar snapshots para que no relea la misma fila
    try:
        REAL_ENTRY_BASELINE[bot] = 0
        SNAPSHOT_FILAS[bot] = contar_filas_csv(bot)
    except Exception:
        pass

    try:
        OCULTAR_HASTA_NUEVO[bot] = False
    except Exception:
        pass

    # Forzar refresco del loop principal
    try:
        reinicio_forzado.set()
    except Exception:
        pass

    # Log visual
    try:
        agregar_evento(f"🔓 Cuenta REAL liberada para {bot.upper()} ({reason})")
    except Exception:
        pass

def registrar_resultado_real(resultado: str, bot: str | None = None, ciclo_operado: int | None = None):
    """
    Actualiza el contador global de ciclos martingala para el HUD y la próxima
    autoasignación REAL.

    Reglas:
    - GANANCIA: resetea a ciclo #1 (contador de pérdidas = 0).
    - PÉRDIDA: incrementa ciclo hasta MAX_CICLOS (tope de blindaje).
    """
    global marti_ciclos_perdidos, marti_paso

    res = normalizar_resultado(resultado)
    if res == "GANANCIA":
        marti_ciclos_perdidos = 0
        marti_paso = 0
    elif res == "PÉRDIDA":
        # Robustez anti-desincronización:
        # si conocemos el ciclo realmente operado, el próximo estado de pérdidas
        # debe ser al menos ese ciclo (p.ej. perder en C2 => pérdidas=2 => próximo C3).
        try:
            ciclo_ref = int(ciclo_operado) if ciclo_operado is not None else 0
        except Exception:
            ciclo_ref = 0
        marti_ciclos_perdidos = min(
            MAX_CICLOS,
            max(int(marti_ciclos_perdidos) + 1, max(0, ciclo_ref))
        )
        # Si ya culminó el 5/5, reinicia a C1 para el siguiente turno.
        if int(marti_ciclos_perdidos) >= int(MAX_CICLOS):
            marti_ciclos_perdidos = 0
            marti_paso = 0
            try:
                agregar_evento("🧯 Martingala 5/5 completada: reinicio automático a ciclo 1.")
            except Exception:
                pass
        else:
            marti_paso = min(MAX_CICLOS - 1, int(marti_ciclos_perdidos))
    else:
        return

    ciclo_sig = int(marti_paso) + 1
    bot_msg = f" [{bot}]" if bot else ""
    agregar_evento(
        f"🔁 Martingala{bot_msg}: resultado={res} | pérdidas seguidas={marti_ciclos_perdidos}/{MAX_CICLOS} | próximo ciclo={ciclo_sig}"
    )

def ciclo_martingala_siguiente() -> int:
    """
    Fuente canónica del ciclo a abrir en REAL:
    - ciclo = pérdidas_consecutivas + 1, con límites [1..MAX_CICLOS]
    """
    try:
        return max(1, min(int(MAX_CICLOS), int(marti_ciclos_perdidos) + 1))
    except Exception:
        return 1

def ciclo_martingala_siguiente() -> int:
    """
    Fuente canónica del ciclo a abrir en REAL:
    - ciclo = pérdidas_consecutivas + 1, con límites [1..MAX_CICLOS]
    """
    try:
        return max(1, min(int(MAX_CICLOS), int(marti_ciclos_perdidos) + 1))
    except Exception:
        return 1

def ciclo_martingala_siguiente() -> int:
    """
    Fuente canónica del ciclo a abrir en REAL:
    - ciclo = pérdidas_consecutivas + 1, con límites [1..MAX_CICLOS]
    """
    try:
        return max(1, min(int(MAX_CICLOS), int(marti_ciclos_perdidos) + 1))
    except Exception:
        return 1

# === FIN BLOQUE 9 ===

# === BLOQUE 10 — IA: DATASET, MODELO Y PREDICCIÓN ===
# Caché y hot-reload de activos del oráculo
_MODEL_PATH = "modelo_xgb.pkl"  # Actualizado para XGBoost
_SCALER_PATH = "scaler.pkl"
_FEATURES_PATH = "feature_names.pkl"
_META_PATH = "model_meta.json"

_ORACLE_CACHE = {
    "model": None,
    "scaler": None,
    "features": None,
    "meta": None,
    "mtimes": {}  # {path: mtime}
}
# ============================
# PATCH IA (FIX): Label canónico + builder X/y ultra-robusto
# - NO asume columna 'y'
# - Filtra y a {0,1} (incluye GANANCIA/PÉRDIDA/✓/✗)
# - X por reindex => nunca KeyError
# - NaN/inf => 0.0
# ============================

# ============================
# IA — Label canónico (una sola vez)
# ============================
# ============================
# IA — Label canónico (FIX) + builder X/y ultra-robusto
# - NO asume columna 'y'
# - Filtra y a {0,1} (incluye GANANCIA/PÉRDIDA/✓/✗)
# - X por reindex => nunca KeyError
# - NaN/inf => 0.0
# ============================

LABEL_CANON = "result_bin"
LABEL_CANDIDATES = (
    "result_bin", "y", "label", "target", "resultado_bin", "result",
    "resultado", "win", "outcome"
)

def _pick_label_col_incremental(df: pd.DataFrame) -> str:
    """
    Devuelve el nombre REAL de la columna label dentro del DF (respetando el nombre original),
    aunque haya espacios/casos raros. Evita KeyError por "limpieza" de strings.
    """
    try:
        if df is None or df.empty or getattr(df, "columns", None) is None:
            return LABEL_CANON
    except Exception:
        return LABEL_CANON

    # mapa: "limpio_lower" -> "original"
    try:
        colmap = {str(c).strip().lower(): c for c in df.columns}
    except Exception:
        return LABEL_CANON

    for cand in LABEL_CANDIDATES:
        key = str(cand).strip().lower()
        if key in colmap:
            return colmap[key]

    # fallback defensivo: última columna real del DF
    try:
        return df.columns[-1]
    except Exception:
        return LABEL_CANON

def _y_to_bin(v) -> int | None:
    """
    Normaliza cualquier cosa a {0,1}. Si no se puede interpretar, devuelve None.
    Acepta: 0/1, "0"/"1", GANANCIA/PÉRDIDA, WIN/LOSS, ✓/✗, etc.
    """
    try:
        if v is None:
            return None
        # pandas NaN
        try:
            if pd.isna(v):
                return None
        except Exception:
            pass

        # numérico directo
        if isinstance(v, (int, np.integer)):
            return int(v) if int(v) in (0, 1) else None
        if isinstance(v, (float, np.floating)):
            if not math.isfinite(float(v)):
                return None
            iv = int(round(float(v)))
            return iv if iv in (0, 1) else None

        s = str(v).strip()
        if s == "":
            return None

        # "0"/"1"
        if s in ("0", "1"):
            return int(s)

        # símbolos / texto (reutiliza tu normalizador)
        rn = normalizar_resultado(s)
        if rn == "GANANCIA":
            return 1
        if rn == "PÉRDIDA":
            return 0

        # extra fallback simple por si llega raro
        up = s.upper()
        if "WIN" in up or "GAN" in up:
            return 1
        if "LOSS" in up or "PERD" in up:
            return 0

        return None
    except Exception:
        return None

def construir_Xy_incremental(
    df: pd.DataFrame,
    feature_names: list | None = None
) -> tuple[pd.DataFrame, np.ndarray, str, list]:
    """
    Construye X/y sin reventar:
    - y se infiere desde la mejor columna label detectada.
    - X se reindexa a feature_names => columnas faltantes se crean (NaN->0.0).
    - NaN/inf => 0.0
    Retorna: (X_df, y_np, label_col_real, features_usadas)
    """
    feats = list(feature_names) if feature_names else list(INCREMENTAL_FEATURES_V2)

    if df is None or df.empty:
        return pd.DataFrame(columns=feats), np.array([], dtype=int), LABEL_CANON, feats

    label_col = _pick_label_col_incremental(df)

    if label_col not in df.columns:
        # fallback brutal si algo raro pasó
        label_col = df.columns[-1] if len(df.columns) else LABEL_CANON

    y_series = df[label_col].apply(_y_to_bin)
    mask = y_series.notna()

    if not mask.any():
        return pd.DataFrame(columns=feats), np.array([], dtype=int), label_col, feats

    df2 = df.loc[mask].copy()
    y = y_series.loc[mask].astype(int).to_numpy()

    # X por reindex => nunca KeyError
    X = df2.reindex(columns=feats)

    # numeric coercion + limpieza
    for c in feats:
        try:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        except Exception:
            X[c] = np.nan

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    return X, y, label_col, feats

def cargar_incremental_Xy(
    ruta: str = "dataset_incremental.csv",
    feature_names: list | None = None
) -> tuple[pd.DataFrame, np.ndarray, str, list]:
    """
    Loader robusto:
    - Si el incremental está "mutante", intenta repararlo.
    - Lee con encodings fallback.
    - Devuelve X/y listos para entrenar.
    """
    if not os.path.exists(ruta):
        feats = list(feature_names) if feature_names else list(INCREMENTAL_FEATURES_V2)
        return pd.DataFrame(columns=feats), np.array([], dtype=int), LABEL_CANON, feats

    # Reparación preventiva (si quedó mutante)
    try:
        cols = _canonical_incremental_cols(feature_names if feature_names else INCREMENTAL_FEATURES_V2)
        reparar_dataset_incremental_mutante(ruta=ruta, cols=cols)
    except Exception:
        pass

    df = None
    for enc in ("utf-8", "latin-1", "windows-1252"):
        try:
            df = pd.read_csv(ruta, sep=",", encoding=enc, engine="python", on_bad_lines="skip")
            break
        except Exception:
            continue

    if df is None or df.empty:
        feats = list(feature_names) if feature_names else list(INCREMENTAL_FEATURES_V2)
        return pd.DataFrame(columns=feats), np.array([], dtype=int), LABEL_CANON, feats

    return construir_Xy_incremental(df, feature_names=feature_names)
# === FIN PATCH IA (FIX) ===

def _coerce_label_to_01(series: pd.Series) -> pd.Series:
    """
    Convierte etiquetas a 0/1:
    - acepta 0/1 numérico
    - acepta strings tipo 'GANANCIA', 'PÉRDIDA', 'WIN', 'LOSS', '✓', '✗'
    - lo demás => NaN
    """
    out = pd.Series(np.nan, index=series.index, dtype="float64")

    # 1) numérico directo (incluye "0", "1", 0.0, 1.0, True/False)
    y_num = pd.to_numeric(series, errors="coerce")
    ok01 = y_num.isin([0, 1])
    if ok01.any():
        out.loc[ok01] = y_num.loc[ok01].astype(float)

    # 2) strings/símbolos
    try:
        s = series.astype(str)
    except Exception:
        s = pd.Series([""] * len(series), index=series.index)

    def _map_one(x: str):
        try:
            raw = (x or "").strip()
            if raw == "" or raw.lower() == "nan":
                return np.nan

            # símbolos primero
            if any(sym in raw for sym in ("✓", "✔", "✅", "🟢")):
                return 1.0
            if any(sym in raw for sym in ("✗", "❌", "🔴", "🟥")):
                return 0.0

            t = raw.upper()
            if "GAN" in t or "WIN" in t:
                return 1.0
            if "PERD" in t or "LOSS" in t:
                return 0.0

            # admitir "1.0" / "0.0" como texto
            try:
                v = float(raw.replace(",", "."))
                if v in (0.0, 1.0):
                    return float(int(v))
            except Exception:
                pass

            return np.nan
        except Exception:
            return np.nan

    mapped = s.map(_map_one)
    out = out.fillna(mapped)
    return out


def build_xy_from_incremental(df: pd.DataFrame, feature_names: list | None = None):
    """
    Builder ultra-robusto:
    - label col canónica (o fallback a última)
    - y a {0,1} con coerción
    - X con reindex => jamás KeyError
    - NaN/inf => 0.0
    Devuelve: X, y, label_col
    """
    if df is None or getattr(df, "empty", True):
        return None, None, None

    feats = list(feature_names) if feature_names else list(INCREMENTAL_FEATURES_V2)
    label_col = _pick_label_col_incremental(df)

    # y
    try:
        y01 = _coerce_label_to_01(df[label_col])
    except Exception:
        return None, None, label_col

    mask = y01.isin([0.0, 1.0])
    if int(mask.sum()) <= 0:
        return None, None, label_col

    # X (reindex = blindaje)
    X = df.reindex(columns=feats, fill_value=0.0).loc[mask].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # y final
    y = y01.loc[mask].astype(int)

    return X, y, label_col

def _clean_X_df(X: pd.DataFrame) -> pd.DataFrame:
    # a numérico, NaN/inf => 0.0, float
    try:
        for c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    except Exception:
        pass

    try:
        X = X.replace([np.inf, -np.inf], np.nan)
    except Exception:
        pass

    try:
        X = X.fillna(0.0)
    except Exception:
        pass

    try:
        X = X.astype("float64")
    except Exception:
        pass

    return X
    
def _build_Xy_incremental(df: pd.DataFrame, feature_names: list | None = None):
    """
    Wrapper canónico (compat):
    Devuelve: X_df, y_arr, feature_names_usadas, label_col_real

    Regla:
    - Internamente usa build_xy_from_incremental (única fuente de verdad).
    - Evita entrenar si queda 1 sola clase.
    """
    feats = list(feature_names) if feature_names else list(INCREMENTAL_FEATURES_V2)

    X, y, label_col = build_xy_from_incremental(df, feats)
    if X is None or y is None:
        return None, None, feats, (label_col or LABEL_CANON)

    # Evitar entrenos falsos con una sola clase
    try:
        if len(set(np.unique(y.to_numpy()))) < 2:
            return None, None, feats, (label_col or LABEL_CANON)
    except Exception:
        pass

    X = _clean_X_df(X)
    try:
        y_arr = y.astype(int).to_numpy()
    except Exception:
        y_arr = np.asarray(y, dtype=int)

    return X, y_arr, feats, (label_col or LABEL_CANON)

# Duplicado eliminado: se usa la versión canónica de _build_Xy_incremental definida arriba.
# Esto evita inconsistencias silenciosas en features/label durante entrenamiento y predicción.


# ============================================================
# COMPAT: FEATURES legacy (evita NameError en entreno)
# - Si existe dataset_incremental.csv, usa su header real
# - Si no existe, cae a FEATURE_NAMES_DEFAULT
# ============================================================
def _infer_features_from_incremental(path: str = "dataset_incremental.csv", fallback=None):
    try:
        if fallback is None:
            fallback = globals().get("FEATURE_NAMES_DEFAULT", None)

        if not os.path.exists(path):
            return list(fallback) if fallback else None

        for enc in ("utf-8", "utf-8-sig", "latin-1", "windows-1252"):
            try:
                with open(path, "r", newline="", encoding=enc, errors="replace") as f:
                    reader = csv.reader(f)
                    header = next(reader, [])
                header = [str(c).strip() for c in header if str(c).strip()]
                header = [c for c in header if c != "result_bin"]  # label fuera
                if header:
                    return header
            except Exception:
                continue

        return list(fallback) if fallback else None
    except Exception:
        return list(fallback) if fallback else None

# Si el código viejo usa FEATURES, aquí lo blindamos:
if "FEATURES" not in globals() or not globals().get("FEATURES"):
    FEATURES = _infer_features_from_incremental(fallback=globals().get("FEATURE_NAMES_DEFAULT", None))

# Último fallback ultra-defensivo
if not FEATURES:
    FEATURES = list(globals().get("FEATURE_NAMES_DEFAULT", []))

_META_CORRUPT_FLAG = False  # Nueva bandera para evitar reintentos en meta corrupto

def _safe_mtime(path):
    try:
        return os.path.getmtime(path)
    except Exception:
        return -1
def _as_list_feature_names(x):
    """Convierte feature names a list[str] de forma segura."""
    if x is None:
        return []
    if isinstance(x, list):
        return [str(a) for a in x]
    if isinstance(x, tuple):
        return [str(a) for a in list(x)]
    if isinstance(x, set):
        return [str(a) for a in list(x)]
    # string (ej: "['a','b']") no lo evaluamos; lo tratamos como 1 feature
    return [str(x)]


def _normalize_model_meta(meta: dict) -> dict:
    """
    Compatibilidad de meta:
    - rows_total -> n_samples
    - pos+neg -> n_samples (fallback)
    - crea alias n -> n_samples
    - normaliza tipos básicos
    """
    if not isinstance(meta, dict):
        return {}

    m = dict(meta)

    def _to_int(v):
        try:
            return int(float(v))
        except Exception:
            return 0

    def _to_float(v):
        try:
            return float(v)
        except Exception:
            return v

    # 1) n_samples
    ns = _to_int(m.get("n_samples"))
    if ns <= 0:
        ns = _to_int(m.get("rows_total"))
    if ns <= 0:
        ns = _to_int(m.get("pos")) + _to_int(m.get("neg"))
    m["n_samples"] = ns

    # 2) alias legacy
    if _to_int(m.get("n")) <= 0:
        m["n"] = ns

    # 3) floats útiles si existen
    for k in ("auc", "cv_auc", "f1", "threshold"):
        if k in m:
            m[k] = _to_float(m[k])

    # 4) reliable como bool
    m["reliable"] = bool(m.get("reliable", False))

    # 5) calibration string
    if "calibration" in m and m["calibration"] is None:
        m["calibration"] = "none"

    return m


def _resolve_oracle_feature_names(model, scaler, features, meta):
    """
    Prioridad EXACTA:
      1) scaler.feature_names_in_
      2) features (features.pkl)
      3) meta['feature_names']
      4) model.feature_names_in_ (fallback raro)
      5) None
    """
    # 1) scaler.feature_names_in_
    try:
        if scaler is not None and hasattr(scaler, "feature_names_in_"):
            fn = _as_list_feature_names(getattr(scaler, "feature_names_in_", None))
            if fn:
                return fn
    except Exception:
        pass

    # 2) features.pkl
    fn = _as_list_feature_names(features)
    if fn:
        return fn

    # 3) meta['feature_names']
    try:
        if meta and isinstance(meta, dict):
            fn = _as_list_feature_names(meta.get("feature_names"))
            if fn:
                return fn
    except Exception:
        pass

    # 4) model.feature_names_in_ (por si entrenaste con DataFrame directo al modelo)
    try:
        if model is not None and hasattr(model, "feature_names_in_"):
            fn = _as_list_feature_names(getattr(model, "feature_names_in_", None))
            if fn:
                return fn
    except Exception:
        pass

    return None


def get_oracle_assets():
    """
    Devuelve SIEMPRE: (model, scaler, features, meta)
    - model: modelo calibrado o None
    - scaler: StandardScaler o None
    - features: lista de nombres de features o None
    - meta: dict o None

    Blindaje:
    - Cache por mtime
    - Si meta se corrompe: renombra .corrupt y sigue con meta=None
    """
    changed = False
    for path in (_MODEL_PATH, _SCALER_PATH, _FEATURES_PATH, _META_PATH):
        mt = _safe_mtime(path)
        if _ORACLE_CACHE["mtimes"].get(path) != mt:
            _ORACLE_CACHE["mtimes"][path] = mt
            changed = True

    if _ORACLE_CACHE["model"] is None or changed:
        # Modelo
        try:
            _ORACLE_CACHE["model"] = joblib.load(_MODEL_PATH) if os.path.exists(_MODEL_PATH) else None
        except Exception as e:
            print(f"⚠️ IA: Error cargando modelo: {e}")
            _ORACLE_CACHE["model"] = None

        # Scaler
        try:
            _ORACLE_CACHE["scaler"] = joblib.load(_SCALER_PATH) if os.path.exists(_SCALER_PATH) else None
        except Exception as e:
            print(f"⚠️ IA: Error cargando scaler: {e}")
            _ORACLE_CACHE["scaler"] = None

        # Features
        try:
            _ORACLE_CACHE["features"] = joblib.load(_FEATURES_PATH) if os.path.exists(_FEATURES_PATH) else None
        except Exception as e:
            print(f"⚠️ IA: Error cargando features: {e}")
            _ORACLE_CACHE["features"] = None

        # Meta
        try:
            if os.path.exists(_META_PATH):
                with open(_META_PATH, "r", encoding="utf-8") as f:
                    _ORACLE_CACHE["meta"] = json.load(f)
            else:
                _ORACLE_CACHE["meta"] = None
        except Exception as e:
            print(f"⚠️ IA: Error cargando meta (archivo corrupto): {e}. Renombrando a .corrupt.")
            try:
                os.rename(_META_PATH, _META_PATH + ".corrupt")
            except Exception:
                pass
            _ORACLE_CACHE["meta"] = None

    # Normalización suave de features/meta (para evitar cosas raras)
    try:
        _ORACLE_CACHE["features"] = _as_list_feature_names(_ORACLE_CACHE.get("features"))
    except Exception:
        pass

    try:
        if isinstance(_ORACLE_CACHE.get("meta"), dict):
            _ORACLE_CACHE["meta"] = _normalize_model_meta(_ORACLE_CACHE["meta"])
        else:
            _ORACLE_CACHE["meta"] = None
    except Exception:
        _ORACLE_CACHE["meta"] = None

    return _ORACLE_CACHE["model"], _ORACLE_CACHE["scaler"], _ORACLE_CACHE["features"], _ORACLE_CACHE["meta"]

def oraculo_predict_visible(fila_dict):
    """
    Predicción para HUD:
    - Si hay modelo: usa oraculo_predict(modelo+scaler+meta/features)
    - Si no hay modelo y LOW_DATA_MODE: usa prob_exploratoria
    - Si no: 0.0
    """
    try:
        model, scaler, features, meta = get_oracle_assets()

        if model is not None:
            meta_local = dict(meta or {})
            fn = _resolve_oracle_feature_names(model, scaler, features, meta_local)
            if fn:
                meta_local["feature_names"] = fn
                prob = oraculo_predict(fila_dict, model, scaler, meta_local, bot_name="HUD")
                return prob, "modelo"

        # Sin modelo: fallback visual (NO opera, solo pinta)
        if LOW_DATA_MODE and contar_filas_incremental() >= MIN_FIT_ROWS_LOW:
            prob = prob_exploratoria(fila_dict)
            return prob, "low_data"

        # Modo exploración visual si existe la bandera
        if globals().get("MODO_EXPLORACION_IA", False):
            prob = prob_exploratoria(fila_dict)
            return prob, "exp"

        return 0.0, "off"

    except Exception as e:
        print(f"⚠️ IA: Error en predict visible: {e}")
        return 0.0, "off"

def get_threshold_sugerido(default_=0.60):
    global _META_CORRUPT_FLAG
    try:
        meta = _ORACLE_CACHE.get("meta")
        if meta is None and not _META_CORRUPT_FLAG:
            if os.path.exists(_META_PATH):
                with open(_META_PATH, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    _ORACLE_CACHE["meta"] = meta
        thr = meta.get("threshold") if meta else None
        if isinstance(thr, (int, float)):
            return float(thr)
    except Exception as e:
        if not _META_CORRUPT_FLAG:
            print(f"⚠️ IA: Error en threshold sugerido (meta corrupto): {e}. Renombrando archivo y usando default.")
            try:
                os.rename(_META_PATH, _META_PATH + ".corrupt")
            except Exception:
                pass
            _META_CORRUPT_FLAG = True  # Evita reintentos
    return float(default_)

def modelo_es_reliable(default=False):
    """
    Usa meta['reliable'] si existe; respalda en default si falta el meta.
    """
    try:
        meta = _ORACLE_CACHE.get("meta")
        if not meta and os.path.exists(_META_PATH):
            with open(_META_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
                _ORACLE_CACHE["meta"] = meta
        return bool(meta and meta.get("reliable"))
    except Exception:
        return bool(default)


# Oráculo robusto
def oraculo_predict(fila_dict, modelo, scaler, meta, bot_name=""):
    """
    Predicción IA robusta.
    payout se trata como ROI [0..1.5].
    """
    try:
        if fila_dict is None:
            return 0.0

        feature_names = _resolve_oracle_feature_names(modelo, scaler, (meta or {}).get("feature_names"), meta or {})
        if not feature_names:
            # último fallback: si meta traía feature_names directo como lista
            feature_names = _as_list_feature_names((meta or {}).get("feature_names"))
        if not feature_names:
            return 0.0

        # =========================================================
        # Completar features derivados si el modelo los requiere
        # (para que inferencia = entrenamiento)
        # =========================================================
        ra = fila_dict.get("racha_actual", 0.0)
        try:
            ra = float(ra)
        except Exception:
            ra = 0.0

        if "racha_signo" in feature_names and "racha_signo" not in fila_dict:
            fila_dict["racha_signo"] = float(np.sign(ra))

        if "racha_abs" in feature_names and "racha_abs" not in fila_dict:
            fila_dict["racha_abs"] = float(abs(ra))

        if "rebote_fuerte" in feature_names and "rebote_fuerte" not in fila_dict:
            esr = fila_dict.get("es_rebote", 0.0)
            try:
                esr = float(esr)
            except Exception:
                esr = 0.0
            fila_dict["rebote_fuerte"] = 1.0 if (esr >= 0.5 and ra <= -4) else 0.0

        if "pay_x_puntaje" in feature_names and "pay_x_puntaje" not in fila_dict:
            fila_dict["pay_x_puntaje"] = float(fila_dict.get("payout", 0.0) or 0.0) * float(fila_dict.get("puntaje_estrategia", 0.0) or 0.0)

        if "vol_x_breakout" in feature_names and "vol_x_breakout" not in fila_dict:
            fila_dict["vol_x_breakout"] = float(fila_dict.get("volatilidad", 0.0) or 0.0) * float(fila_dict.get("breakout", 0.0) or 0.0)

        if "hora_x_rebote" in feature_names and "hora_x_rebote" not in fila_dict:
            fila_dict["hora_x_rebote"] = float(fila_dict.get("hora_bucket", 0.0) or 0.0) * float(fila_dict.get("es_rebote", 0.0) or 0.0)

        if not feature_names:
            return 0.0

        # Asegurar payout como ROI [0..1.5] (si falta, derivarlo)
        if "payout" in feature_names:
            if "payout" not in fila_dict or fila_dict.get("payout") in (None, "", 0, 0.0):
                roi_tmp = calcular_payout_feature(fila_dict)
                if roi_tmp is not None:
                    fila_dict["payout"] = roi_tmp

        # Clamp final por seguridad
        if "payout" in fila_dict:
            try:
                p = float(fila_dict["payout"])
            except Exception:
                p = 0.0
            if not math.isfinite(p):
                p = 0.0
            fila_dict["payout"] = max(0.0, min(p, 1.5))
        # Normalizar features faltantes (TRAIN vs INFER): volatilidad + hora_bucket
        try:
            fila_dict["volatilidad"] = float(calcular_volatilidad_simple(fila_dict))
        except Exception:
            fila_dict["volatilidad"] = 0.0

        try:
            fila_dict["hora_bucket"] = float(calcular_hora_bucket(fila_dict))
        except Exception:
            fila_dict["hora_bucket"] = 0.5

        # Armar X en orden del modelo
        X = []
        for col in feature_names:
            v = fila_dict.get(col, 0.0)
            try:
                v = float(v)
            except Exception:
                v = 0.0
            if not math.isfinite(v):
                v = 0.0
            X.append(v)

        X = np.array(X, dtype=float).reshape(1, -1)

        # Escalar si existe scaler
        if scaler is not None:
            try:
                X = scaler.transform(X)
            except Exception:
                pass

        # Predecir proba
        if hasattr(modelo, "predict_proba"):
            proba = modelo.predict_proba(X)[0][1]
        else:
            # fallback
            proba = float(modelo.predict(X)[0])

        try:
            proba = float(proba)
        except Exception:
            proba = 0.0

        if not math.isfinite(proba):
            proba = 0.0
        return max(0.0, min(proba, 1.0))

    except Exception:
        return 0.0
def prob_exploratoria(fila):
    """
    Probabilidad simple (heurística) solo para VISUAL / fallback.
    payout = ROI [0..1.5]
    """
    try:
        # payout = ROI [0..1.5] (solo visual)
        pay = calcular_payout_feature(fila)
        try:
            pay = float(pay) if pay is not None else 0.0
        except Exception:
            pay = 0.0
        if not math.isfinite(pay):
            pay = 0.0
        pay = max(0.0, min(pay, 1.5))

        # score básico (conservar simple)
        score = 0.50
        if str(fila.get("breakout", 0)).strip() in ("1", "True", "true"):
            score += 0.05
        if str(fila.get("cruce_sma", 0)).strip() in ("1", "True", "true"):
            score += 0.05
        if str(fila.get("rsi_reversion", 0)).strip() in ("1", "True", "true"):
            score += 0.04

        # ROI alto ayuda un poco (sin convertir a %)
        score += (pay / 1.5) * 0.10

        if not math.isfinite(score):
            score = 0.0
        return max(0.0, min(score, 1.0))
    except Exception:
        return 0.0

# --- Nueva: leer_model_meta (blindado) ---
def leer_model_meta():
    global _META_CORRUPT_FLAG
    try:
        if _META_CORRUPT_FLAG:
            return {}

        if os.path.exists(_META_PATH):
            with open(_META_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f) or {}
            if not isinstance(meta, dict):
                return {}
            return _normalize_model_meta(meta)

        return {}
    except Exception as e:
        try:
            corrupt = f"{_META_PATH}.corrupt_{int(time.time())}"
            if os.path.exists(_META_PATH):
                os.replace(_META_PATH, corrupt)
                agregar_evento(f"⚠️ META corrupta. Renombrada a {corrupt} ({e})")
        except Exception:
            pass
        _META_CORRUPT_FLAG = True
        return {}

# --- Nueva: guardar_model_meta (atómico) ---
def guardar_model_meta(meta: dict):
    try:
        tmp = _META_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=4)
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, _META_PATH)
        _ORACLE_CACHE["meta"] = meta
    except Exception as e:
        print(f"⚠️ IA: Falló guardar meta: {e}")
# =========================================================
# GUARDADO ATÓMICO DE ARTEFACTOS IA (modelo/scaler/features/meta)
# Evita corrupción y "archivos fantasma" al reiniciar.
# =========================================================
def _joblib_dump_atomic(obj, path: str):
    tmp = path + ".tmp"
    try:
        joblib.dump(obj, tmp)
        os.replace(tmp, path)
        return True
    except Exception:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        return False

def _json_dump_atomic(data: dict, path: str):
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        return True
    except Exception:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        return False

def guardar_oracle_assets_atomico(modelo, scaler, feature_names, meta: dict | None):
    """
    Escribe de forma atómica:
      - modelo_xgb.pkl   (_MODEL_PATH)
      - scaler.pkl       (_SCALER_PATH)
      - feature_names.pkl(_FEATURES_PATH)
      - model_meta.json  (_META_PATH)
    """
    def _ensure_parent(path: str):
        try:
            d = os.path.dirname(path)
            if d:
                os.makedirs(d, exist_ok=True)
        except Exception:
            pass

    def _dump_atomic(obj, path: str):
        _ensure_parent(path)
        tmp = path + ".tmp"
        joblib.dump(obj, tmp)
        os.replace(tmp, path)

    def _json_atomic(obj, path: str):
        _ensure_parent(path)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    try:
        if meta is None or not isinstance(meta, dict):
            meta = {}

        # asegurar feature_names dentro de meta
        try:
            meta["feature_names"] = list(feature_names) if feature_names else []
        except Exception:
            meta["feature_names"] = []

        # compat: asegurar n_samples/n y tipos básicos
        try:
            meta = _normalize_model_meta(meta)
        except Exception:
            pass

        _dump_atomic(modelo, _MODEL_PATH)
        _dump_atomic(scaler, _SCALER_PATH)
        _dump_atomic(list(feature_names) if feature_names else [], _FEATURES_PATH)
        _json_atomic(meta, _META_PATH)

        try:
            agregar_evento("💾 IA: artefactos guardados (modelo+scaler+features+meta)")
        except Exception:
            pass

        return True

    except Exception as e:
        # limpieza best-effort de temporales si quedaron
        for p in (_MODEL_PATH, _SCALER_PATH, _FEATURES_PATH, _META_PATH):
            try:
                tmp = p + ".tmp"
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass

        try:
            agregar_evento(f"⚠️ IA: fallo guardando artefactos: {e}")
        except Exception:
            pass
        return False
     
# --- Nueva: get_prob_ia_historica (estable) ---
def get_prob_ia_historica(bot: str) -> float:
    try:
        sig = estado_bots[bot]["ia_seniales"]
        acc = estado_bots[bot]["ia_aciertos"]
        if sig >= MIN_IA_SENIALES_CONF:
            return acc / sig
        return 0.0
    except Exception:
        return 0.0

# --- Nueva: calcular_confianza_ia (para HUD) ---
def calcular_confianza_ia(bot: str, meta: dict) -> float:
    try:
        auc = float(meta.get("auc", 0.0))
        n = int(meta.get("n_samples", 0))
        reliable = bool(meta.get("reliable", False))
        sig = estado_bots[bot]["ia_seniales"]
        acc = estado_bots[bot]["ia_aciertos"]
        pct = acc / sig if sig > 0 else 0.0
        conf = (auc * 0.4) + (pct * 0.3) + (0.2 if reliable else 0.0) + (0.1 if n >= MIN_FIT_ROWS_PROD else 0.0)
        return min(1.0, max(0.0, conf))
    except Exception:
        return 0.0

# --- Nueva: get_umbral_dinamico (para audio/HUD) ---
def get_umbral_dinamico(meta: dict, base_thr: float) -> float:
    try:
        auc = float(meta.get("auc", 0.0))
        delta = max(0.0, min(0.15, (auc - 0.75) * 0.2))
        return max(0.5, base_thr - delta)
    except Exception:
        return base_thr
def get_umbral_operativo(meta: dict | None = None) -> float:
    """
    Umbral único de operación IA (HUD, audio, selección).

    Regla dura:
    - Si el modelo NO es confiable, si el AUC es bajo, o si hay pocos samples,
      se bloquea cualquier señal (umbral ~ imposible).
    """
    base_thr = get_threshold_sugerido(IA_METRIC_THRESHOLD)
    if base_thr < IA_METRIC_THRESHOLD:
        base_thr = IA_METRIC_THRESHOLD
    if base_thr < ORACULO_THR_MIN:
        base_thr = ORACULO_THR_MIN

    if meta is None:
        try:
            meta = leer_model_meta()
        except Exception:
            meta = {}

    # Meta robusta
    try:
        auc = float((meta or {}).get("auc", 0.0) or 0.0)
    except Exception:
        auc = 0.0

    try:
        reliable = bool((meta or {}).get("reliable", False))
    except Exception:
        reliable = False

    try:
        n_samples = int((meta or {}).get("n_samples", (meta or {}).get("n", 0)) or 0)
    except Exception:
        n_samples = 0

    thr = get_umbral_dinamico(meta or {}, base_thr)

    # 🔒 BLOQUEO DE SEÑALES CUANDO ES EXPERIMENTAL / BAJO DATOS
    MIN_AUC_GREEN = 0.55  # “al menos no somos un dado”
    if (not reliable) or (n_samples < MIN_FIT_ROWS_PROD) or (auc < MIN_AUC_GREEN):
        return 0.99

    return thr
# =========================================================
# DISPARADOR ÚNICO DE ALERTA IA (AUDIO + FLAG)
# Regla dura pedida:
#   - SOLO dispara si prob >= 70% (o el umbral operativo si es más alto)
#   - Blindado contra prob en % (53) vs fracción (0.53)
#   - Cooldown + rearme por histéresis
# =========================================================
def _umbral_alerta_ia(meta: dict | None = None) -> float:
    """
    Umbral del aviso de audio IA (fijo, como definiste en config).
    Devuelve un float en [0..1].
    """
    try:
        thr = float(AUDIO_IA53_THR)
    except Exception:
        thr = 0.75
    if thr < 0.0:
        thr = 0.0
    if thr > 1.0:
        thr = 1.0
    return thr

def evaluar_alerta_ia_y_disparar(bot: str, prob_ia: float, meta: dict | None = None, dentro_gatewin: bool = True):
    try:
        p = _norm_prob(prob_ia)
        thr = _umbral_alerta_ia(meta or {})

        now = time.time()

        # Rearme: si cae por debajo del umbral - hyst, se permite volver a disparar luego
        if p <= (thr - float(AUDIO_IA53_RESET_HYST)):
            IA53_TRIGGERED[bot] = False
            # limpiamos flag de señal pendiente si ya no califica
            estado_bots[bot]["ia_senal_pendiente"] = False
            estado_bots[bot]["ia_prob_senal"] = None
            return

        # Disparo: solo si cruza y no está ya disparado + respeta cooldown
        if (p >= thr) and (not IA53_TRIGGERED[bot]) and ((now - IA53_LAST_TS[bot]) >= float(AUDIO_IA53_COOLDOWN_S)):
            reproducir_evento("ia_53", es_demo=False, dentro_gatewin=dentro_gatewin)
            IA53_TRIGGERED[bot] = True
            IA53_LAST_TS[bot] = now

            # Flags para tu UI/decisión
            estado_bots[bot]["ia_senal_pendiente"] = True
            estado_bots[bot]["ia_prob_senal"] = float(p)

            # Evento claro (sin "53% suerte")
            agregar_evento(f"🔔 IA: {bot} {p*100:.0f}% >= {thr*100:.0f}% | ✅ ES HORA DE INVERTIR")
    except Exception:
        pass
# =========================================================
# UMBRAL VISUAL (HUD) — 70% = VERDE SIEMPRE
# No depende de AUC/reliable/n_samples (eso solo bloquea "operar", no pintar).
# Evita fallos por redondeo: 0.699999 -> lo tratamos como 0.70.
# =========================================================
def _thr_visual_verde() -> float:
    try:
        return float(IA_VERDE_THR)
    except Exception:
        return 0.75

def _thr_visual_amarillo() -> float:
    # Amarillo: zona previa (por defecto 65% si verde es 70%)
    try:
        return max(0.0, float(_thr_visual_verde()) - 0.05)
    except Exception:
        return 0.75

# =========================================================
# NORMALIZADOR ÚNICO DE PROBABILIDAD
# Acepta: 0.53, "0.53", 53, "53", 53.0
# Devuelve SIEMPRE en rango [0..1]
# =========================================================
def _norm_prob(p) -> float:
    try:
        if p is None:
            return 0.0
        if isinstance(p, str):
            p = p.strip().replace("%", "")
            if p == "" or p.lower() == "nan":
                return 0.0
        v = float(p)
        if not math.isfinite(v):
            return 0.0

        # Si viene en "porcentaje" (ej 53), convertirlo a 0.53
        if v > 1.0:
            if v <= 100.0:
                v = v / 100.0
            else:
                v = 1.0

        if v < 0.0:
            v = 0.0
        if v > 1.0:
            v = 1.0
        return v
    except Exception:
        return 0.0

def color_prob_ia(prob: float) -> str:
    """
    Colores SOLO para HUD:
      - VERDE si prob >= 0.70
      - AMARILLO si prob >= 0.65
      - ROJO si menor
    Blindado: si llega 53, lo convierte a 0.53.
    """
    p = _norm_prob(prob)

    # EPS anti-borde por flotantes (0.6999999)
    EPS = 1e-9
    tv = _thr_visual_verde()
    ty = _thr_visual_amarillo()

    if p + EPS >= tv:
        return Fore.GREEN
    if p + EPS >= ty:
        return Fore.YELLOW
    return Fore.RED


def icono_prob_ia(prob: float) -> str:
    """Icono SOLO para HUD (no altera lógica). Blindado a 53 vs 0.53."""
    p = _norm_prob(prob)
    EPS = 1e-9
    if p + EPS >= _thr_visual_verde():
        return "🟢"
    if p + EPS >= _thr_visual_amarillo():
        return "🟡"
    return "🔴"

# --- Nueva: anexar_incremental_desde_bot (completa 3 features + anti-duplicados) ---
def anexar_incremental_desde_bot(bot: str):
    """
    Anexa 1 fila al dataset incremental usando:
    - label (GANANCIA/PÉRDIDA) del último trade CERRADO
    - features del PRE_TRADE emparejado por epoch (evita leakage)
    Con lock cross-proceso y anti-duplicados por firma.
    """
    try:
        fila_dict, label = leer_ultima_fila_con_resultado(bot)
        if fila_dict is None or label is None:
            return

        try:
            label = int(label)
        except Exception:
            return
        if label not in (0, 1):
            return

        feature_names = list(INCREMENTAL_FEATURES_V2)
        cols = feature_names + ["result_bin"]
        ruta_inc = "dataset_incremental.csv"

        # Construir row completo + features derivadas (volatilidad/hora_bucket)
        row_dict_full = dict(fila_dict)
        row_dict_full["result_bin"] = label
        try:
            row_dict_full["volatilidad"] = float(calcular_volatilidad_simple(row_dict_full))
        except Exception:
            row_dict_full["volatilidad"] = 0.0
        try:
            row_dict_full["hora_bucket"] = float(calcular_hora_bucket(row_dict_full))
        except Exception:
            row_dict_full["hora_bucket"] = 0.0

        # Clip + validar
        row_dict_full = clip_feature_values(row_dict_full, feature_names)
        ok, reason = validar_fila_incremental(row_dict_full, feature_names)
        if not ok:
            agregar_evento(f"⚠️ Incremental: fila descartada ({bot}) => {reason}")
            return

        # Firma anti-dup
        row_vals_sig = []
        for k in feature_names:
            v = row_dict_full.get(k, 0.0)
            try:
                v = float(v)
            except Exception:
                v = 0.0
            row_vals_sig.append(str(round(v, 6)))
        sig = "|".join(row_vals_sig) + "|" + str(label)

        last_sig = _load_last_sig(bot)
        if last_sig == sig:
            return

        try:
            huellas_usadas.setdefault(bot, set())
            if sig in huellas_usadas[bot]:
                return
        except Exception:
            pass

        max_retries = 8
        for attempt in range(max_retries):
            try:
                # LOCK ÚNICO: mismo que maybe_retrain/backfill
                with file_lock("inc.lock"):
                    # Repara incremental mutante antes de escribir
                    try:
                        repaired = reparar_dataset_incremental_mutante(ruta=ruta_inc, cols=cols)
                        if repaired:
                            agregar_evento("🧹 IA: dataset_incremental reparado (mutante).")
                    except Exception:
                        pass

                    need_header = (not os.path.exists(ruta_inc)) or (os.path.getsize(ruta_inc) == 0)

                    with open(ruta_inc, "a", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
                        if need_header:
                            writer.writeheader()
                        writer.writerow(row_dict_full)
                        f.flush()
                        os.fsync(f.fileno())

                # Marcar firma solo si escribimos OK
                try:
                    huellas_usadas.setdefault(bot, set()).add(sig)
                except Exception:
                    pass
                _save_last_sig(bot, sig)

                # Log con throttle (1/min por bot) para que lo VEAS sin spamear
                try:
                    d = globals().setdefault("_INC_LAST_LOG_TS", {})
                    now = time.time()
                    if now - float(d.get(bot, 0.0)) >= 60.0:
                        d[bot] = now
                        agregar_evento(f"✅ Incremental: +1 fila desde {bot} ({'G' if label == 1 else 'P'}).")
                except Exception:
                    pass

                return

            except PermissionError:
                time.sleep(0.15 + 0.10 * attempt)
                continue
            except Exception as e:
                agregar_evento(f"⚠️ Incremental: excepción anexar ({bot}) => {type(e).__name__}: {e}")
                return

    except Exception as e:
        agregar_evento(f"⚠️ Incremental: excepción outer ({bot}) => {type(e).__name__}: {e}")
        return

# --- Nueva: maybe_retrain (con validaciones) ---
def maybe_retrain(force: bool = False):
    """
    Reentreno IA HONESTO (sin fuga temporal) + uso REAL de TimeSeriesSplit.

    - Split temporal: TRAIN_BASE (pasado) / CALIB (más reciente) / TEST (último)
    - StandardScaler FIT SOLO en TRAIN_BASE
    - Calibración SIN reentrenar base (sigmoid/isotonic) usando ModeloXGBCalibrado
    - TimeSeriesSplit sobre TRAIN_BASE para CV AUC (diagnóstico, no toca el split final)
    - Guardado atómico: modelo_xgb.pkl, scaler.pkl, feature_names.pkl, model_meta.json
    """
    global last_retrain_count, last_retrain_ts, _ORACLE_CACHE

    # 0) XGBoost disponible
    if not _XGBOOST_OK or xgb is None:
        try:
            agregar_evento("⚠️ IA: xgboost no disponible. No se reentrena.")
        except Exception:
            pass
        return False

    # 1) Anti re-entrada
    if not _entrenando_lock.acquire(blocking=False):
        return False

    try:
        now = time.time()

        # 2) Gatillos por filas/tiempo
        filas = contar_filas_incremental()

        if not force:
            new_rows = max(0, int(filas) - int(last_retrain_count or 0))
            mins = (now - float(last_retrain_ts or 0.0)) / 60.0

            if new_rows >= int(RETRAIN_INTERVAL_ROWS):
                pass
            else:
                if mins >= float(RETRAIN_INTERVAL_MIN) and new_rows >= int(MIN_NEW_ROWS_FOR_TIME):
                    pass
                else:
                    return False

        # 3) Reparar incremental si quedó “mutante” + leer incremental (con LOCK)
        ruta_inc = "dataset_incremental.csv"
        try:
            with file_lock("inc.lock"):
                try:
                    reparar_dataset_incremental_mutante(
                        ruta=ruta_inc,
                        cols=_canonical_incremental_cols(INCREMENTAL_FEATURES_V2)
                    )
                except Exception:
                    pass

                df = None
                for enc in ("utf-8", "utf-8-sig", "latin-1", "windows-1252"):
                    try:
                        df = pd.read_csv(ruta_inc, encoding=enc, engine="python", on_bad_lines="skip")
                        break
                    except Exception:
                        continue
        except Exception:
            df = None

        if df is None or df.empty:
            return False

        # 4) Construir X/y robusto (usa tus builders)
        feats_pref = None
        try:
            feats_pref = list(FEATURES) if FEATURES else None
        except Exception:
            feats_pref = None

        X, y, feats_used, label_col = _build_Xy_incremental(df, feature_names=feats_pref)
        if X is None or y is None or feats_used is None:
            return False

        # 5) Recorte a MAX_DATASET_ROWS (manteniendo orden temporal)
        try:
            if int(MAX_DATASET_ROWS) > 0 and len(X) > int(MAX_DATASET_ROWS):
                X = X.iloc[-int(MAX_DATASET_ROWS):].copy()
                y = np.asarray(y)[-int(MAX_DATASET_ROWS):]
        except Exception:
            pass

        n_total = len(X)
        if n_total < int(MIN_FIT_ROWS_LOW):
            try:
                agregar_evento(f"⚠️ IA: muy poca data ({n_total}). Mínimo={MIN_FIT_ROWS_LOW}.")
            except Exception:
                pass
            return False

        # 6) Corte duro: una sola clase -> no entrenar
        try:
            pos = int(np.sum(np.asarray(y) == 1))
            neg = int(np.sum(np.asarray(y) == 0))
            if pos == 0 or neg == 0:
                try:
                    agregar_evento(f"⚠️ IA: solo una clase (pos={pos}, neg={neg}). Skip.")
                except Exception:
                    pass
                return False
        except Exception:
            pass

        # 7) Balance de clases (evita entrenar con 99% de una clase)
        try:
            pos = int(np.sum(y == 1))
            neg = int(np.sum(y == 0))
            if (pos + neg) > 0:
                frac = max(pos, neg) / float(pos + neg)
                if frac >= float(MAX_CLASS_IMBALANCE) and n_total >= int(MIN_FIT_ROWS_PROD):
                    try:
                        agregar_evento(f"⚠️ IA: clase desbalanceada (pos={pos}, neg={neg}). Skip.")
                    except Exception:
                        pass
                    return False
        except Exception:
            pass

        # 8) Split temporal TRAIN/CALIB/TEST
        def _calc_sizes(n):
            n_test = int(max(MIN_TEST_ROWS, int(round(n * float(TEST_SIZE_FRAC)))))
            n_cal  = int(max(MIN_CALIB_ROWS, int(round(n * float(CALIB_SIZE_FRAC)))))

            if n < (MIN_TEST_ROWS + MIN_CALIB_ROWS + MIN_FIT_ROWS_LOW):
                n_test = max(5, int(round(n * 0.20)))
                n_cal  = max(0, int(round(n * 0.15)))

            n_train = n - n_cal - n_test
            if n_train < int(MIN_FIT_ROWS_LOW):
                falta = int(MIN_FIT_ROWS_LOW) - n_train
                if n_cal > 0:
                    cut = min(n_cal, falta)
                    n_cal -= cut
                    falta -= cut
                if falta > 0 and n_test > 5:
                    cut = min(n_test - 5, falta)
                    n_test -= cut
                    falta -= cut
                n_train = n - n_cal - n_test

            if n_train < int(MIN_FIT_ROWS_LOW):
                return None

            return n_train, n_cal, n_test

        sizes = _calc_sizes(n_total)
        if sizes is None:
            return False
        n_train, n_cal, n_test = sizes

        i0 = 0
        i1 = n_train
        i2 = n_train + n_cal
        i3 = n_total

        X_train = X.iloc[i0:i1].copy()
        y_train = np.asarray(y)[i0:i1]

        X_calib = X.iloc[i1:i2].copy() if n_cal > 0 else None
        y_calib = np.asarray(y)[i1:i2] if n_cal > 0 else None

        X_test  = X.iloc[i2:i3].copy()
        y_test  = np.asarray(y)[i2:i3]

        # 9) Escalado SOLO con TRAIN_BASE
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(X_train)
        Xte_s = scaler.transform(X_test)
        Xcal_s = scaler.transform(X_calib) if X_calib is not None and len(X_calib) > 0 else None

        # 10) Entrenar modelo base
        modelo_base = xgb.XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=4,
            eval_metric="logloss",
        )
        modelo_base.fit(Xtr_s, y_train)

        # 11) Calibración (si hay calib y hay 2 clases)
        modelo_final = modelo_base
        calib_kind = "none"

        if Xcal_s is not None and y_calib is not None and len(y_calib) >= 10:
            try:
                if len(np.unique(y_calib)) == 2:
                    p_cal = modelo_base.predict_proba(Xcal_s)[:, 1]
                    p_cal = np.clip(np.asarray(p_cal, dtype=float), 1e-6, 1.0 - 1e-6)

                    if len(y_calib) >= 200:
                        calib_kind = "isotonic"
                        iso = IsotonicRegression(out_of_bounds="clip")
                        iso.fit(p_cal, y_calib)
                        modelo_final = ModeloXGBCalibrado(modelo_base, "isotonic", iso)
                    else:
                        calib_kind = "sigmoid"
                        z = np.log(p_cal / (1.0 - p_cal)).reshape(-1, 1)
                        lr = LogisticRegression(max_iter=200)
                        lr.fit(z, y_calib)
                        modelo_final = ModeloXGBCalibrado(modelo_base, "sigmoid", lr)
            except Exception:
                calib_kind = "none"
                modelo_final = modelo_base

        # 12) Threshold sugerido: optimiza F1 sobre CALIB si existe, si no 0.5
        thr = float(THR_DEFAULT)
        try:
            if Xcal_s is not None and y_calib is not None and len(y_calib) >= 20 and len(np.unique(y_calib)) == 2:
                p = modelo_final.predict_proba(Xcal_s)[:, 1]
                best_thr, best_f1 = thr, -1.0
                for t in np.linspace(0.10, 0.90, 81):
                    yp = (p >= t).astype(int)
                    f1v = f1_score(y_calib, yp, zero_division=0)
                    if f1v > best_f1:
                        best_f1 = f1v
                        best_thr = float(t)
                thr = float(best_thr)
        except Exception:
            thr = float(THR_DEFAULT)

        # 13) Métricas en TEST (último bloque, honesto)
        try:
            p_test = modelo_final.predict_proba(Xte_s)[:, 1]
            p_test = np.clip(np.asarray(p_test, dtype=float), 1e-6, 1.0 - 1e-6)
        except Exception:
            p_test = None

        auc = 0.0
        f1t = 0.0
        brier = 1.0
        try:
            if p_test is not None and len(np.unique(y_test)) == 2:
                auc = float(roc_auc_score(y_test, p_test))
            else:
                auc = 0.0
        except Exception:
            auc = 0.0

        try:
            if p_test is not None:
                yhat = (p_test >= float(thr)).astype(int)
                f1t = float(f1_score(y_test, yhat, zero_division=0))
                brier = float(brier_score_loss(y_test, p_test))
        except Exception:
            pass

        # 14) TimeSeriesSplit REAL (CV AUC) sobre TRAIN_BASE (diagnóstico)
        cv_auc = None
        try:
            if len(X_train) >= 200 and len(np.unique(y_train)) == 2:
                tscv = TimeSeriesSplit(n_splits=4)
                aucs = []

                cv_params = dict(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_lambda=1.0,
                    random_state=42,
                    n_jobs=4,
                    eval_metric="logloss",
                )

                for tr_idx, va_idx in tscv.split(X_train):
                    Xtr = X_train.iloc[tr_idx]
                    ytr = y_train[tr_idx]
                    Xva = X_train.iloc[va_idx]
                    yva = y_train[va_idx]

                    if len(np.unique(ytr)) < 2 or len(np.unique(yva)) < 2:
                        continue

                    sc = StandardScaler()
                    Xtr_s2 = sc.fit_transform(Xtr)
                    Xva_s2 = sc.transform(Xva)

                    m = xgb.XGBClassifier(**cv_params)
                    m.fit(Xtr_s2, ytr)
                    pp = m.predict_proba(Xva_s2)[:, 1]
                    aucs.append(float(roc_auc_score(yva, pp)))

                if aucs:
                    cv_auc = float(np.mean(aucs))
        except Exception:
            cv_auc = None

        # 15) Reliable (criterio “producción”)
        try:
            pos_all = int(np.sum(y == 1))
            neg_all = int(np.sum(y == 0))
            reliable = (
                (n_total >= int(MIN_FIT_ROWS_PROD)) and
                (pos_all >= int(RELIABLE_POS_MIN)) and
                (neg_all >= int(RELIABLE_NEG_MIN)) and
                (auc >= float(MIN_AUC_CONF))
            )
        except Exception:
            reliable = False

        # 16) Anti “machacar” si baja AUC vs modelo anterior (salvo force)
        prev_auc = None
        try:
            meta_path = globals().get("_META_PATH", "model_meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta_old = json.load(f)
                if isinstance(meta_old, dict):
                    prev_auc = meta_old.get("auc", None)
                    prev_auc = float(prev_auc) if isinstance(prev_auc, (int, float)) else None
        except Exception:
            prev_auc = None

        if (not force) and (prev_auc is not None) and (auc < (prev_auc - float(AUC_DROP_TOL))):
            try:
                agregar_evento(f"🛡️ IA: NO actualizo (AUC bajó {prev_auc:.3f}→{auc:.3f}).")
            except Exception:
                pass
            return False

        # 17) Guardado atómico (compatible con tu función si existe)
        meta = {
            "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "rows_total": int(n_total),
            "rows_train": int(n_train),
            "rows_calib": int(n_cal),
            "rows_test": int(n_test),
            "pos": int(np.sum(y == 1)),
            "neg": int(np.sum(y == 0)),
            "auc": float(auc),
            "f1": float(f1t),
            "brier": float(brier),
            "cv_auc": float(cv_auc) if isinstance(cv_auc, (int, float)) else None,
            "threshold": float(thr),
            "reliable": bool(reliable),
            "calibration": str(calib_kind),
            "feature_names": list(feats_used),
            "label_col": str(label_col),
        }

        ok_save = False
        try:
            if "guardar_oracle_assets_atomico" in globals() and callable(guardar_oracle_assets_atomico):
                ok_save = bool(guardar_oracle_assets_atomico(modelo_final, scaler, list(feats_used), meta))
            else:
                # fallback por paths
                model_path = globals().get("_MODEL_PATH", "modelo_xgb.pkl")
                scaler_path = globals().get("_SCALER_PATH", "scaler.pkl")
                feats_path = globals().get("_FEATURES_PATH", "feature_names.pkl")
                meta_path = globals().get("_META_PATH", "model_meta.json")

                def _joblib_dump_atomic(obj, path):
                    tmp = path + ".tmp"
                    joblib.dump(obj, tmp)
                    os.replace(tmp, path)

                def _atomic_write_local(path, text):
                    tmp = path + ".tmp"
                    with open(tmp, "w", encoding="utf-8") as f:
                        f.write(text)
                    os.replace(tmp, path)

                with file_lock("oracle_assets.lock"):
                    _joblib_dump_atomic(modelo_final, model_path)
                    _joblib_dump_atomic(scaler, scaler_path)
                    _joblib_dump_atomic(list(feats_used), feats_path)
                    _atomic_write_local(meta_path, json.dumps(meta, ensure_ascii=False, indent=2))

                ok_save = True
        except Exception as e:
            try:
                agregar_evento(f"⚠️ IA: fallo guardado artefactos: {e}")
            except Exception:
                pass
            ok_save = False

        if not ok_save:
            return False

        # 18) Refrescar cache
        try:
            _ORACLE_CACHE["model"] = modelo_final
            _ORACLE_CACHE["scaler"] = scaler
            _ORACLE_CACHE["features"] = list(feats_used)
            _ORACLE_CACHE["meta"] = dict(meta)

            def _mt(p):
                try:
                    return os.path.getmtime(p)
                except Exception:
                    return None

            model_path = globals().get("_MODEL_PATH", "modelo_xgb.pkl")
            scaler_path = globals().get("_SCALER_PATH", "scaler.pkl")
            feats_path = globals().get("_FEATURES_PATH", "feature_names.pkl")
            meta_path = globals().get("_META_PATH", "model_meta.json")

            if "mtimes" not in _ORACLE_CACHE or not isinstance(_ORACLE_CACHE["mtimes"], dict):
                _ORACLE_CACHE["mtimes"] = {}
            _ORACLE_CACHE["mtimes"][model_path] = _mt(model_path)
            _ORACLE_CACHE["mtimes"][scaler_path] = _mt(scaler_path)
            _ORACLE_CACHE["mtimes"][feats_path] = _mt(feats_path)
            _ORACLE_CACHE["mtimes"][meta_path] = _mt(meta_path)
        except Exception:
            pass

        # 19) Marcar “reentreno hecho”
        last_retrain_count = int(filas)
        last_retrain_ts = float(now)

        try:
            msg = f"✅ IA reentrenada | AUC={auc:.3f} F1={f1t:.3f} thr={thr:.2f} calib={calib_kind}"
            if cv_auc is not None:
                msg += f" | CV_AUC={cv_auc:.3f}"
            agregar_evento(msg)
        except Exception:
            pass

        return True

    finally:
        try:
            _entrenando_lock.release()
        except Exception:
            pass

# === FIN BLOQUE 10 ===

# === BLOQUE 11 — HUD Y PANEL VISUAL ===
RENDER_LOCK = threading.Lock()

def agregar_evento(texto: str):
    eventos_recentes.append(f"[{time.strftime('%H:%M:%S')}] {texto}")

def limpiar_consola():
    os.system("cls" if os.name == "nt" else "clear")

# Mostrar panel
def mostrar_panel():
    # === IA: actualizar Prob IA antes de render (NO afecta lógica de trading) ===
    try:
        actualizar_prob_ia_todos()
    except Exception:
        pass

    """
    HUD principal: muestra estado de los bots, saldos, IA y eventos recientes.
    """
    global meta_mostrada

    # Respetar ventana de limpieza (para mensajes especiales)
    if time.time() < LIMPIEZA_PANEL_HASTA:
        limpiar_consola()
        return

    # Limpiar consola con protección
    try:
        limpiar_consola()
    except Exception:
        pass

    # Margen a la izquierda para encuadrar mejor
    padding = " " * 4

    # ==========================
    # CABECERA SUPERIOR DEL HUD
    # ==========================

    # Línea de estado general
    print(padding + Fore.GREEN + "🟢 MODO OPERACIÓN ACTIVO – Escaneando…")

    # Etapa activa para depuración de flujo
    try:
        edad_etapa = max(0, int(time.time() - float(ETAPA_TS)))
        print(padding + Fore.YELLOW + f"🧭 ETAPA {ETAPA_ACTUAL}: {ETAPA_DETALLE} ({edad_etapa}s)")
    except Exception:
        pass

    # Saldo actual (archivo Deriv o saldo_real en memoria)
    try:
        valor = obtener_valor_saldo()
        if valor is None:
            valor = float(saldo_real)
        saldo_str = f"{float(valor):.2f}"
    except Exception:
        valor = None
        saldo_str = "--"

    print(padding + Fore.GREEN + f"💰 SALDO EN CUENTA REAL DERIV: {saldo_str}")

    # Saldo inicial y meta
    try:
        inicial_str = f"{float(SALDO_INICIAL):.2f}" if SALDO_INICIAL is not None else "--"
    except Exception:
        inicial_str = "--"

    try:
        meta_str = f"{float(META):.2f}" if META is not None else "--"
    except Exception:
        meta_str = "--"

    print(padding + Fore.GREEN + f"💰 SALDO INICIAL {inicial_str} 🎯 META {meta_str}")

    # Resumen rápido para que el HUD no se vea "vacío"
    try:
        bots_con_prob = 0
        bots_75 = 0
        mejor = None
        for b in BOT_NAMES:
            pb = estado_bots.get(b, {}).get("prob_ia")
            if isinstance(pb, (int, float)):
                bots_con_prob += 1
                if float(pb) >= float(AUTO_REAL_THR):
                    bots_75 += 1
                if (mejor is None) or (float(pb) > mejor[1]):
                    mejor = (b, float(pb))
        owner = REAL_OWNER_LOCK if REAL_OWNER_LOCK in BOT_NAMES else leer_token_actual()
        owner_txt = "DEMO" if owner in (None, "none") else f"REAL:{owner}"
        mejor_txt = "--" if mejor is None else f"{mejor[0]} {mejor[1]*100:.1f}%"
        print(padding + Fore.CYAN + f"📊 Prob IA visibles: {bots_con_prob}/{len(BOT_NAMES)} | ≥75%: {bots_75} | Mejor: {mejor_txt} | Token: {owner_txt}")

        if owner not in (None, "none") and mejor is not None and owner != mejor[0]:
            print(padding + Fore.YELLOW + f"⛓️ Token bloqueado en {owner}; mejor IA actual es {mejor[0]} ({mejor[1]*100:.1f}%).")
    except Exception:
        pass

    # Marcar meta_mostrada si ya se alcanzó la META y todavía no fue aceptada
    try:
        if valor is not None and META is not None and valor >= META and not META_ACEPTADA:
            meta_mostrada = True
    except Exception:
        # No tocamos meta_mostrada si hay algún problema de conversión
        pass

    # ==========================
    # TABLA PRINCIPAL DE BOTS
    # ==========================

    print(padding + Fore.CYAN + "┌────────┬────────────────────────────────────────────────────────────────────────────────┬─────────┬──────────┬──────────┬──────────┬──────────┬──────────┐")
    print(padding + Fore.CYAN + Style.BRIGHT + "│ ✨ ESTADO INTELIGENTE DE BOTS · ÚLTIMOS 40 · TOKEN · IA · RENDIMIENTO      │" + Style.RESET_ALL)
    print(padding + Fore.CYAN + "├────────┼────────────────────────────────────────────────────────────────────────────────┼─────────┬──────────┬──────────┬──────────┬──────────┬──────────┤")
    print(padding + Fore.CYAN + "│ BOT    │ Últimos 40 Resultados                                                  │ Token   │ GANANCIAS│ PÉRDIDAS │ % ÉXITO  │ Prob IA  │ Modo IA  │")
    print(padding + Fore.CYAN + "├────────┼────────────────────────────────────────────────────────────────────────────────┼─────────┬──────────┬──────────┬──────────┬──────────┬──────────┤")

    # Meta IA para colorear Prob IA
    meta = leer_model_meta()
    umbral_ia = get_umbral_dinamico(meta, ORACULO_THR_MIN)

    # Sincronía visual dura: si hay owner REAL en memoria, la tabla SIEMPRE lo refleja.
    owner_visual = REAL_OWNER_LOCK if REAL_OWNER_LOCK in BOT_NAMES else leer_token_actual()

    for bot in BOT_NAMES:
        r = estado_bots[bot]["resultados"]
        token = "REAL" if owner_visual == bot else "DEMO"
        estado_bots[bot]["token"] = token
        src = estado_bots[bot].get("fuente")

        # Token + origen
        token_text = token
        if src and str(src).strip().upper() != "MANUAL":
            token_text += f" ({src})"
        token_color = Fore.GREEN if token_text.startswith("REAL") else Fore.CYAN
        token_text = token_color + token_text + Fore.RESET

        # Últimos 40 resultados visuales
        visual = []
        for x in r[-40:]:
            if x == "GANANCIA":
                visual.append(Fore.GREEN + "✓")
            elif x == "PÉRDIDA":
                visual.append(Fore.RED + "✗")
            elif x == "INDEFINIDO":
                visual.append(Fore.YELLOW + "·")
            else:
                visual.append(Fore.LIGHTBLACK_EX + "─")
        while len(visual) < 40:
            visual.insert(0, Fore.LIGHTBLACK_EX + "─")
        col_resultados = " ".join(visual)

        # Ganancias / Pérdidas / % éxito
        g = estado_bots[bot]["ganancias"]
        p = estado_bots[bot]["perdidas"]
        ganancias = Fore.GREEN + f"{g}"
        perdidas = Fore.RED + f"{p}"
        porc = estado_bots[bot].get("porcentaje_exito")
        total = estado_bots[bot].get("tamano_muestra", 0)

        if porc is not None:
            exito = f"{porc:.1f}% (n={total})"
            exito_color = (
                Fore.YELLOW if total < 10
                else (Fore.GREEN if porc >= 50 else Fore.RED)
            )
            if total < 10:
                exito += " ⚠"
            exito = exito_color + exito + Fore.RESET
        else:
            exito = "--"

        # --- Modo IA ---
        modo_raw = estado_bots.get(bot, {}).get("modo_ia", "off")
        modo = str(modo_raw or "off").strip().lower()

        # Normalizar variantes típicas que vienen del CSV/estado
        if modo in ("0", "false", "none", "null", ""):
            modo = "off"
        # (extra) por si llega tipo "off algo" o "off-xyz"
        if modo.startswith("off"):
            modo = "off"

        # Meta IA por bot (umbrales UI, etc.) — evita NameError y mantiene tu semáforo estable
        meta = IA_META.get(bot, {}) if "IA_META" in globals() else {}

        # --- Stats IA (solo telemetría) ---
        ia_sen   = estado_bots[bot].get("ia_seniales", 0)
        ia_acc   = estado_bots[bot].get("ia_aciertos", 0)
        ia_fal   = estado_bots[bot].get("ia_fallos", 0)
        ia_ready = estado_bots[bot].get("ia_ready", False)

        # ==========================================================
        # ✅ Prob IA = probabilidad ACTUAL del modelo (ya calibrada/ajustada)
        #    - NO mezclar con IA90_stats (eso es histórico)
        #    - OFF / inválida / vieja => "--" (gris)
        # ==========================================================

        prob_hist = get_prob_ia_historica(bot)  # se conserva por si lo usas en telemetría externa
        prob_ok = False
        prob = 0.0
        prob_str = "--"

        # 1) IA OFF => "--"
        # 2) IA ON pero prob no fresca/valida => "--"
        # 3) IA ON + prob fresca => mostrar %
        try:
            if modo != "off" and ia_prob_valida(bot, max_age_s=120.0):
                p_now = estado_bots[bot].get("prob_ia", None)
                try:
                    import math
                    if p_now is not None:
                        p = float(p_now)
                        # si viene en % (ej 53), convertir a 0.53
                        if p > 1.0 and p <= 100.0:
                            p = p / 100.0
                        if math.isfinite(p) and 0.0 <= p <= 1.0:
                            prob_ok = True
                            prob = p
                            prob_str = f"{prob*100.0:.1f}%"
                except Exception:
                    prob_ok = False
                    prob = 0.0
                    prob_str = "--"
        except Exception:
            prob_ok = False
            prob = 0.0
            prob_str = "--"

        # Fallback visual: si hay última prob reciente pero no cumplió gate, mostrarla con *
        if not prob_ok:
            try:
                p_last = estado_bots[bot].get("prob_ia", None)
                ts_last = float(estado_bots[bot].get("ia_last_prob_ts", 0.0) or 0.0)
                if isinstance(p_last, (int, float)) and ts_last > 0 and (time.time() - ts_last) <= IA_PRED_TTL_S:
                    p_aux = float(p_last)
                    if p_aux > 1.0 and p_aux <= 100.0:
                        p_aux = p_aux / 100.0
                    if 0.0 <= p_aux <= 1.0:
                        prob_ok = True
                        prob = p_aux
                        prob_str = f"{p_aux*100.0:.1f}%*"
            except Exception:
                pass

        # Confianza IA (NECESARIA: se usa para colorear modo_str)
        confianza = calcular_confianza_ia(bot, meta)

        # Decoración SOLO cuando hay prob real
        if prob_ok:
            if modo == "low_data":
                prob_str += " ⚠"
            elif modo == "exp":
                prob_str += "☆"

        # Semáforo IA (UI FIJA)
        # ----------------------------------------------
        # Regla anti-confusión:
        # Si este bot es el que tiene el token REAL (trigger_real=True o token="REAL"),
        # NO mostramos Prob IA en vivo (puede cambiar mientras corre el contrato).
        # En su lugar mostramos: "-- | OFF" para evitar decisiones “en caliente”.
        try:
            st_ui = estado_bots.get(bot, {}) if isinstance(estado_bots, dict) else {}
        except Exception:
            st_ui = {}

        token_ui = str(st_ui.get("token") or "DEMO").strip().upper()
        ui_hide_ia = bool(st_ui.get("trigger_real", False)) or token_ui.startswith("REAL")

        # ✅ FIX: high_thr_ui SIEMPRE definido (aunque ui_hide_ia=True)
        try:
            _fn = globals().get("get_umbral_ia_vigente", None)
            if callable(_fn):
                high_thr_ui = float(_fn())
            else:
                high_thr_ui = float(IA_VERDE_THR)
        except Exception:
            high_thr_ui = float(IA_VERDE_THR)

        mid_thr_ui = max(0.0, high_thr_ui - 0.05)

        if ui_hide_ia:
            prob_ok = False
            prob_str = Fore.LIGHTBLACK_EX + "--" + Fore.RESET
            modo_str = Fore.LIGHTBLACK_EX + "OFF" + Fore.RESET
        else:
            if (modo != "off") and prob_ok:
                if prob >= high_thr_ui:
                    prob_color = Fore.GREEN
                elif prob >= mid_thr_ui:
                    prob_color = Fore.YELLOW
                else:
                    prob_color = Fore.RED
            else:
                prob_color = Fore.LIGHTBLACK_EX

            prob_str = prob_color + prob_str + Fore.RESET

            modo_str = (modo.upper() if modo != "off" else "OFF")

            if modo != "off":
                if confianza >= 0.75:
                    modo_color = Fore.GREEN
                elif confianza >= 0.55:
                    modo_color = Fore.YELLOW
                else:
                    modo_color = Fore.LIGHTBLACK_EX
            else:
                modo_color = Fore.LIGHTBLACK_EX

            modo_str = modo_color + modo_str + Fore.RESET

        # --- Audio IA "es hora de invertir" (umbral fijo) ---
        audio_thr = float(globals().get("AUDIO_IA53_THR", high_thr_ui))


        token_local = (estado_bots.get(bot, {}).get("token") or "DEMO")
        es_demo_local = (str(token_local).strip().upper() == "DEMO")

        if (modo != "off") and prob_ok and (prob >= audio_thr) and not IA53_TRIGGERED[bot]:
            reproducir_evento("ia_53", es_demo=es_demo_local, dentro_gatewin=True)
            IA53_TRIGGERED[bot] = True
        elif (not prob_ok) or (modo == "off") or (prob < audio_thr):
            IA53_TRIGGERED[bot] = False


        # Línea completa del bot
        linea_bot = (
            padding + f"│ {bot:<6} │ {col_resultados:<80} │ "
            f"{token_text:<9} │ "
            f"{ganancias:<10} │ "
            f"{perdidas:<10} │ "
            f"{exito:<10} │ "
            f"{prob_str:<10} │ "
            f"{modo_str:<10} │"
        )
        print(linea_bot)

    print(padding + Fore.CYAN + "└────────┴────────────────────────────────────────────────────────────────────────────────┴─────────┴──────────┴──────────┴──────────┴──────────┴──────────┘")

    # ==========================
    # EVENTOS + TELEMETRÍA IA
    # ==========================

    # Eventos recientes
    mostrar_eventos()

    # Telemetría IA (modelo XGBoost)
    ruta_inc = "dataset_incremental.csv"
    dataset_rows = contar_filas_incremental()
    meta = _ORACLE_CACHE.get("meta") or {}
    try:
        # Normalizar SIEMPRE el meta en memoria
        if isinstance(meta, dict) and meta:
            meta = _normalize_model_meta(meta)
        else:
            meta = {}
        # Fallback duro: si el cache está incompleto, lee disco
        if (int(meta.get("n_samples", meta.get("n", 0)) or 0) == 0) and os.path.exists(_META_PATH):
            meta_disk = leer_model_meta() or {}
            if isinstance(meta_disk, dict) and meta_disk:
                meta = meta_disk
                _ORACLE_CACHE["meta"] = meta_disk
    except Exception:
        meta = meta if isinstance(meta, dict) else {}

    if dataset_rows == 0 and not meta:
        print(Fore.CYAN + " IA ▶ sin dataset_incremental.csv (n=0). Esperando que los bots generen datos...")
    elif dataset_rows < MIN_FIT_ROWS_LOW and not meta:
        faltan = max(0, MIN_FIT_ROWS_LOW - dataset_rows)
        print(Fore.CYAN + f" IA ▶ dataset con n={dataset_rows}, pero sin modelo entrenado aún.")
        print(Fore.CYAN + f"      Faltan {faltan} filas para el primer entrenamiento.")
    elif not meta:
        print(Fore.CYAN + f" IA ▶ dataset listo (n={dataset_rows}), pero sin modelo entrenado todavía.")
        print(Fore.CYAN + "      Se entrenará automáticamente cuando se llame al oráculo / maybe_retrain().")
    else:
        pos = int(meta.get("pos", meta.get("n_pos", 0)) or 0)
        neg = int(meta.get("neg", meta.get("n_neg", 0)) or 0)
        n   = int(meta.get("n_samples", meta.get("n", 0)) or 0)

        # Fallback final: si no hay n, usa pos+neg
        if n == 0 and (pos + neg) > 0:
            n = pos + neg

        auc = float(meta.get("auc", 0.0) or 0.0)
        thr = float(meta.get("threshold", ORACULO_THR_MIN))
        reliable = bool(meta.get("reliable", False))

        modo_txt = "CONFIABLE ✅" if (reliable and n >= MIN_FIT_ROWS_PROD) else "EXPERIMENTAL ⚠"

        print(Fore.CYAN + f" IA ▶ modelo XGBoost entrenado: n={n} (GAN={pos}, PERD={neg})")
        print(Fore.CYAN + f"      AUC={auc:.3f}  | Thr={thr:.2f}  | Modo={modo_txt}")

    # Mostrar contadores de aciertos IA por bot
    print(Fore.CYAN + " IA ACIERTOS POR BOT:")
    for bot in BOT_NAMES:
        sig = estado_bots[bot]["ia_seniales"]
        if sig > 0:
            ac = estado_bots[bot]["ia_aciertos"]
            fa = estado_bots[bot]["ia_fallos"]
            pct = (ac / sig * 100) if sig > 0 else 0
            print(Fore.CYAN + f"   {bot}: {ac}/{sig} ({pct:.1f}%)")

    # Contadores IA ≥70% por bot
        # HISTÓRICO: señales IA (>=70%) que llegaron a ejecutarse y cerraron con resultado
    print(Fore.YELLOW + " IA HISTÓRICO (señales cerradas, ≥70%):")
    has_hist = False
    for bot in BOT_NAMES:
        stats = IA90_stats.get(bot)
        if stats and stats.get("n", 0) > 0:
            has_hist = True
            print(Fore.YELLOW + f"   {bot}: {stats['ok']}/{stats['n']} ({stats['pct']:.1f}%)")
    if not has_hist:
        print(Fore.YELLOW + "   (Aún no hay operaciones cerradas con señal IA ≥70%.)")

    # ACTUAL: quién está >=70% ahora mismo (tick actual)
    print(Fore.YELLOW + f"\nIA SEÑALES ACTUALES (≥{IA_METRIC_THRESHOLD*100:.0f}% ahora):")
    now = []
    for bot in BOT_NAMES:
        st = estado_bots.get(bot, {}) if isinstance(estado_bots, dict) else {}

        # Anti-confusión: si este bot tiene token REAL, no lo consideramos “señal actual”
        token_ui = str(st.get("token") or "DEMO").strip().upper()
        if bool(st.get("trigger_real", False)) or token_ui.startswith("REAL"):
            continue

        modo = (st.get("modo_ia") or "off").lower()
        p = st.get("prob_ia", None)
        if modo != "off" and isinstance(p, (int, float)) and p >= float(IA_METRIC_THRESHOLD):
            now.append((bot, float(p)))

    if not now:
        print(Fore.YELLOW + f"(Ningún bot ≥{IA_METRIC_THRESHOLD*100:.0f}% en este tick.)")
    else:
        for b, p in sorted(now, key=lambda x: x[1], reverse=True):
            print(Fore.YELLOW + f"  {b}: {p*100:.1f}%")

    # ==========================================================
    # Mini-panel: Prob IA “Ficticia” vs “Real” (calibración observada)
    # Mide inflación = pred_mean - win_rate (en puntos porcentuales)
    # Usa IA_SIGNALS_LOG (señales cerradas). No toca trading.
    # ==========================================================
    try:
        global _IA_CALIB_CACHE
        if "_IA_CALIB_CACHE" not in globals():
            _IA_CALIB_CACHE = {"ts": 0.0, "rep": None, "rep_goal": None}

        if (time.time() - float(_IA_CALIB_CACHE.get("ts", 0.0) or 0.0)) >= 15.0:
            _IA_CALIB_CACHE["rep"] = auditar_calibracion_seniales_reales(min_prob=float(IA_CALIB_THRESHOLD))
            _IA_CALIB_CACHE["rep_goal"] = auditar_calibracion_seniales_reales(min_prob=float(IA_CALIB_GOAL_THRESHOLD))
            _IA_CALIB_CACHE["ts"] = float(time.time())

        rep = _IA_CALIB_CACHE.get("rep", None) or {}
        rep_goal = _IA_CALIB_CACHE.get("rep_goal", None) or {}
        n = int(rep.get("n", 0) or 0)
        n_total_closed = int(rep.get("n_total_closed", 0) or 0)
        min_prob_cal = float(rep.get("min_prob", IA_CALIB_THRESHOLD) or IA_CALIB_THRESHOLD)

        print(Fore.MAGENTA + f"\n✅ Prob IA REAL vs Prob IA FICTICIA (señales cerradas, ≥{IA_CALIB_THRESHOLD*100:.0f}%):")
        print(Fore.MAGENTA + f"   Alcance: {n} de {n_total_closed} cierres (solo señales con Prob IA ≥{min_prob_cal*100:.0f}% de todos los bots).")
        n_goal = int(rep_goal.get("n", 0) or 0)
        wr_goal = rep_goal.get("win_rate", None)
        if n_goal > 0 and isinstance(wr_goal, (int, float)):
            print(Fore.MAGENTA + f"   Meta 70%: n={n_goal} cierres con Prob IA ≥{IA_CALIB_GOAL_THRESHOLD*100:.0f}% | Real={float(wr_goal)*100:.1f}%")
        else:
            print(Fore.MAGENTA + f"   Meta 70%: aún sin cierres suficientes con Prob IA ≥{IA_CALIB_GOAL_THRESHOLD*100:.0f}%.")
        if n <= 0:
            print(Fore.MAGENTA + "   (Aún no hay cierres suficientes para medir calibración.)")
        else:
            pred_mean = float(rep.get("avg_pred", 0.0) or 0.0)
            win_rate = float(rep.get("win_rate", 0.0) or 0.0)
            infl_pp = float(rep.get("inflacion_pp", 0.0) or 0.0)
            factor = float(rep.get("factor", 1.0) or 1.0)
            min_reco = int(rep.get("min_recommended_n", IA_CALIB_MIN_CLOSED) or IA_CALIB_MIN_CLOSED)

            print(
                Fore.MAGENTA
                + f"   n={n} | PredMedia={pred_mean*100:.1f}% | Real={win_rate*100:.1f}% | Inflación={infl_pp:+.1f}pp | Factor≈{factor:.3f}"
            )

            sem_emoji, sem_label, sem_det = semaforo_calibracion(n, infl_pp)
            print(Fore.MAGENTA + f"   Semáforo calibración: {sem_emoji} {sem_label} ({sem_det})")
            print(
                Fore.MAGENTA
                + "   "
                + diagnostico_calibracion(n=n, pred_mean=pred_mean, win_rate=win_rate, infl_pp=infl_pp)
            )

            if n < int(MIN_IA_SENIALES_CONF):
                print(
                    Fore.MAGENTA
                    + f"   ⚠ muestra baja (n<{MIN_IA_SENIALES_CONF}). Úsalo como referencia, no como decisión final."
                )
            elif n < min_reco:
                print(
                    Fore.MAGENTA
                    + f"   ⚠ muestra aún en formación (recomendado n≥{min_reco} para estabilidad)."
                )

            por_bot = rep.get("por_bot", {}) if isinstance(rep.get("por_bot", {}), dict) else {}
            # Mostrar por bot (solo si hay datos)
            for bn in BOT_NAMES:
                sb = por_bot.get(bn, None)
                if isinstance(sb, dict) and int(sb.get("n", 0) or 0) >= int(MIN_IA_SENIALES_CONF):
                    pm = float(sb.get("avg_pred", 0.0) or 0.0)
                    wr = float(sb.get("win_rate", 0.0) or 0.0)
                    ip = float(sb.get("inflacion_pp", 0.0) or 0.0)
                    print(Fore.MAGENTA + f"   - {bn}: n={int(sb.get('n',0) or 0)} | Pred={pm*100:.1f}% | Real={wr*100:.1f}% | Infl={ip:+.1f}pp")
    except Exception:
        # No rompemos el HUD por auditoría
        try:
            print(Fore.MAGENTA + "\n✅ Prob IA REAL vs Prob IA FICTICIA: (error leyendo auditoría)")
        except Exception:
            pass    

    panel_lines = [
        "┌────────────────────────────────────────────┐",
        "│ 🎮 PANEL DE CONTROL TECLADO               │",
        "├────────────────────────────────────────────┤",
        "│ [S] Salir  [P] Pausar  [C] Continuar      │",
        "│ [R] Reiniciar ciclo  [T] Ver token        │",
        "│ [L] Limpiar visual  [D] Limpieza dura     │",
        "│ [G] Probar audio  [E] Entrenar IA ya      │",
        "├────────────────────────────────────────────┤",
        "│ 🤖 ¿CÓMO INVIERTES?                        │",
        "│ [5–0] Elige bot (p.ej. 7 = fulll47)       │",
        "│ [1–5] Elige ciclo [p.ej. 3 = Marti #3)    │",
    ]

    token_file = leer_token_actual()
    token_hud  = "DEMO" if (token_file in (None, "none")) else f"REAL:{token_file}"
    activo_real = REAL_OWNER_LOCK if REAL_OWNER_LOCK in BOT_NAMES else next((b for b in BOT_NAMES if estado_bots[b]["token"] == "REAL"), None)
    fuente = estado_bots.get(activo_real, {}).get("fuente") or "AUTO" if activo_real else "--"
    panel_lines.append(f"│ Fuente={fuente} → Token={token_hud:<12}          │")

    panel_lines.append("└────────────────────────────────────────────┘")

    if PENDIENTE_FORZAR_BOT:
        rest = 0
        if PENDIENTE_FORZAR_EXPIRA:
            rest = max(0, int(PENDIENTE_FORZAR_EXPIRA - time.time()))
        panel_lines += [
            "┌────────────────────────────────────────────┐",
            f"│ Bot seleccionado: {PENDIENTE_FORZAR_BOT:<22}│",
            f"│ Tiempo para decidir: {rest:>3}s               │",
            f"│ Elige ciclo [1..{MAX_CICLOS}] o ESC          │",
            "└────────────────────────────────────────────┘",
        ]


    if HUD_VISIBLE:
        dibujar_hud_gatewin(len(panel_lines), HUD_LAYOUT)
    def _strip_ansi(s: str) -> str:
        return re.sub(r'\x1b\[[0-9;]*m', '', s)
    panel_width = max(len(_strip_ansi(l)) for l in panel_lines)
    panel_height = len(panel_lines)
    try:
        term_cols, term_rows = os.get_terminal_size()
    except:
        term_cols, term_rows = 140, 50
    start_col = max(1, term_cols - panel_width - 1)
    start_row = max(1, term_rows - panel_height - 1)
    for i, line in enumerate(panel_lines):
        print(f"\x1b[{start_row + i};{start_col}H" + Fore.MAGENTA + line + Fore.RESET)
    print(f"\x1b[{term_rows};1H", end="")

# Mostrar advertencia meta
def mostrar_advertencia_meta():
    global salir, pausado, MODAL_ACTIVO, META_ACEPTADA, meta_mostrada
    pausado = True
    MODAL_ACTIVO = True
    limpiar_consola()
    try:
        terminal_width = max(os.get_terminal_size().columns, 100)
    except:
        terminal_width = 100
    ancho = terminal_width
    print("\n" * 3)
    print(Fore.YELLOW + "█" * ancho)
    print("🎉 ¡¡¡FELICIDADES!!! 🎉".center(ancho))
    print("✅ Has alcanzado tu meta diaria del +15% de ganancia.".center(ancho))
    print("")
    print("💡 Recomendación:".center(ancho))
    print("EvaBot está diseñada para buscar una ganancia diaria aproximada del 15% de tu capital.".center(ancho))
    print("Puedes seguir invirtiendo hoy bajo su responsabilidad o esperar hasta mañana para intentar nuevamente ese 15%".center(ancho))
    print("en condiciones potencialmente más favorables.".center(ancho))
    print("")
    print("⚠️ Importante:".center(ancho))
    print("EvaBot está diseñada para el análisis de miles de operaciones reales y tiene alta tasa de acierto,".center(ancho))
    print("pero ningún sistema es infalible y siempre existe riesgo de pérdida.".center(ancho))
    print("Lee el Manual de Usuario para horarios recomendados y detalles clave del programa.".center(ancho))
    print("")
    print("Presiona [S] para SALIR y asegurar beneficios, o [C] para continuar invirtiendo.".center(ancho))
    print("█" * ancho)
    print("\n" * 3)

    if AUDIO_AVAILABLE:
        if pygame.mixer.get_init() and "meta_15" in SOUND_CACHE:
            try:
                sound = SOUND_CACHE["meta_15"]
                sound.play(loops=-1)
            except Exception:
                pass
        elif winsound:
            try:
                base_dir = os.path.dirname(__file__)
                sound_path = os.path.join(base_dir, "meta15.wav")
                winsound.PlaySound(sound_path, winsound.SND_LOOP | winsound.SND_ASYNC)
            except Exception:
                pass

    while True:
        if HAVE_MSVCRT and msvcrt.kbhit():
            try:
                t = msvcrt.getch()
                if t in (b'\x00', b'\xe0'):
                    msvcrt.getch()
                    continue
                tecla = t.decode("utf-8", errors="ignore").lower()
            except:
                continue
            if tecla in ("s"):
                print("🛑 Cerrando EvaBot...")
                if AUDIO_AVAILABLE:
                    if pygame.mixer.get_init() and "meta_15" in SOUND_CACHE:
                        try:
                            SOUND_CACHE["meta_15"].stop()
                        except:
                            pass
                    elif winsound:
                        try:
                            winsound.PlaySound(None, winsound.SND_PURGE)
                        except:
                            pass
                salir = True
                MODAL_ACTIVO = False
                break
            elif tecla in ("c", "\r"):
                print("✔️ Continuando bajo responsabilidad del usuario...")
                if AUDIO_AVAILABLE:
                    if pygame.mixer.get_init() and "meta_15" in SOUND_CACHE:
                        try:
                            SOUND_CACHE["meta_15"].stop()
                        except:
                            pass
                    elif winsound:
                        try:
                            winsound.PlaySound(None, winsound.SND_PURGE)
                        except:
                            pass
                try:
                    if MAIN_LOOP:
                        fut = asyncio.run_coroutine_threadsafe(refresh_saldo_real(forzado=True), MAIN_LOOP)
                        fut.result(timeout=15)
                    valor = obtener_valor_saldo()
                    if valor is not None:
                        SALDO_INICIAL = round(valor, 2)
                        META = round(SALDO_INICIAL * 1.15, 2)
                        inicializar_saldo_real(SALDO_INICIAL)
                except Exception as e:
                    print(f"⚠️ Error reiniciando meta: {e}")
                pausado = False
                META_ACEPTADA = True
                meta_mostrada = False
                MODAL_ACTIVO = False
                break

# Dibujar HUD
def dibujar_hud_gatewin(panel_height=8, layout=None):
    if not sys.stdout.isatty():
        return
    HUD_INNER_WIDTH = 50
    HUD_WIDTH = HUD_INNER_WIDTH + 2
    activo_real = next((b for b in BOT_NAMES if estado_bots[b]["token"] == "REAL"), None)
    try:
        term_cols, term_rows = os.get_terminal_size()
    except:
        term_cols, term_rows = 140, 50
    hud_lines = [
        "┌" + "─" * HUD_INNER_WIDTH + "┐",
        "│ ⏱️  HUD: Oráculo evaluando bots..." + " " * (HUD_INNER_WIDTH - 32) + " │",
        "├" + "─" * HUD_INNER_WIDTH + "┤",
    ]
    emoji, estado, detalle = evaluar_semaforo()
    hud_lines += [
        f"│ Estado: {emoji} {estado:<{HUD_INNER_WIDTH-10}}│",
        f"│ {detalle:<{HUD_INNER_WIDTH}}│",
        "├" + "─" * HUD_INNER_WIDTH + "┤",
        f"│ {'🤖 ¿CÓMO INVIERTES?':<{HUD_INNER_WIDTH}}│",
        f"│ {'[5–0] Elige bot (p.ej. 7 = fulll47)':<{HUD_INNER_WIDTH}}│",
        f"│ {'[1–5] Elige ciclo (p.ej. 3 = Marti #3)':<{HUD_INNER_WIDTH}}│",
    ]
    activo_real = next((b for b in BOT_NAMES if estado_bots[b]["token"] == "REAL"), None)
    if activo_real:
        cyc = estado_bots[activo_real].get("ciclo_actual", 1)
        hud_lines.insert(-1, f"│ Bot REAL: {activo_real} · Ciclo {cyc}/{MAX_CICLOS}".ljust(HUD_INNER_WIDTH) + " │")
    hud_lines.insert(-1, f"│ Martingala: {marti_ciclos_perdidos}/{MAX_CICLOS} pérdidas seguidas · Próx C{ciclo_martingala_siguiente()}".ljust(HUD_INNER_WIDTH) + " │")
    hud_lines.append("└" + "─" * HUD_INNER_WIDTH + "┘")
    layout = (layout or HUD_LAYOUT).lower()
    hud_height = len(hud_lines)
    start_col = max(1, (term_cols - HUD_WIDTH) // 2)
    start_row = max(2, term_rows - hud_height - 1)
    for i, line in enumerate(hud_lines):
        print(f"\x1b[{start_row + i};{start_col}H" + Fore.YELLOW + line + Fore.RESET)

def mostrar_eventos():
    if eventos_recentes:
        print(Fore.MAGENTA + "\nEventos recientes:")
        for ev in list(eventos_recentes)[-5:]:
            print(Fore.MAGENTA + " - " + ev)
# === FIN BLOQUE 11 ===

# === BLOQUE 12 — CONTROL MANUAL REAL Y CONDICIONES SEGURAS ===
MAIN_LOOP = None

def set_main_loop(loop):
    global MAIN_LOOP
    MAIN_LOOP = loop

# ==================== VENTANA DE DECISIÓN IA ====================
# Debe empatar con el BOT (VENTANA_DECISION_IA_S) para que el humano alcance a elegir ciclo.
VENTANA_DECISION_IA_S = 30

PENDIENTE_FORZAR_BOT = None
PENDIENTE_FORZAR_INICIO = 0.0
PENDIENTE_FORZAR_EXPIRA = 0.0

FORZAR_LOCK = threading.Lock()

def condiciones_seguras_para(bot: str) -> bool:
    # Usa el mismo umbral operativo que el HUD/audio
    thr = get_umbral_operativo()
    prob = estado_bots.get(bot, {}).get("prob_ia") or 0.0
    n    = estado_bots.get(bot, {}).get("tamano_muestra", 0)
    return (n >= ORACULO_N_MIN) and (prob >= thr)

# forzar_real_manual
def forzar_real_manual(bot: str, ciclo: int):
    if not FORZAR_LOCK.acquire(blocking=False):
        agregar_evento("🔒 Forzar REAL: ya en progreso; espera.")
        return
    try:
        ciclo = max(1, min(int(ciclo), MAX_CICLOS))

        # Añadido: Confirmación en rojo si no es seguro (para evitar cierres forzados por malas decisiones)
        CONFIRMAR_EN_ROJO = True  # Activado por defecto para seguridad
        if CONFIRMAR_EN_ROJO and HAVE_MSVCRT and not condiciones_seguras_para(bot):
            global MODAL_ACTIVO
            MODAL_ACTIVO = True
            try:
                with RENDER_LOCK:
                    print(Fore.YELLOW + f"⚠️ Semáforo no verde para {bot}. ¿Forzar de todos modos? [Y/N]")
                while True:
                    if msvcrt.kbhit():
                        k = msvcrt.getch().decode("utf-8", errors="ignore").lower()
                        if k == "y":
                            break
                        elif k == "n":
                            agregar_evento("❎ Forzar REAL cancelado (no confirmado).")
                            return
                    time.sleep(0.05)
            finally:
                MODAL_ACTIVO = False

        # Nueva lógica: Marcar como señal IA si prob >= thr_ia
        prob = estado_bots.get(bot, {}).get("prob_ia") or 0.0
        thr_ia = get_umbral_operativo()

        if prob >= thr_ia and not estado_bots[bot]["ia_senal_pendiente"]:
            estado_bots[bot]["ia_senal_pendiente"] = True
            estado_bots[bot]["ia_prob_senal"] = prob

            # ✅ FIX REAL: registrar APERTURA de señal con epoch PRE real (para contabilidad correcta)
            # Esto sí “lo consume” el cierre automático posterior (log_ia_close vía ia_audit_scan_close).
            try:
                epoch_sig = None
                try:
                    epoch_sig = ia_audit_get_last_pre_epoch(bot)
                except Exception:
                    epoch_sig = None

                if epoch_sig is not None:
                    log_ia_open(
                        bot,
                        int(epoch_sig),
                        float(prob),
                        float(thr_ia),
                        "MANUAL"
                    )
            except Exception:
                pass


        if not escribir_orden_real(bot, ciclo):
            agregar_evento(f"🔒 Forzar REAL bloqueado para {bot.upper()}: ya hay otro bot en REAL.")
            return

        estado_bots[bot]["reintentar_ciclo"] = True
        estado_bots[bot]["ciclo_actual"] = ciclo
        global marti_paso
        marti_paso = ciclo - 1
        estado_bots[bot]["fuente"] = "MANUAL"

        requerido = float(MARTI_ESCALADO[ciclo - 1])
        val = obtener_valor_saldo()
        if val is None or val < requerido:
            agregar_evento(f"⚠️ Saldo < requerido para ciclo #{ciclo} en {bot} (pide {requerido}). Intentando igual.")

        # escribir_orden_real(...) ya dejó token+HUD sincronizados; evitamos doble token_sync.
        agregar_evento(f"⚡ Forzar REAL: {bot} → ciclo #{ciclo} (fuente=MANUAL)")
        with RENDER_LOCK:
            mostrar_panel()
    except Exception as e:
        agregar_evento(f"⛔ Forzar REAL falló en {bot}: {e}")
    finally:
        FORZAR_LOCK.release()

def evaluar_semaforo():
    thr = get_umbral_operativo()

    mejor = (None, None, 0)
    for b in BOT_NAMES:
        d = estado_bots.get(b, {})
        prob = d.get("prob_ia")
        n    = d.get("tamano_muestra", 0)
        if prob is not None and (mejor[0] is None or prob > mejor[0]):
            mejor = (prob, b, n)
    prob, bbest, n = mejor

    owner = REAL_OWNER_LOCK if REAL_OWNER_LOCK in BOT_NAMES else leer_token_actual()
    try: saldo_val = float(obtener_valor_saldo() or 0.0)
    except: saldo_val = 0.0
    costo = float(MARTI_ESCALADO[0])

    detalle = ""
    if owner and owner not in (None, "none"):
        detalle = f"Token en uso por {owner}. Puedes forzar 5–0 → 1..5."
        return "🟡", "AVISO", detalle
    if saldo_val < costo:
        falta = costo - saldo_val
        detalle = f"Saldo < requerido para ciclo1 ({costo:.2f}). Faltan {falta:.2f} USD para cubrir el ciclo1 de Martingala."
        return "🟡", "AVISO", detalle

    n_inc = contar_filas_incremental()
    if n_inc < MIN_FIT_ROWS_LOW:
        detalle = f"IA en modo WARMUP: n={n_inc}. Faltan {MIN_FIT_ROWS_LOW - n_inc} filas para entrenamiento estable."
        return "🟡", "EN ESPERA", detalle

    if prob is None or n < 10:
        return "🟡", "EN ESPERA", "Pocos datos útiles aún."
    if n < ORACULO_N_MIN and (prob or 0) < thr:
        return "🟡", "EN ESPERA", f"n={n}<{ORACULO_N_MIN} y prob={prob:.0%}<{int(thr*100)}%"
    if n < ORACULO_N_MIN:
        return "🟡", "EN ESPERA", f"n={n}<{ORACULO_N_MIN}"
    if (prob or 0) < thr:
        return "🟡", "EN ESPERA", f"prob={prob:.0%}<{int(thr*100)}%"

    tecla = (bbest or "?")[-2:]
    detalle = f"{bbest} • Prob={prob:.0%} • n={n} → pulsa [{tecla}] y el ciclo."
    return "🟢", "SEÑAL LISTA", detalle

# NUEVAS FUNCIONES PARA RESET
RESET_ON_START = False  # Cambiado a False para mantener historial entre sesiones

def _csv_header_bot():
    return ["fecha","activo","direction","monto","resultado","ganancia_perdida",
            "rsi_9","rsi_14","sma_5","sma_20","cruce_sma","breakout",
            "rsi_reversion","racha_actual","payout","puntaje_estrategia",
            "result_bin","payout_decimal_rounded"]

def resetear_csv_bot(nombre_bot: str):
    ruta = f"registro_enriquecido_{nombre_bot}.csv"
    try:
        with open(ruta, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(_csv_header_bot())
    except Exception as e:
        print(f"⚠️ No pude resetear {ruta}: {e}")

def resetear_incremental_y_modelos(borrar_modelos: bool = True):
    try:
        if os.path.exists("dataset_incremental.csv"):
            os.remove("dataset_incremental.csv")
    except Exception as e:
        print(f"⚠️ No pude borrar dataset_incremental.csv: {e}")

    if borrar_modelos:
        for f in ("modelo_xgb.pkl","scaler.pkl","feature_names.pkl"):
            try:
                if os.path.exists(f):
                    os.remove(f)
            except Exception as e:
                print(f"⚠️ No pude borrar {f}: {e}")

def resetear_estado_hud(estado_bots: dict):
    for bot in list(estado_bots.keys()):
        estado_bots[bot].update({
            "resultados": [], "ganancias": 0, "perdidas": 0,
            "porcentaje_exito": None, "tamano_muestra": 0,
            "prob_ia": 0.0, "token": "DEMO",
            "fuente": None, "modo_ia": "off",
            "ia_seniales": 0, "ia_aciertos": 0, "ia_fallos": 0, "ia_senal_pendiente": False,
            "ia_prob_senal": None
        })

def limpieza_dura():
    for nb in BOT_NAMES:
        resetear_csv_bot(nb)
    resetear_incremental_y_modelos(borrar_modelos=True)
    resetear_estado_hud(estado_bots)
    print("🧨 Limpieza dura ejecutada. Ok.")

# Backfill seguro
def backfill_incremental(ultimas=500):
    try:
        try:
            feature_names = joblib.load("feature_names.pkl")
            feature_names = [c for c in feature_names if c != "result_bin"]
        except Exception:
            feature_names = [
                "rsi_9","rsi_14","sma_5","sma_20","cruce_sma","breakout",
                "rsi_reversion","racha_actual","payout","puntaje_estrategia",
                "volatilidad","es_rebote","hora_bucket",
            ]
        inc = "dataset_incremental.csv"
        cols = feature_names + ["result_bin"]

        # 0) Reparar incremental si quedó "mutante" (header corrupto / columnas extra / mezcla de campos)
        with file_lock("inc.lock"):
            if reparar_dataset_incremental_mutante(inc, cols):
                agregar_evento("🧹 Incremental: esquema reparado (header/filas inconsistentes).")
            if not os.path.exists(inc) or os.stat(inc).st_size == 0:
                with open(inc, "w", newline="", encoding="utf-8") as f:
                    csv.DictWriter(f, fieldnames=cols).writeheader()

        firmas_existentes = set()
        if os.path.exists(inc):
            df_inc = pd.read_csv(inc, encoding="utf-8", on_bad_lines="skip")
            if not df_inc.empty:
                sigs = df_inc[feature_names].round(6).astype(str).agg("|".join, axis=1) + "|" + df_inc["result_bin"].astype(int).astype(str)
                firmas_existentes = set(sigs.tolist())


        for bot in BOT_NAMES:
            ruta = f"registro_enriquecido_{bot}.csv"
            if not os.path.exists(ruta):
                continue
            df = None
            for enc in ("utf-8","latin-1","windows-1252"):
                try:
                    df = pd.read_csv(ruta, encoding=enc, on_bad_lines="skip")
                    break
                except Exception as e:
                    print(f"⚠️ Error en backfill para {bot}: {e}")
                    continue
            if df is None or df.empty:
                continue

            req = [
                "rsi_9","rsi_14","sma_5","sma_20","cruce_sma","breakout",
                "rsi_reversion","racha_actual","puntaje_estrategia"
            ]
            if not set(req).issubset(df.columns) or "resultado" not in df.columns:
                continue

            for c in req + ["payout","payout_decimal_rounded"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")

            df["resultado_norm"] = df["resultado"].apply(normalizar_resultado)

            sub = df[df["resultado_norm"].isin(["GANANCIA","PÉRDIDA"])]
            sub = sub[sub[req].notna().all(axis=1)].tail(ultimas)
            if sub.empty:
                continue

            nuevas_filas = []
            descartadas = 0
            
            nuevas_filas = []
            for _, r in sub.iterrows():
                # base mínima
                fila = {k: float(r[k]) for k in req}

                # Diccionario completo para helpers enriquecidos
                row_dict_full = r.to_dict()

                                # ==========================
                # payout normalizado (ROI 0–1.5 aprox)
                # ==========================
                pay = calcular_payout_feature(row_dict_full)
                # Si falta payout, NO lo inventamos como 0.0: descartamos la fila
                # (backfill es entrenamiento, aquí ser conservador = IA más estable)
                if pay is None or pay < 0.05:
                    descartadas += 1
                    continue

                # ✅ FIX: estas asignaciones DEBEN ocurrir antes del continue
                fila["payout"] = float(pay)
                row_dict_full["payout"] = float(pay)

                # ==========================
                # puntaje_estrategia normalizado 0–1
                # ==========================
                pe = calcular_puntaje_estrategia_normalizado(row_dict_full)
                if pe is None and "puntaje_estrategia" in r:
                    pe = _norm_01(r.get("puntaje_estrategia"))
                if pe is not None:
                    fila["puntaje_estrategia"] = pe

                # ==========================
                # volatilidad: normalizada a [0,1]
                # - si viene en el CSV y es válida, la usamos
                # - si falta / NaN, la calculamos con calcular_volatilidad_simple (proxy SMA5 vs SMA20)
                # ==========================
                vol_raw = row_dict_full.get("volatilidad", None)
                try:
                    vol = float(vol_raw) if vol_raw not in (None, "") else np.nan
                except Exception:
                    vol = np.nan
                if pd.isna(vol):
                    vol = calcular_volatilidad_simple(row_dict_full)

                fila["volatilidad"] = max(0.0, min(float(vol), 1.0))

                # ==========================
                # nuevas features: rebote y hora (0–1)
                # ==========================
                fila["es_rebote"]   = calcular_es_rebote(row_dict_full)
                fila["hora_bucket"] = calcular_hora_bucket(row_dict_full)

                # ==========================
                # label final (GANANCIA / PÉRDIDA)
                # ==========================
                label = 1 if r["resultado_norm"] == "GANANCIA" else 0
                fila_dict = fila.copy()
                fila_dict["result_bin"] = label

                # Validación defensiva
                valid, reason = validar_fila_incremental(fila_dict, feature_names)
                if not valid:
                    agregar_evento(f"⚠️ Incremental: fila descartada en backfill ({reason})")
                    descartadas += 1
                    continue

                # Clipping defensivo
                fila_dict = clip_feature_values(fila_dict, feature_names)

                # Evitar duplicados vía firma
                sig = _make_sig(fila_dict)
                if sig in firmas_existentes:
                    continue
                firmas_existentes.add(sig)

                nuevas_filas.append(fila_dict)

            if nuevas_filas:
                with file_lock("inc.lock"):
                    with open(inc, "a", newline="", encoding="utf-8") as f:
                        w = csv.DictWriter(f, fieldnames=cols)
                        for rd in nuevas_filas:
                            w.writerow(rd)
                        f.flush(); os.fsync(f.fileno())
        agregar_evento("✅ IA: backfill incremental completado.")
    except Exception as e:
        agregar_evento(f"⚠️ IA: fallo en backfill: {e}")
# === FIN BLOQUE 12 ===

# === BLOQUE 13 — LOOP PRINCIPAL, WEBSOCKET Y TECLADO ===
# Orden operativo por etapas (solo trazabilidad/depuración; no altera trading)
ETAPAS_PROGRAMA = {
    "BOOT_01": "Arranque y validación de entorno",
    "BOOT_02": "Carga de audio/tokens y reset opcional",
    "BOOT_03": "Backfill + primer entrenamiento IA",
    "BOOT_04": "Sincronización inicial HUD/CSV",
    "TICK_01": "Lectura de token y carga incremental por bot",
    "TICK_02": "Watchdog REAL + detección de cierre",
    "TICK_03": "Selección IA / ventana manual / asignación REAL",
    "TICK_04": "Refresh saldo + render HUD",
    "STOP": "Salida controlada",
}
ETAPA_ACTUAL = "BOOT_01"
ETAPA_DETALLE = ETAPAS_PROGRAMA[ETAPA_ACTUAL]
ETAPA_TS = time.time()

def set_etapa(codigo, detalle_extra=None, anunciar=False):
    """
    Marca etapa actual del programa para facilitar diagnóstico en vivo.
    No modifica ninguna decisión de trading.
    """
    global ETAPA_ACTUAL, ETAPA_DETALLE, ETAPA_TS

    codigo = str(codigo or "").strip().upper()
    if codigo not in ETAPAS_PROGRAMA:
        return

    base = ETAPAS_PROGRAMA[codigo]
    detalle = f"{base} | {detalle_extra}" if detalle_extra else base

    ETAPA_ACTUAL = codigo
    ETAPA_DETALLE = detalle
    ETAPA_TS = time.time()

    if anunciar:
        agregar_evento(f"🧭 ETAPA {codigo}: {detalle}")

# Nueva constante para watchdog de REAL - Bajado para más reactividad
REAL_TIMEOUT_S = 120  # 2 minutos sin actividad para aviso/rearme (sin salir de REAL)

# Cargar datos bot
# Cargar datos bot
async def cargar_datos_bot(bot, token_actual):
    ruta = f"registro_enriquecido_{bot}.csv"
    if not os.path.exists(ruta):
        return

    try:
        snapshot = SNAPSHOT_FILAS.get(bot, 0)

        # Fuente de verdad de owner REAL para no pintar DEMO transitorio en HUD/tabla.
        effective_owner = REAL_OWNER_LOCK if REAL_OWNER_LOCK in BOT_NAMES else (token_actual if token_actual in BOT_NAMES else next((b for b in BOT_NAMES if estado_bots.get(b, {}).get("token") == "REAL"), None))

        # Sincroniza token visual SIEMPRE, incluso si no entran filas nuevas este tick.
        estado_bots[bot]["token"] = "REAL" if effective_owner == bot else "DEMO"

        # Gate rápido (opcional): si el archivo no creció, salimos sin leer todo el CSV
        actual = contar_filas_csv(bot)
        if actual <= snapshot:
            return

        df = pd.read_csv(ruta, encoding="utf-8", on_bad_lines="skip")
        if df.empty:
            # Lectura temporalmente vacía (archivo en escritura / parse falló).
            # NO resetees SNAPSHOT_FILAS porque provoca re-procesos y ruido.
            return

        # Si snapshot quedó desfasado frente al df real, lo corregimos
        if snapshot >= len(df):
            SNAPSHOT_FILAS[bot] = len(df)
            return

        nuevas = df.iloc[snapshot:]
        # IMPORTANTE: el snapshot debe seguir a df (no a contar_filas_csv),
        # porque read_csv puede saltarse líneas malas con on_bad_lines="skip".
        SNAPSHOT_FILAS[bot] = len(df)

        required_cols = [
            "rsi_9", "rsi_14", "sma_5", "sma_20", "cruce_sma", "breakout",
            "rsi_reversion", "racha_actual", "puntaje_estrategia"
        ]

        for _, row in nuevas.iterrows():
            fila_dict = row.to_dict()

            trade_status = str(fila_dict.get("trade_status", "")).strip().upper()
            resultado = normalizar_resultado(fila_dict.get("resultado", ""))

            # =========================
            # 1) FILAS NO-CERRADAS (PRE_TRADE / incompletas)
            #    - Calculamos Prob IA para el HUD
            #    - NO tocamos historial, n, ni %éxito (evita los “·” intercalados)
            # =========================
            if resultado not in ("GANANCIA", "PÉRDIDA"):
                # Guarda epoch PRE más reciente para heartbeat ACK (sin depender de filas nuevas constantes)
                try:
                    ep_pre = fila_dict.get("epoch", 0)
                    ep_pre = int(float(ep_pre)) if str(ep_pre).strip() != "" else 0
                    if ep_pre > 0:
                        estado_bots[bot]["ultimo_epoch_pretrade"] = ep_pre
                except Exception:
                    pass

                # Si el bot marcó CERRADO pero no trajo resultado válido,
                # cerramos señal pendiente (si existía) sin contaminar historial.
                if trade_status == "CERRADO":
                    if estado_bots[bot].get("ia_senal_pendiente"):
                        estado_bots[bot]["ia_senal_pendiente"] = False
                        estado_bots[bot]["ia_prob_senal"] = None

                    estado_bots[bot]["token"] = "REAL" if effective_owner == bot else "DEMO"
                    last_update_time[bot] = time.time()
                    continue

                missing = [col for col in required_cols if pd.isna(fila_dict.get(col))]
                if missing:
                    agregar_evento(f"⚠️ {bot}: PRE_TRADE incompleto, faltan {len(missing)} cols: {missing[:5]}")
                    # IMPORTANTE: si falta data, no inventamos 0.0 (queda sin predicción)
                    estado_bots[bot]["prob_ia"] = None
                    estado_bots[bot]["modo_ia"] = "off"
                    estado_bots[bot]["ia_ready"] = False

                    meta = leer_model_meta() or {}
                    escribir_ia_ack(bot, fila_dict.get("epoch"), None, "OFF", meta)

                else:
                    try:
                        prob_ia, modo_ia = oraculo_predict_visible(fila_dict)

                        # Normaliza (evita 'OFF' vs 'off' y valores fuera de rango)
                        modo_norm = str(modo_ia or "off").strip().lower()
                        prob_norm = None
                        try:
                            if prob_ia is not None:
                                p = float(prob_ia)
                                if 0.0 <= p <= 1.0:
                                    prob_norm = p
                        except Exception:
                            prob_norm = None

                        estado_bots[bot]["prob_ia"] = prob_norm
                        estado_bots[bot]["modo_ia"] = modo_norm

                        meta = leer_model_meta() or {}
                        reliable = bool(meta.get("reliable", False))
                        n_inc = contar_filas_incremental()
                        rows = int(n_inc or 0)
                        estado_bots[bot]["ia_ready"] = bool(
                            reliable and (rows >= MIN_FIT_ROWS_LOW) and (modo_norm != "off") and (prob_norm is not None)
                        )

                        escribir_ia_ack(bot, fila_dict.get("epoch"), prob_norm, modo_norm.upper(), meta)

                    except Exception as e:
                        agregar_evento(f"⚠️ {bot}: PRED_FAIL pretrade: {type(e).__name__}")
                        estado_bots[bot]["prob_ia"] = None
                        estado_bots[bot]["modo_ia"] = "off"
                        estado_bots[bot]["ia_ready"] = False

                        meta = leer_model_meta() or {}
                        escribir_ia_ack(bot, fila_dict.get("epoch"), None, "OFF", meta)



                estado_bots[bot]["token"] = "REAL" if effective_owner == bot else "DEMO"
                last_update_time[bot] = time.time()
                continue

            # =========================
            # 2) FILAS CERRADAS (GANANCIA / PÉRDIDA)
            #    - Aquí sí actualizamos historial y estadísticas reales
            # =========================
            estado_bots[bot]["ultimo_resultado"] = resultado
            estado_bots[bot]["resultados"].append(resultado)
            estado_bots[bot]["tamano_muestra"] += 1

            if resultado == "GANANCIA":
                estado_bots[bot]["ganancias"] += 1
            elif resultado == "PÉRDIDA":
                estado_bots[bot]["perdidas"] += 1

            total = estado_bots[bot]["tamano_muestra"]
            if total > 0:
                estado_bots[bot]["porcentaje_exito"] = (estado_bots[bot]["ganancias"] / total) * 100

            # Cierre especial para REAL manual: SIEMPRE 1 sola operación
            if (
                MODO_REAL_MANUAL
                and estado_bots[bot].get("fuente") == "MANUAL"
                and resultado in ("GANANCIA", "PÉRDIDA")
            ):
                reason = f"REAL manual: {resultado} → una operación y regreso a DEMO"
                cerrar_por_fin_de_ciclo(bot, reason)
                agregar_evento(f"✅ REAL MANUAL cerrado para {bot.upper()} tras {resultado}. Volviendo a DEMO.")

            # --- Contadores de IA: SOLO cuando llega un cierre real ---
            if estado_bots[bot].get("ia_senal_pendiente"):
                prob_senal = estado_bots[bot].get("ia_prob_senal")
                thr_ia = get_umbral_operativo()

                if prob_senal is not None and prob_senal >= thr_ia:
                    estado_bots[bot]["ia_seniales"] += 1
                    if resultado == "GANANCIA":
                        estado_bots[bot]["ia_aciertos"] += 1
                    elif resultado == "PÉRDIDA":
                        estado_bots[bot]["ia_fallos"] += 1

                if prob_senal is not None and prob_senal >= 0.7:
                    IA90_stats[bot]["n"] += 1
                    if resultado == "GANANCIA":
                        IA90_stats[bot]["ok"] += 1

                n_ia90 = IA90_stats[bot]["n"]
                ok_ia90 = IA90_stats[bot]["ok"]
                if n_ia90 > 0:
                    pct_suav = (ok_ia90 + 1) / (n_ia90 + 2) * 100.0
                    IA90_stats[bot]["pct"] = pct_suav
                else:
                    IA90_stats[bot]["pct"] = 50.0

                # Cerramos señal pendiente SOLO aquí (en cierre)
                estado_bots[bot]["ia_senal_pendiente"] = False
                estado_bots[bot]["ia_prob_senal"] = None

            estado_bots[bot]["token"] = "REAL" if effective_owner == bot else "DEMO"
            last_update_time[bot] = time.time()

        # Mantén tu pipeline incremental como estaba
        anexar_incremental_desde_bot(bot)

    except Exception as e:
        print(f"⚠️ Error cargando datos para {bot}: {e}")

# Obtener saldo real
async def obtener_saldo_real():
    global saldo_real, ULTIMA_ACT_SALDO
    token_demo, token_real = leer_tokens_usuario()
    if not token_real:
        return
    try:
        async with websockets.connect(DERIV_WS_URL) as ws:
            auth_msg = json.dumps({"authorize": token_real})
            await ws.send(auth_msg)
            resp = json.loads(await ws.recv())
            if "error" in resp:
                print(f"⚠️ Error en auth: {resp['error']['message']}")
                return
            bal_msg = json.dumps({"balance": 1, "subscribe": 1})
            await ws.send(bal_msg)
            resp = json.loads(await ws.recv())
            if "error" in resp:
                print(f"⚠️ Error en balance: {resp['error']['message']}")
                return
            if "balance" in resp:
                saldo_real = f"{resp['balance']['balance']:.2f}"
                ULTIMA_ACT_SALDO = time.time()
    except Exception as e:
        print(f"⚠️ Error obteniendo saldo: {e}")

async def refresh_saldo_real(forzado=False):
    global ULTIMA_ACT_SALDO
    if forzado or time.time() - ULTIMA_ACT_SALDO > REFRESCO_SALDO:
        await obtener_saldo_real()

def obtener_valor_saldo():
    global saldo_real
    try:
        return float(saldo_real)
    except:
        return None

def inicializar_saldo_real(valor):
    global SALDO_INICIAL, META
    SALDO_INICIAL = round(valor, 2)
    META = round(SALDO_INICIAL * 1.15, 2)

# Escuchar teclas
def escuchar_teclas():
    global pausado, salir, reinicio_manual, LIMPIEZA_PANEL_HASTA, HUD_VISIBLE
    global PENDIENTE_FORZAR_BOT, PENDIENTE_FORZAR_INICIO, PENDIENTE_FORZAR_EXPIRA

    bot_map = {'5': 'fulll45', '6': 'fulll46', '7': 'fulll47', '8': 'fulll48', '9': 'fulll49', '0': 'fulll50'}
    last_key_time = 0  # debounce 200 ms

    while True:
        if MODAL_ACTIVO:
            time.sleep(0.1); continue

        now = time.time()
        if HAVE_MSVCRT and msvcrt.kbhit():
            if now - last_key_time < 0.2:
                time.sleep(0.05); continue
            last_key_time = now

            try:
                k = msvcrt.getch()
                if k in (b'\x00', b'\xe0'):  
                    msvcrt.getch(); continue
                k = k.decode("utf-8", errors="ignore").lower()
            except:
                continue

            if k == "s":
                print("\n\n🔴 Saliendo del programa..."); salir = True; break
            elif k == "p":
                pausado = True; print("\n⏸️ Programa pausado. Presiona [C] para continuar.")
            elif k == "c":
                pausado = False; print("\n▶️ Programa reanudado.")
            elif k == "r":
                reinicio_manual = True; print("\n🔁 Reinicio de Martingala solicitado.")
            elif k == "t":
                tok = leer_token_actual(); print(f"\n🔍 TOKEN ACTUAL: {tok or 'none'}")
            elif k == "l":
                LIMPIEZA_PANEL_HASTA = time.time() + 15; print("\n🎹 Limpieza visual…")
            elif k == "d":
                reiniciar_completo(borrar_csv=False, limpiar_visual_segundos=15, modo_suave=True); print("\n🧽 Limpieza dura ejecutada.")
            elif k == "g":
                reproducir_evento("test", es_demo=True, dentro_gatewin=True); print("\n🎵 Test de audio…")
            elif k == "e":
                try:
                    # Cooldown anti-repetición (Windows repite tecla y entrena 2 veces)
                    if "LAST_MANUAL_RETRAIN_TS" not in globals():
                        globals()["LAST_MANUAL_RETRAIN_TS"] = 0.0
                    nowt = time.time()
                    if (nowt - float(globals()["LAST_MANUAL_RETRAIN_TS"])) < 30.0:
                        agregar_evento("🧠 Entrenamiento ignorado (cooldown 30s).")
                    else:
                        globals()["LAST_MANUAL_RETRAIN_TS"] = nowt
                        maybe_retrain(force=True)
                        print("\n🧠 Entrenamiento forzado.")
                except Exception as e:
                    print(f"\n⚠️ No se pudo entrenar: {e}")

            elif k in bot_map:
                PENDIENTE_FORZAR_BOT = bot_map[k]
                PENDIENTE_FORZAR_INICIO = time.time()
                PENDIENTE_FORZAR_EXPIRA = PENDIENTE_FORZAR_INICIO + VENTANA_DECISION_IA_S
                agregar_evento(f"🎯 Bot seleccionado: {PENDIENTE_FORZAR_BOT}. Elige ciclo [1..{MAX_CICLOS}] o ESC.")
                with RENDER_LOCK:
                    mostrar_panel()

            elif PENDIENTE_FORZAR_BOT and k.isdigit() and k in [str(i) for i in range(1, MAX_CICLOS+1)]:
                ciclo = int(k)
                bot_sel = PENDIENTE_FORZAR_BOT
                PENDIENTE_FORZAR_BOT = None
                PENDIENTE_FORZAR_INICIO = 0.0
                PENDIENTE_FORZAR_EXPIRA = 0.0
                forzar_real_manual(bot_sel, ciclo)

            elif PENDIENTE_FORZAR_BOT and k == "\x1b":  # ESC
                agregar_evento("❎ Forzar REAL cancelado.")
                PENDIENTE_FORZAR_BOT = None
                PENDIENTE_FORZAR_INICIO = 0.0
                PENDIENTE_FORZAR_EXPIRA = 0.0
                with RENDER_LOCK:
                    mostrar_panel()

        else:
            time.sleep(0.05)

if sys.stdout.isatty():
    threading.Thread(target=escuchar_teclas, daemon=True).start()

# Main - Añadida pasada inicial para sincronizar HUD con CSV existentes
async def main():
    global salir, pausado, reinicio_manual, SALDO_INICIAL
    global PENDIENTE_FORZAR_BOT, PENDIENTE_FORZAR_INICIO, PENDIENTE_FORZAR_EXPIRA, REAL_OWNER_LOCK

    try:
        set_etapa("BOOT_01", "Inicializando main()", anunciar=True)
        try:
            os.remove("real.lock")
        except:
            pass
        set_etapa("BOOT_02", "Leyendo tokens de usuario")
        tokens = leer_tokens_usuario()
        if tokens == (None, None):
            print("⚠️ Tokens ausentes. Modo sin-saldo activo (HUD/IA continúan).")
        init_audio()
        if RESET_ON_START:
            for nb in BOT_NAMES:
                resetear_csv_bot(nb)
            resetear_incremental_y_modelos(borrar_modelos=True)
            resetear_estado_hud(estado_bots)
            print("🧼 Sesión limpia: CSVs de bots, dataset incremental y estado HUD reiniciados.")
        reiniciar_completo(borrar_csv=False, limpiar_visual_segundos=15, modo_suave=True)
        loop = asyncio.get_running_loop()
        set_main_loop(loop)
        await refresh_saldo_real(forzado=True)
        valor = obtener_valor_saldo()
        if valor is not None:
            inicializar_saldo_real(valor)

        set_etapa("BOOT_03", "Backfill y primer entrenamiento")
        # Backfill IA desde los logs enriquecidos
        try:
            backfill_incremental(ultimas=1500)
        except Exception as e:
            agregar_evento(f"⚠️ IA: error en backfill inicial: {e}")

        # Intentar un primer entrenamiento, si ya hay suficientes filas
        try:
            maybe_retrain(force=True)
        except Exception as e:
            agregar_evento(f"⚠️ IA: error al intentar entrenar tras el backfill: {e}")

        set_etapa("BOOT_04", "Sincronizando HUD con CSV")
        # Pasada inicial para sincronizar HUD con CSV existentes
        token_actual_loop = "--"  # Dummy para carga inicial
        for bot in BOT_NAMES:
            await cargar_datos_bot(bot, token_actual_loop)

        while True:
            if salir:
                set_etapa("STOP", "Señal de salida detectada", anunciar=True)
                break
            if pausado:
                await asyncio.sleep(1)
                continue
            if reinicio_manual:
                reinicio_manual = False
                reiniciar_completo(borrar_csv=False, limpiar_visual_segundos=15, modo_suave=True)
                await refresh_saldo_real(forzado=True)

            try:  
                set_etapa("TICK_01")
                token_actual_loop = REAL_OWNER_LOCK if REAL_OWNER_LOCK in BOT_NAMES else (leer_token_actual() or next((b for b in BOT_NAMES if estado_bots.get(b, {}).get("token") == "REAL"), None))
                # Heartbeat: mantiene ACK alineado al HUD aunque no entren filas nuevas ese tick.
                refrescar_ia_ack_desde_hud(intervalo_s=1.0)
                owner_mem = REAL_OWNER_LOCK if REAL_OWNER_LOCK in BOT_NAMES else None
                owner_file = token_actual_loop if token_actual_loop in BOT_NAMES else None
                activo_real = owner_mem or owner_file or next((b for b in BOT_NAMES if estado_bots[b]["token"] == "REAL"), None)
                if activo_real in BOT_NAMES:
                    _set_ui_token_holder(activo_real)
                    _enforce_single_real_standby(activo_real)
                for bot in BOT_NAMES:
                    try:  # Aislamiento per-bot para evitar skips globales
                        if reinicio_forzado.is_set():
                            # Menos ruido: no agregar evento si repetido
                            reinicio_forzado.clear()
                            # No mostrar_panel inmediato; dejar al tick
                            break
                        await cargar_datos_bot(bot, token_actual_loop)
                        # Evita desincronizar REAL por inactividad normal durante contrato.
                        # El owner REAL se vigila en TICK_02 (watchdog sin salida a DEMO).
                        if time.time() - last_update_time[bot] > 60:
                            if estado_bots.get(bot, {}).get("token") != "REAL":
                                reiniciar_bot(bot)
                    except Exception as e_bot:
                        agregar_evento(f"⚠️ Error en {bot}: {e_bot}")
                else:
                    set_etapa("TICK_02")
                    # Watchdog para REAL pegado
                    ahora = time.time()
                    for bot in BOT_NAMES:
                        if estado_bots[bot]["token"] == "REAL":
                            t_last = last_update_time.get(bot, 0)
                            t_real = estado_bots[bot].get("real_activado_en", 0.0)
                            # Si lleva demasiado sin actualizarse desde que entró a REAL:
                            # NO salir a DEMO aquí: la salida solo ocurre con cierre GANANCIA/PÉRDIDA.
                            if t_real > 0 and (ahora - max(t_last, t_real) > REAL_TIMEOUT_S):
                                agregar_evento(f"⏱️ Seguridad: {bot} sin actividad reciente en REAL. Se mantiene REAL hasta cierre (G/P).")
                                # Rearme anti-spam del watchdog (sin liberar token ni cambiar owner).
                                estado_bots[bot]["real_activado_en"] = ahora

                    for bot in BOT_NAMES:
                        if estado_bots[bot]["token"] == "REAL":
                            # Detecta el último cierre REAL de forma robusta (sin depender de SNAPSHOT_FILAS,
                            # porque TICK_01 ya puede haber avanzado el snapshot antes de este bloque).
                            cierre_info = detectar_cierre_martingala(
                                bot,
                                min_fila=REAL_ENTRY_BASELINE.get(bot, 0),
                                require_closed=True,
                                require_real_token=True,
                                expected_ciclo=estado_bots.get(bot, {}).get("ciclo_actual", None),
                            )

                            # Ventana anti-stale tras activar REAL (protección vigente)
                            if time.time() < (estado_bots[bot].get("ignore_cierres_hasta") or 0):
                                cierre_info = None

                            # Cierre inmediato: en REAL siempre 1 operación y vuelve a DEMO (gane o pierda)
                            if cierre_info and isinstance(cierre_info, tuple) and len(cierre_info) >= 4:
                                res, monto, ciclo, payout_total = cierre_info
                                sig = (res, round(float(monto or 0.0), 2), int(ciclo or 0), round(float(payout_total or 0.0), 4))

                                # Evita reprocesar el mismo cierre en ticks consecutivos
                                if sig == LAST_REAL_CLOSE_SIG.get(bot):
                                    continue

                                LAST_REAL_CLOSE_SIG[bot] = sig

                                if res in ("GANANCIA", "PÉRDIDA"):
                                    registrar_resultado_real(res, bot=bot, ciclo_operado=ciclo)
                                    if res == "GANANCIA":
                                        cerrar_por_fin_de_ciclo(bot, "Ganancia en REAL (fin de turno)")
                                    else:
                                        cerrar_por_fin_de_ciclo(bot, "Pérdida en REAL (fin de turno)")
                                    activo_real = None
                                    break

                    if not activo_real:
                        set_etapa("TICK_03")

                        # 🔒 Lock estricto: si token_actual.txt ya tiene dueño REAL,
                        # no evaluamos ni promovemos otro bot aunque cumpla umbral.
                        owner_lock = REAL_OWNER_LOCK if REAL_OWNER_LOCK in BOT_NAMES else leer_token_actual()
                        lock_activo = owner_lock in BOT_NAMES
                        if lock_activo:
                            activo_real = owner_lock
                            _enforce_single_real_standby(owner_lock)

                        # Usamos el MISMO umbral operativo que HUD + audio
                        meta_local = _ORACLE_CACHE.get("meta") or leer_model_meta()
                        umbral_ia = max(get_umbral_operativo(meta_local or {}), float(AUTO_REAL_THR))

                        # Bloqueo por saldo (no abras ventanas si no puedes ejecutar ciclo1)
                        try:
                            saldo_val = float(obtener_valor_saldo() or 0.0)
                        except Exception:
                            saldo_val = 0.0
                        costo_ciclo1 = float(MARTI_ESCALADO[0])

                        # Candidatos: prob válida, reciente, IA activa (no OFF)
                        candidatos = []
                        if not lock_activo:
                            for b in BOT_NAMES:
                                try:
                                    modo_b = str(estado_bots.get(b, {}).get("modo_ia", "off")).lower()
                                    if modo_b == "off":
                                        continue
                                    if not ia_prob_valida(b, max_age_s=12.0):
                                        continue
                                    p = estado_bots[b].get("prob_ia", None)
                                    if isinstance(p, (int, float)) and float(p) >= float(umbral_ia):
                                        candidatos.append((float(p), b))
                                except Exception:
                                    continue

                            candidatos.sort(key=lambda x: x[0], reverse=True)

                        # Si hay señal pero saldo insuficiente -> avisar y NO abrir ventana
                        if candidatos and saldo_val < costo_ciclo1:
                            falta = costo_ciclo1 - saldo_val
                            agregar_evento(f"🚫 Señal IA bloqueada por saldo: falta {falta:.2f} USD para ciclo1 ({costo_ciclo1:.2f}).")
                            candidatos = []

                        # ==================== AUTO-PRESELECCIÓN (MODO MANUAL) ====================
                        # Si la IA detecta señal y tú estás en manual, preselecciona el mejor bot y abre la ventana
                        # para que solo elijas el ciclo (1..MAX_CICLOS) dentro del tiempo.
                        if MODO_REAL_MANUAL:
                            ahora = time.time()

                            # Si expiró, limpiamos
                            if PENDIENTE_FORZAR_BOT and PENDIENTE_FORZAR_EXPIRA and ahora > PENDIENTE_FORZAR_EXPIRA:
                                agregar_evento("⌛ Ventana de decisión expirada. Señal descartada.")
                                PENDIENTE_FORZAR_BOT = None
                                PENDIENTE_FORZAR_INICIO = 0.0
                                PENDIENTE_FORZAR_EXPIRA = 0.0

                            owner = REAL_OWNER_LOCK if REAL_OWNER_LOCK in BOT_NAMES else leer_token_actual()
                            if candidatos and (PENDIENTE_FORZAR_BOT is None) and (owner in (None, "none")):
                                candidatos.sort(reverse=True)
                                prob, mejor_bot = candidatos[0]
                                PENDIENTE_FORZAR_BOT = mejor_bot
                                PENDIENTE_FORZAR_INICIO = ahora
                                PENDIENTE_FORZAR_EXPIRA = ahora + VENTANA_DECISION_IA_S

                                # marcamos señal pendiente (sirve para contabilidad IA luego)
                                estado_bots[mejor_bot]["ia_senal_pendiente"] = True
                                estado_bots[mejor_bot]["ia_prob_senal"] = prob

                                agregar_evento(
                                    f"🟢 Señal IA en {mejor_bot} ({prob*100:.1f}%). "
                                    f"Tienes {VENTANA_DECISION_IA_S}s para elegir ciclo [1..{MAX_CICLOS}] o ESC."
                                )
                        # ==================== /AUTO-PRESELECCIÓN ====================

                        if candidatos and not MODO_REAL_MANUAL:
                            candidatos.sort(reverse=True)
                            prob, mejor_bot = candidatos[0]
                            ciclo_auto = ciclo_martingala_siguiente()
                            monto = MARTI_ESCALADO[max(0, min(len(MARTI_ESCALADO)-1, ciclo_auto - 1))]
                            val = obtener_valor_saldo()
                            if val is None or val < monto:
                                pass
                            else:
                                estado_bots[mejor_bot]["ia_senal_pendiente"] = True
                                estado_bots[mejor_bot]["ia_prob_senal"] = prob
                                ok_real = escribir_orden_real(mejor_bot, ciclo_auto)
                                if ok_real:
                                    estado_bots[mejor_bot]["fuente"] = "IA_AUTO"
                                    estado_bots[mejor_bot]["ciclo_actual"] = ciclo_auto
                                    activo_real = REAL_OWNER_LOCK if REAL_OWNER_LOCK in BOT_NAMES else mejor_bot
                                    marti_activa = True
                                else:
                                    estado_bots[mejor_bot]["ia_senal_pendiente"] = False
                                    estado_bots[mejor_bot]["ia_prob_senal"] = None
                        else:
                            max_prob = max((estado_bots[bot]["prob_ia"] for bot in BOT_NAMES if estado_bots[bot]["ia_ready"]), default=0)
                            if max_prob < umbral_ia:
                                pass

                    set_etapa("TICK_04")
                    await refresh_saldo_real()
                    if meta_mostrada and not pausado and not MODAL_ACTIVO:
                        mostrar_advertencia_meta()
                    if not MODAL_ACTIVO:
                        with RENDER_LOCK:
                            mostrar_panel()
            except Exception as e:
                set_etapa("TICK_04", f"Error: {str(e)}")
                agregar_evento(f"⚠️ Error en loop principal: {str(e)}")
                await asyncio.sleep(1)  
            await asyncio.sleep(2)
    except Exception as e:
        set_etapa("STOP", f"Error en main: {str(e)}", anunciar=True)
        agregar_evento(f"⛔ Error en main: {str(e)}")

if __name__ == "__main__":
    # ============================================
    # MODO LIMPIEZA INICIAL (Opción A datos buenos)
    # ============================================
    MODO_LIMPIEZA_DATASET = False  # ← PONER True SOLO PARA EJECUTAR LA LIMPIEZA UNA VEZ

    if MODO_LIMPIEZA_DATASET:
        print("\n🚿 MODO LIMPIEZA DATASET_INCREMENTAL ACTIVADO")
        print("   - Se borrará dataset_incremental.csv")
        print("   - Se borrarán modelo_ia.json y meta_ia.json")
        print("   - Luego se reconstruirá dataset_incremental.csv")
        print("     usando las últimas 500 filas enriquecidas de cada bot.\n")

        try:
            # 1) Borrar dataset_incremental + modelo + meta
            resetear_incremental_y_modelos(borrar_modelos=True)

            # 2) Volver a llenar dataset_incremental SOLO con datos enriquecidos buenos
            backfill_incremental(ultimas=500)

            print("\n✅ Limpieza + backfill completados correctamente.")
            print("   dataset_incremental.csv ahora contiene solo filas con:")
            print("   volatilidad, es_rebote, hora_bucket y resto de features nuevas.")
        except Exception as e:
            print(f"\n⛔ Error durante limpieza/backfill: {e}")

        input("\nPulsa ENTER para cerrar este modo, luego edita el archivo y pon MODO_LIMPIEZA_DATASET = False.")
        sys.exit(0)

    # ======================
    # MODO NORMAL (loop loop)
    # ======================
    while True:  
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            print("\n🔴 Programa terminado por el usuario.")
            break
        except Exception as e:
            print(f"⛔ Error crítico: {str(e)}")
            with open("crash.log","a",encoding="utf-8") as f:
                f.write(f"[{time.strftime('%F %T')}] {e}\n")
            time.sleep(5) 

# === FIN BLOQUE 13 ===
# === BLOQUE 99 — RESUMEN FINAL DE LO QUE SE LOGRA ===
#
# - Bot maestro 5R6M-1-2-4-8-16 con:
#   * Martingala 1-2-4-8-16 intacta.
#   * Tokens DEMO/REAL y handshake maestro→bots intactos.
#   * CSV enriquecidos, dataset_incremental.csv, IA XGBoost, reentrenos intactos.
#   * HUD visual con Prob IA, % éxito, saldo, meta, eventos
#   * Audio para GANANCIA/PÉRDIDA, racha, meta, IA 53%, etc.
# - Organización por bloques numerados:
#   ver índice de bloques al inicio del archivo.
#
# Esta organización no cambia la lógica original, solo la hace más mantenible.
# === FIN BLOQUE 99 ===
