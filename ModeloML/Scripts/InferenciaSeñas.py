# Inferencia en tiempo real con modelo de lenguaje de señas
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque
import os
import json
from pathlib import Path
import unicodedata
import time

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = ImageDraw = ImageFont = None

from landmarks_normalization import frame_from_mediapipe_results

def _get_data_root() -> Path:
    """Resolve dataset root directory.

    Priority:
    1) env var LSCH_DATA_DIR
    2) Scripts/DataLS (default for this repo)
    """
    env = os.environ.get("LSCH_DATA_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent / "DataLS"


def _load_labels(data_root: Path) -> tuple[list[str], dict[str, str]]:
    """Load labels mapping from labels.json and return sorted label IDs.

    labels.json format: {"LABEL_ID": "Etiqueta humana", ...}
    """
    labels_path = data_root / "labels.json"
    if not labels_path.exists():
        raise FileNotFoundError(
            f"No se encontró labels.json en: {labels_path}. "
            "Primero captura datos con CapturarSecuencias.py."
        )
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    if not isinstance(labels, dict) or not labels:
        raise ValueError(f"labels.json inválido o vacío: {labels_path}")
    label_ids = sorted(labels.keys())
    return label_ids, {str(k): str(v) for k, v in labels.items()}


def _resolve_model_path() -> Path:
    """Prefer model next to this script; fallback to current working dir."""
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "modelo_señas_final.h5",
        Path.cwd() / "modelo_señas_final.h5",
        script_dir / "modelo_senas_final.h5",
        Path.cwd() / "modelo_senas_final.h5",
    ]
    for p in candidates:
        if p.exists():
            return p
    # default (will raise a clearer error from load_model)
    return candidates[0]


# Cargar labels (mismo orden que entrenamiento)
DATA_ROOT = _get_data_root()
LABEL_IDS, LABELS_MAP = _load_labels(DATA_ROOT)
PALABRAS = LABEL_IDS

# Cargar modelo entrenado
MODEL_PATH = _resolve_model_path()
modelo = tf.keras.models.load_model(str(MODEL_PATH))

# Configuración
FRAMES_POR_SECUENCIA = 30  # Máximo de frames que el modelo espera
secuencia_buffer = deque(maxlen=FRAMES_POR_SECUENCIA)
ROTATE_PALM = True  # Debe coincidir con captura/entrenamiento

# --- Anti falsos positivos (gating / smoothing) ---
# Umbral mínimo de confianza para aceptar una predicción
CONFIDENCE_THRESHOLD = 0.90
# Exigir separación entre top1 y top2 (evita predicciones "dudosas")
MARGIN_THRESHOLD = 0.15
# La predicción debe repetirse estos frames seguidos para mostrarse
STABLE_PRED_FRAMES = 8
# Mantener el texto un rato aunque la confianza baje (evita parpadeo)
DISPLAY_HOLD_SECONDS = 0.8

# Opcional: exigir algo de movimiento reciente para "armar" predicción.
# Esto ayuda a que una mano en reposo no dispare "GRACIAS" u otra clase.
REQUIRE_MOTION = True
MOTION_WINDOW = 12
MIN_MOTION = 0.004  # Ajustable: sube si sigue disparando en reposo; baja si cuesta detectar

pred_history: deque[int | None] = deque(maxlen=STABLE_PRED_FRAMES)
last_shown_at = 0.0

# Inicializar MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Inicializar cámara
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

print("Sistema de reconocimiento de lenguaje de señas activo")
print("Presiona ESC para salir")
print(f"Dataset root: {DATA_ROOT}")
print(f"Modelo: {MODEL_PATH}")
print(f"Clases ({len(PALABRAS)}): {PALABRAS}")

prediccion_actual = ""
confianza_actual = 0


def _strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def _draw_text(frame_bgr: np.ndarray, text: str, org: tuple[int, int], font_size: int = 48) -> None:
    """Draw UTF-8 text onto an OpenCV BGR frame.

    OpenCV's cv2.putText doesn't reliably render accents on Windows (often shows '?').
    We use Pillow with a system TTF font when available.
    """
    if ImageFont is None:
        # Fallback: OpenCV only (ASCII-safe)
        safe = text.encode("ascii", "replace").decode("ascii")
        cv2.putText(frame_bgr, safe, org, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        return

    # Try common Windows fonts
    font_candidates = [
        r"C:\\Windows\\Fonts\\segoeui.ttf",
        r"C:\\Windows\\Fonts\\arial.ttf",
    ]
    font = None
    for fp in font_candidates:
        try:
            if Path(fp).exists():
                font = ImageFont.truetype(fp, font_size)
                break
        except Exception:
            font = None

    if font is None:
        # Last resort: no font; strip accents and use OpenCV
        safe = _strip_accents(text)
        cv2.putText(frame_bgr, safe, org, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        return

    # Pillow draw
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(img_pil)
    draw.text(org, text, font=font, fill=(255, 255, 255))
    frame_bgr[:, :, :] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convertir a RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    # Dibujar manos
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Extraer landmarks normalizados (siempre 126 valores exactos)
        frame_data = frame_from_mediapipe_results(results, rotate_palm=ROTATE_PALM)
        
        # Agregar a buffer
        secuencia_buffer.append(frame_data)
        
        # Hacer predicción cuando tengamos al menos 30 frames
        if len(secuencia_buffer) >= 30:
            secuencia = np.array(list(secuencia_buffer), dtype=np.float32)
            secuencia = np.expand_dims(secuencia, axis=0)  # (1, 30, 126)

            # Motion gate: si está muy "quieto" (reposo), no intentes clasificar
            if REQUIRE_MOTION:
                window = min(MOTION_WINDOW, secuencia.shape[1])
                seq_w = secuencia[0, -window:, :]
                if window >= 2:
                    motion = float(np.mean(np.abs(np.diff(seq_w, axis=0))))
                else:
                    motion = 0.0
            else:
                motion = 1.0
            
            # Predecir
            predicciones = modelo.predict(secuencia, verbose=0)[0]
            idx_prediccion = np.argmax(predicciones)
            confianza = predicciones[idx_prediccion]

            # Margin gate: top1 debe superar suficientemente a top2
            if predicciones.shape[0] >= 2:
                top2 = float(np.partition(predicciones, -2)[-2])
            else:
                top2 = 0.0
            margin = float(confianza - top2)

            accepted = (
                confianza >= CONFIDENCE_THRESHOLD
                and margin >= MARGIN_THRESHOLD
                and motion >= MIN_MOTION
            )

            pred_history.append(int(idx_prediccion) if accepted else None)

            # Si los últimos N frames aceptados son la misma clase, mostrarla
            if len(pred_history) == STABLE_PRED_FRAMES and all(
                p is not None and p == pred_history[0] for p in pred_history
            ):
                label_id = PALABRAS[pred_history[0]]
                prediccion_actual = LABELS_MAP.get(label_id, label_id)
                confianza_actual = float(confianza)
                last_shown_at = time.monotonic()
    else:
        # Si no hay manos, limpiar buffer
        secuencia_buffer.clear()
        pred_history.clear()
    
    # Si no hay señal estable hace un rato, limpiar texto
    if prediccion_actual and (time.monotonic() - last_shown_at) > DISPLAY_HOLD_SECONDS:
        # Si seguimos teniendo manos pero no hay una predicción estable/aceptada, ocultar
        if not results.multi_hand_landmarks or (len(pred_history) > 0 and pred_history[-1] is None):
            prediccion_actual = ""
            confianza_actual = 0

    # Mostrar predicción principal
    if prediccion_actual:
        texto = f"{prediccion_actual} ({confianza_actual*100:.1f}%)"
        cv2.rectangle(frame, (10, 10), (500, 80), (0, 255, 0), -1)
        _draw_text(frame, texto, (20, 20), font_size=48)
    
    # Mostrar estado del buffer
    estado = f"Buffer: {len(secuencia_buffer)}/{FRAMES_POR_SECUENCIA}"
    cv2.putText(frame, estado, (10, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    cv2.imshow("Reconocimiento de Lenguaje de Señas", frame)
    
    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
