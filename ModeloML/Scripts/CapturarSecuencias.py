# Capturar secuencias de poses de manos para palabras en lenguaje de señas
import cv2
import mediapipe as mp
import numpy as np
import os
import json
import re
import unicodedata
import time
from pathlib import Path

from landmarks_normalization import frame_from_mediapipe_results

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Configuración
PALABRA_ACTUAL = "bienvenidos"  # Cambia esto por cada palabra
NUM_SECUENCIAS = 100  # Número de videos por palabra
FRAMES_POR_SECUENCIA = 30  # Frames por video (1 segundo a 30 fps)
ROTATE_PALM = True  # Rotar para alinear eje palma (reduce variación por pose)

# --- Estabilidad con movimiento rápido ---
# Si la mano se mueve rápido, el tracker puede perderla (motion blur + baja FPS).
# 1) "DETECT_EVERY_FRAME" fuerza detección en cada frame (más robusto, pero más costoso).
# 2) Procesar a menor resolución aumenta FPS y reduce pérdidas.
DETECT_EVERY_FRAME = True
MODEL_COMPLEXITY = 0  # 0 = más rápido, 1 = más preciso (pero más lento)
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.3
PROCESS_WIDTH = 640  # 0 para desactivar, o 640/800 para subir FPS

# Suavizado: mantener el último resultado algunos frames cuando MediaPipe pierde la mano
HOLD_LAST_FRAMES = 0

# Captura bimanual
# Para señas donde se usan 2 manos (p.ej. "por favor"), evita grabar frames con solo 1 mano.
REQUIRE_TWO_HANDS = True
# Permite perder una mano por pocos frames sin cancelar (oclusiones al tocarse).
TWO_HANDS_GRACE_FRAMES = 5
# Tiempo para que el usuario coloque ambas manos antes de empezar a grabar.
# Si es <= 0, espera indefinidamente hasta detectar 2 manos.
WAIT_FOR_TWO_HANDS_SECONDS = 0.0


def to_safe_label_id(text: str) -> str:
    """Convertir una etiqueta de formato libre en una identificación de carpeta/archivo compatible con Windows.

    - Elimina los acentos
    - Elimina los caracteres no válidos en Windows (por ejemplo, ? * : < > | \\ / \")
    - Reduce los espacios en blanco a guiones bajos
    - Pone mayúsculas para mantener la coherencia
    """
    text = text.strip()
    # Normalize accents: "cómo" -> "como"
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.upper()
    # Replace whitespace with underscore
    text = re.sub(r"\s+", "_", text)
    # Keep only A-Z, 0-9, underscore, hyphen
    text = re.sub(r"[^A-Z0-9_-]", "", text)
    # Avoid empty label
    return text or "LABEL"

# Inicializar cámara
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Inicializar detector de manos
hands = mp_hands.Hands(
    static_image_mode=DETECT_EVERY_FRAME,
    max_num_hands=2,
    model_complexity=MODEL_COMPLEXITY,
    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
)

LABEL_ID = to_safe_label_id(PALABRA_ACTUAL)
print(f"Capturando secuencias para la palabra: {PALABRA_ACTUAL} (ID: {LABEL_ID})")

# Verificar si ya existe la carpeta y contar videos existentes
default_root = Path(__file__).resolve().parent / "DataLS"
data_root = Path(os.environ.get("LSCH_DATA_DIR", str(default_root)))
carpeta_datos = data_root / LABEL_ID
secuencia_actual = 0

# Guardar/actualizar mapeo de etiquetas (ID -> etiqueta humana)
labels_path = data_root / "labels.json"
try:
    data_root.mkdir(parents=True, exist_ok=True)
    if labels_path.exists():
        labels = json.loads(labels_path.read_text(encoding="utf-8"))
        if not isinstance(labels, dict):
            labels = {}
    else:
        labels = {}
    labels[LABEL_ID] = PALABRA_ACTUAL
    labels_path.write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
except Exception as e:
    print(f"Advertencia: no se pudo escribir labels.json ({e})")

if carpeta_datos.exists():
    # Contar archivos .npy existentes
    archivos_existentes = [f for f in os.listdir(carpeta_datos) if f.endswith('.npy')]
    secuencia_actual = len(archivos_existentes)
    print(f"Carpeta existente encontrada: {carpeta_datos}")
    print(f"Videos ya capturados: {secuencia_actual}")
    print(f"Continuando desde la secuencia {secuencia_actual + 1}")
else:
    print(f"Carpeta nueva: {carpeta_datos}")

print(f"Total a capturar: {NUM_SECUENCIAS}")
print(f"Faltan: {NUM_SECUENCIAS - secuencia_actual} secuencias")
print("\nPresiona ESPACIO para comenzar a grabar cada secuencia")
print("Presiona ESC para salir")

grabando = False
frames_capturados = 0
secuencia_frames = []
missing_two_hands = 0
recording_started = False
armed_at = 0.0

# Cache para suavizado
last_multi_hand_landmarks = None
last_frame_data = None
missing_hands_frames = 0

while secuencia_actual < NUM_SECUENCIAS:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convertir a RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesar a menor resolución para mejorar FPS (mejor tracking en movimientos rápidos)
    if PROCESS_WIDTH and frame_rgb.shape[1] > PROCESS_WIDTH:
        scale = PROCESS_WIDTH / frame_rgb.shape[1]
        proc_h = max(1, int(frame_rgb.shape[0] * scale))
        frame_rgb_proc = cv2.resize(frame_rgb, (PROCESS_WIDTH, proc_h), interpolation=cv2.INTER_AREA)
    else:
        frame_rgb_proc = frame_rgb

    results = hands.process(frame_rgb_proc)
    
    # Dibujar manos
    if results.multi_hand_landmarks:
        last_multi_hand_landmarks = results.multi_hand_landmarks
        missing_hands_frames = 0
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        missing_hands_frames += 1
        # Si hay pérdida breve, dibujar el último esqueleto para evitar "parpadeo"
        if last_multi_hand_landmarks is not None and missing_hands_frames <= HOLD_LAST_FRAMES:
            for hand_landmarks in last_multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    # Contar manos detectadas
    hands_detected = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0

    # Mostrar estado
    if not grabando:
        texto = f"Secuencia {secuencia_actual + 1}/{NUM_SECUENCIAS} - Presiona ESPACIO"
        color = (0, 255, 0)
    else:
        if REQUIRE_TWO_HANDS and not recording_started:
            if WAIT_FOR_TWO_HANDS_SECONDS > 0:
                remaining = max(0.0, WAIT_FOR_TWO_HANDS_SECONDS - (time.monotonic() - armed_at))
                texto = f"COLOCA 2 MANOS: {remaining:0.1f}s"
            else:
                texto = "COLOCA 2 MANOS"
            color = (0, 165, 255)
        elif REQUIRE_TWO_HANDS and hands_detected < 2:
            texto = f"ESPERANDO 2 MANOS ({missing_two_hands}/{TWO_HANDS_GRACE_FRAMES})"
            color = (0, 165, 255)
        else:
            texto = f"GRABANDO: {frames_capturados}/{FRAMES_POR_SECUENCIA}"
            color = (0, 0, 255)
    
    cv2.putText(frame, texto, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, f"Palabra: {PALABRA_ACTUAL}", (10, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    
    # Capturar landmarks si está grabando
    if grabando:
        # Fase 1: dar tiempo para mostrar 2 manos antes de empezar a grabar
        if REQUIRE_TWO_HANDS and not recording_started:
            if hands_detected >= 2:
                recording_started = True
                missing_two_hands = 0
                print("2 manos detectadas. Comenzando grabación...")
            else:
                if WAIT_FOR_TWO_HANDS_SECONDS > 0 and (time.monotonic() - armed_at) > WAIT_FOR_TWO_HANDS_SECONDS:
                    print("No se detectaron 2 manos a tiempo. Cancelando secuencia...")
                    grabando = False
                    frames_capturados = 0
                    secuencia_frames = []
                    missing_two_hands = 0
                    recording_started = False
                # No grabar frames hasta que estén ambas manos
                cv2.imshow("Captura de Lenguaje de Señas", frame)
                key = cv2.waitKey(1)
                if key == 27:  # ESC
                    break
                continue

        # Fase 2: grabación activa (con tolerancia a oclusiones)
        if REQUIRE_TWO_HANDS and hands_detected < 2:
            missing_two_hands += 1
            if missing_two_hands > TWO_HANDS_GRACE_FRAMES:
                print("Se perdieron 2 manos por demasiado tiempo. Reiniciando secuencia...")
                grabando = False
                frames_capturados = 0
                secuencia_frames = []
                missing_two_hands = 0
                recording_started = False
        elif hands_detected > 0:
            # frame_from_mediapipe_results ahora ordena LEFT/RIGHT cuando MediaPipe entrega handedness
            frame_data = frame_from_mediapipe_results(results, rotate_palm=ROTATE_PALM)
            secuencia_frames.append(frame_data)
            frames_capturados += 1
            missing_two_hands = 0
            last_frame_data = frame_data
        elif hands_detected == 0 and (not REQUIRE_TWO_HANDS):
            # Si se pierde la mano por instantes (movimiento rápido), reutilizar el último frame válido
            # para mantener continuidad de la secuencia.
            if last_frame_data is not None and missing_hands_frames <= HOLD_LAST_FRAMES:
                secuencia_frames.append(last_frame_data)
                frames_capturados += 1
        
        # Si completamos la secuencia
        if frames_capturados >= FRAMES_POR_SECUENCIA:
            # Guardar secuencia
            archivo = carpeta_datos / f"{LABEL_ID}_{secuencia_actual}.npy"
            np.save(str(archivo), np.array(secuencia_frames, dtype=np.float32))
            
            print(f"Secuencia {secuencia_actual + 1} guardada: {archivo}")
            
            # Reset
            secuencia_actual += 1
            grabando = False
            frames_capturados = 0
            secuencia_frames = []
            missing_two_hands = 0
            recording_started = False
    
    cv2.imshow("Captura de Lenguaje de Señas", frame)
    
    # Controles
    key = cv2.waitKey(1)
    if key == 32 and not grabando:  # ESPACIO
        # Crear carpeta si no existe
        if not carpeta_datos.exists():
            carpeta_datos.mkdir(parents=True, exist_ok=True)
            print(f"Carpeta creada: {carpeta_datos}")
        
        grabando = True
        missing_two_hands = 0
        recording_started = False
        armed_at = time.monotonic()
        print(f"Iniciando grabación de secuencia {secuencia_actual + 1}...")
    elif key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
hands.close()

print(f"\n¡Captura completada! {secuencia_actual} secuencias guardadas en {carpeta_datos}")
