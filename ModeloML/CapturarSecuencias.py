# Capturar secuencias de poses de manos para palabras en lenguaje de señas
import cv2
import mediapipe as mp
import numpy as np
import os
import json

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Configuración
PALABRA_ACTUAL = "NOS_VEMOS_2"  # Cambia esto por cada palabra
NUM_SECUENCIAS = 100  # Número de videos por palabra
FRAMES_POR_SECUENCIA = 30  # Frames por video (1 segundo a 30 fps)

# Inicializar cámara
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Inicializar detector de manos
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

print(f"Capturando secuencias para la palabra: {PALABRA_ACTUAL}")

# Verificar si ya existe la carpeta y contar videos existentes
carpeta_datos = f"C:/Users/Diego/Documents/Test8/SecuenciasSeñas/{PALABRA_ACTUAL}"
secuencia_actual = 0

if os.path.exists(carpeta_datos):
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

while secuencia_actual < NUM_SECUENCIAS:
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
    
    # Mostrar estado
    if not grabando:
        texto = f"Secuencia {secuencia_actual + 1}/{NUM_SECUENCIAS} - Presiona ESPACIO"
        color = (0, 255, 0)
    else:
        texto = f"GRABANDO: {frames_capturados}/{FRAMES_POR_SECUENCIA}"
        color = (0, 0, 255)
    
    cv2.putText(frame, texto, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, f"Palabra: {PALABRA_ACTUAL}", (10, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    
    # Capturar landmarks si está grabando Y HAY MANOS DETECTADAS
    if grabando and results.multi_hand_landmarks:
        # Inicializar con ceros (126 valores = 21 landmarks × 3 coords × 2 manos)
        frame_data = [0.0] * 126
        
        # Llenar con coordenadas de las manos detectadas
        idx = 0
        for hand_landmarks in results.multi_hand_landmarks[:2]:  # Máximo 2 manos
            for landmark in hand_landmarks.landmark:
                frame_data[idx] = landmark.x
                frame_data[idx + 1] = landmark.y
                frame_data[idx + 2] = landmark.z
                idx += 3
        
        secuencia_frames.append(frame_data)
        frames_capturados += 1
        
        # Si completamos la secuencia
        if frames_capturados >= FRAMES_POR_SECUENCIA:
            # Guardar secuencia
            archivo = os.path.join(carpeta_datos, f"{PALABRA_ACTUAL}_{secuencia_actual}.npy")
            np.save(archivo, np.array(secuencia_frames))
            
            print(f"Secuencia {secuencia_actual + 1} guardada: {archivo}")
            
            # Reset
            secuencia_actual += 1
            grabando = False
            frames_capturados = 0
            secuencia_frames = []
    
    cv2.imshow("Captura de Lenguaje de Señas", frame)
    
    # Controles
    key = cv2.waitKey(1)
    if key == 32 and not grabando:  # ESPACIO
        # Crear carpeta si no existe
        if not os.path.exists(carpeta_datos):
            os.makedirs(carpeta_datos)
            print(f"Carpeta creada: {carpeta_datos}")
        
        grabando = True
        print(f"Iniciando grabación de secuencia {secuencia_actual + 1}...")
    elif key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
hands.close()

print(f"\n¡Captura completada! {secuencia_actual} secuencias guardadas en {carpeta_datos}")
