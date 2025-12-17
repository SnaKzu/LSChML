# Inferencia en tiempo real con modelo de lenguaje de señas
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque

# Cargar modelo entrenado
modelo = load_model('modelo_señas_final.h5')

# Palabras (deben estar en el mismo orden que en el entrenamiento)
PALABRAS = ["DIEGO", "GRACIAS", "HOLA", "MI_NOMBRE", "NOS_VEMOS"]

# Configuración
FRAMES_POR_SECUENCIA = 30  # Máximo de frames que el modelo espera
secuencia_buffer = deque(maxlen=FRAMES_POR_SECUENCIA)

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

prediccion_actual = ""
confianza_actual = 0

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
        
        # Extraer landmarks (siempre 126 valores exactos)
        frame_data = [0.0] * 126
        
        idx = 0
        for hand_landmarks in results.multi_hand_landmarks[:2]:  # Máximo 2 manos
            for landmark in hand_landmarks.landmark:
                frame_data[idx] = landmark.x
                frame_data[idx + 1] = landmark.y
                frame_data[idx + 2] = landmark.z
                idx += 3
        
        # Agregar a buffer
        secuencia_buffer.append(frame_data)
        
        # Hacer predicción cuando tengamos al menos 30 frames
        if len(secuencia_buffer) >= 30:
            secuencia = np.array(list(secuencia_buffer))
            
            # Aplicar padding si es necesario (para alcanzar 60 frames)
            if len(secuencia) < FRAMES_POR_SECUENCIA:
                padding = np.zeros((FRAMES_POR_SECUENCIA - len(secuencia), 126))
                secuencia = np.vstack([secuencia, padding])
            
            secuencia = np.expand_dims(secuencia, axis=0)
            
            # Predecir
            predicciones = modelo.predict(secuencia, verbose=0)[0]
            idx_prediccion = np.argmax(predicciones)
            confianza = predicciones[idx_prediccion]
            
            if confianza > 0.6:  # Umbral de confianza (reducido para mejor detección)
                prediccion_actual = PALABRAS[idx_prediccion]
                confianza_actual = confianza
    else:
        # Si no hay manos, limpiar buffer
        secuencia_buffer.clear()
    
    # Mostrar predicción principal
    if prediccion_actual:
        texto = f"{prediccion_actual} ({confianza_actual*100:.1f}%)"
        cv2.rectangle(frame, (10, 10), (500, 80), (0, 255, 0), -1)
        cv2.putText(frame, texto, (20, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    
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
