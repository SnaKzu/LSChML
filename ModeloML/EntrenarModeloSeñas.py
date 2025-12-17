# Entrenar modelo LSTM para clasificar palabras en lenguaje de señas
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Palabras a reconocer (agrega más según captures)
PALABRAS = ["DIEGO", "GRACIAS", "HOLA", "MI_NOMBRE", "NOS_VEMOS"]

# Cargar datos con padding para manejar diferentes longitudes
def cargar_datos(carpeta_base="CAMBIAR A UBICACIÓN DE SecuenciasSeñas", max_frames=30):
    X = []
    y = []
    
    for idx, palabra in enumerate(PALABRAS):
        carpeta_palabra = os.path.join(carpeta_base, palabra)
        if not os.path.exists(carpeta_palabra):
            print(f"Advertencia: No existe carpeta para '{palabra}'")
            continue
        
        archivos = [f for f in os.listdir(carpeta_palabra) if f.endswith('.npy')]
        print(f"Cargando {len(archivos)} secuencias de '{palabra}'...")
        
        for archivo in archivos:
            ruta = os.path.join(carpeta_palabra, archivo)
            secuencia = np.load(ruta)
            
            # Verificar que tenga 126 valores por frame
            if secuencia.shape[1] != 126:
                print(f"Archivo ignorado (dimensión incorrecta): {archivo} - {secuencia.shape}")
                continue
            
            # Ajustar longitud: padding con ceros o truncar
            num_frames = secuencia.shape[0]
            if num_frames < max_frames:
                # Padding: rellenar con ceros al final
                padding = np.zeros((max_frames - num_frames, 126))
                secuencia_ajustada = np.vstack([secuencia, padding])
            elif num_frames > max_frames:
                # Truncar: tomar solo los primeros max_frames
                secuencia_ajustada = secuencia[:max_frames]
            else:
                secuencia_ajustada = secuencia
            
            X.append(secuencia_ajustada)
            y.append(idx)
    
    return np.array(X), np.array(y)

# Cargar datos
print("Cargando secuencias...")
X, y = cargar_datos()

if len(X) == 0:
    print("ERROR: No se encontraron datos. Primero ejecuta CapturarSecuencias.py")
    exit()

print(f"Datos cargados: {X.shape[0]} secuencias")
print(f"Forma de cada secuencia: {X.shape}")

# Convertir etiquetas a one-hot
y_categorical = to_categorical(y, num_classes=len(PALABRAS))

# Dividir en train/test (30% para validación para mejor evaluación)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.3, random_state=42, stratify=y
)

print(f"Entrenamiento: {X_train.shape[0]} secuencias")
print(f"Validación: {X_test.shape[0]} secuencias")

# Crear modelo LSTM (reducido para evitar overfitting)
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.4),
    LSTM(64, return_sequences=False),
    Dropout(0.4),
    Dense(32, activation='relu'),
    Dropout(0.4),
    Dense(len(PALABRAS), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Callbacks
checkpoint = ModelCheckpoint(
    'modelo_señas_best.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=50,  # Más paciencia para permitir mejor convergencia
    restore_best_weights=True,
    verbose=1
)

# Entrenar
print("\nIniciando entrenamiento...")
print("Esto tomará varios minutos. Paciencia...\n")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=200,  # Más épocas para mejor entrenamiento
    batch_size=8,  # Batch más pequeño para aprendizaje más detallado
    callbacks=[checkpoint, early_stop],
    verbose=1
)

# Guardar modelo final
model.save('modelo_señas_final.h5')
print("\n¡Entrenamiento completado!")
print(f"Modelo guardado en: modelo_señas_best.h5 y modelo_señas_final.h5")

# Evaluar
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nPrecisión en validación: {accuracy * 100:.2f}%")
