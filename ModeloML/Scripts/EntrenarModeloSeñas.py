# Entrenar modelo LSTM para clasificar palabras en lenguaje de señas
import numpy as np
import os
import json
import hashlib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf

from landmarks_normalization import maybe_normalize_sequence

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

    labels.json format: {"LABEL_ID": "Etiqueta", ...}
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

# Normalización (debe coincidir con CapturarSecuencias.py / InferenciaSeñas.py)
ROTATE_PALM = True

# Cargar datos con padding para manejar diferentes longitudes
def cargar_datos(carpeta_base: Path, label_ids: list[str], max_frames=30):
    X: list[np.ndarray] = []
    y: list[int] = []
    meta: list[dict] = []

    per_class_counts: dict[str, int] = {lid: 0 for lid in label_ids}
    near_zero_count = 0
    total_count = 0

    for idx, label_id in enumerate(label_ids):
        carpeta_palabra = carpeta_base / label_id
        if not carpeta_palabra.exists():
            print(f"Advertencia: No existe carpeta para '{label_id}'")
            continue
        
        archivos = [f for f in os.listdir(carpeta_palabra) if f.endswith('.npy')]
        print(f"Cargando {len(archivos)} secuencias de '{label_id}'...")
        
        for archivo in archivos:
            ruta = os.path.join(str(carpeta_palabra), archivo)
            secuencia = np.load(ruta)

            # Normalizar si corresponde (heurística auto-detecta)
            secuencia = maybe_normalize_sequence(secuencia, rotate_palm=ROTATE_PALM)
            
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

            total_count += 1
            # Heurística: detectar secuencias casi-ceros (suele pasar si se grabó sin manos o con detección fallida)
            mean_abs = float(np.mean(np.abs(secuencia_ajustada)))
            if mean_abs < 1e-4:
                near_zero_count += 1
            
            X.append(secuencia_ajustada)
            y.append(idx)
            per_class_counts[label_id] = per_class_counts.get(label_id, 0) + 1
            meta.append({"label_id": label_id, "path": ruta, "mean_abs": mean_abs})

    print("\nResumen dataset:")
    for lid in label_ids:
        if per_class_counts.get(lid, 0) > 0:
            print(f"- {lid}: {per_class_counts[lid]} secuencias")
    if total_count > 0:
        pct = 100.0 * near_zero_count / total_count
        print(f"- Secuencias casi-ceros: {near_zero_count}/{total_count} ({pct:.1f}%)")
        if pct >= 5.0:
            print("  Advertencia: muchas secuencias casi-ceros pueden inflar accuracy o hacer que el modelo aprenda atajos.")
    
    return np.array(X), np.array(y), meta

# Cargar datos
print("Cargando secuencias...")
DATA_ROOT = _get_data_root()
LABEL_IDS, LABELS_MAP = _load_labels(DATA_ROOT)

# IDs que usará el modelo (orden de clases)
PALABRAS = LABEL_IDS
print(f"Dataset root: {DATA_ROOT}")
print(f"Clases detectadas ({len(PALABRAS)}): {PALABRAS}")

X, y, _meta = cargar_datos(DATA_ROOT, LABEL_IDS)

def _hash_seq(arr: np.ndarray) -> str:
    # Hash estable para detectar duplicados exactos (posible fuga train/test si se repite el mismo archivo)
    h = hashlib.sha1()
    h.update(arr.astype(np.float32, copy=False).tobytes())
    return h.hexdigest()

if len(X) == 0:
    print("ERROR: No se encontraron datos. Primero ejecuta CapturarSecuencias.py")
    exit()

print(f"Datos cargados: {X.shape[0]} secuencias")
print(f"Forma de cada secuencia: {X.shape}")

# Chequeo: duplicados exactos
hashes = [_hash_seq(x) for x in X]
unique_hashes = set(hashes)
dup_count = len(hashes) - len(unique_hashes)
if dup_count > 0:
    print(f"\nAdvertencia: se detectaron {dup_count} duplicados exactos en el dataset.")
    print("Esto puede inflar la accuracy, especialmente si hay duplicados repartidos entre train/val.")

# Convertir etiquetas a one-hot
y_categorical = tf.keras.utils.to_categorical(y, num_classes=len(PALABRAS))

# Dividir en train/test (30% para validación para mejor evaluación)
indices = np.arange(len(X))
train_idx, test_idx = train_test_split(
    indices, test_size=0.3, random_state=42, stratify=y
)
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y_categorical[train_idx], y_categorical[test_idx]

# Chequeo: fuga por muestras idénticas entre train/val
train_hashes = set(hashes[i] for i in train_idx)
test_hashes = set(hashes[i] for i in test_idx)
leak = len(train_hashes.intersection(test_hashes))
if leak > 0:
    print(f"\nALERTA: {leak} secuencias idénticas aparecen en train y validación (leakage probable).")
    print("Solución: elimina duplicados o separa por sesiones/personas, no por archivos sueltos.")

print(f"Entrenamiento: {X_train.shape[0]} secuencias")
print(f"Validación: {X_test.shape[0]} secuencias")

# Crear modelo LSTM (reducido para evitar overfitting)
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.LSTM(64, return_sequences=False),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(len(PALABRAS), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Callbacks
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'modelo_señas_best.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stop = tf.keras.callbacks.EarlyStopping(
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

# Métricas por clase (ayuda a ver si solo memoriza una clase o confunde pocas)
y_true = np.argmax(y_test, axis=1)
y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
print("\nMatriz de confusión (validación):")
print(confusion_matrix(y_true, y_pred))
print("\nReporte por clase (validación):")
print(classification_report(y_true, y_pred, target_names=PALABRAS, digits=4))
