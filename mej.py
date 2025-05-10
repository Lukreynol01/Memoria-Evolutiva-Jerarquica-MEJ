import numpy as np
import pandas as pd
import time
import joblib
import logging
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss, f1_score, precision_score, recall_score
from sklearn.datasets import make_classification
from joblib import Parallel, delayed
import psutil
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

# Configuración de logs para depuración
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Inicializar memoria evolutiva
try:
    memoria_evolutiva = joblib.load("memoria_evolutiva.pkl")
    logging.info("Memoria evolutiva cargada exitosamente.")
except FileNotFoundError:
    memoria_evolutiva = []
    logging.info("No se encontró una memoria evolutiva previa. Se inicia una nueva.")

# Generar datos de prueba
X_train, y_train = make_classification(n_samples=10000, n_features=256, n_informative=200, n_classes=50, random_state=42)
X_test, y_test = make_classification(n_samples=2000, n_features=256, n_informative=200, n_classes=50, random_state=43)

# Configuración de nodos y generaciones
NUM_NODOS = 5
NUM_GENERACIONES = 10
MAX_MODELOS_POR_NODO = 5

# Definición de niveles jerárquicos
jerarquia_nodos = {
    0: "Exploración (Modelos rápidos)",
    1: "Exploración Regularizada",
    2: "Especialización (Redes neuronales pequeñas)",
    3: "Especialización avanzada",
    4: "Refinamiento (Boosting y Random Forest)"
}

# Función segura para valores None
def safe(value, default=0.0):
    return value if value is not None else default

# Log de memoria
def log_memoria():
    memoria = psutil.virtual_memory()
    logging.info(f"Memoria utilizada: {memoria.percent}% - Libre: {memoria.available / (1024 ** 3):.2f} GB")

# Guardar resultados de modelos entrenados
def guardar_memoria(modelo, nodo_id, gen, acc, tiempo, logloss, f1, precision, recall):
    entrada = {
        "nodo": nodo_id,
        "generación": gen,
        "modelo": modelo,
        "precisión": safe(acc),
        "logloss": safe(logloss),
        "f1": safe(f1),
        "precision": safe(precision),
        "recall": safe(recall),
        "tiempo": tiempo
    }
    memoria_evolutiva.append(entrada)
    joblib.dump(memoria_evolutiva, "memoria_evolutiva.pkl")
    logging.info(f"Modelo guardado: Nodo {nodo_id}, Gen {gen}, Precisión {acc:.4f}, F1 {f1:.4f}")

# Entrenamiento y evaluación de modelo
def entrenar_modelo(modelo, X_train, y_train, X_test, y_test, nodo_id, gen):
    try:
        start_time = time.time()
        modelo.fit(X_train, y_train)
        train_time = time.time() - start_time
        pred = modelo.predict(X_test)
        probas = modelo.predict_proba(X_test) if hasattr(modelo, 'predict_proba') else None

        acc = accuracy_score(y_test, pred)
        logloss = log_loss(y_test, probas) if probas is not None else None
        f1 = f1_score(y_test, pred, average='macro')
        precision = precision_score(y_test, pred, average='macro')
        recall = recall_score(y_test, pred, average='macro')

        guardar_memoria(modelo, nodo_id, gen, acc, train_time, logloss, f1, precision, recall)
        return modelo, acc, train_time, logloss, f1
    except Exception as e:
        logging.error(f"Error entrenando modelo Nodo {nodo_id}, Gen {gen}: {e}")
        return None, None, None, None, None

# Migración genética con límite
def migracion_genetica(nodos_modelos, memoria_evolutiva):
    logging.info("Iniciando migración genética")
    precisiones = [m["precisión"] for m in memoria_evolutiva if m["precisión"] is not None]
    if not precisiones:
        return nodos_modelos
    promedio = np.mean(precisiones)

    for i in range(NUM_NODOS):
        destino = (i + 1) % NUM_NODOS
        candidatos = [m for m in memoria_evolutiva if m["nodo"] == i and m["precisión"] > promedio][:2]
        for m in candidatos:
            if len(nodos_modelos[destino]) < MAX_MODELOS_POR_NODO:
                nodos_modelos[destino].append(m["modelo"])
    return nodos_modelos

# Evolución de modelos por nodo
def evolucionar_nodo(nodo_id, modelos):
    logging.info(f"{jerarquia_nodos[nodo_id]} iniciando evolución (Nodo {nodo_id})")
    resultados = []

    for gen in range(NUM_GENERACIONES):
        logging.info(f"Nodo {nodo_id} - Generación {gen+1}")
        log_memoria()
        try:
            entrenados = Parallel(n_jobs=-1, backend="threading")(delayed(entrenar_modelo)(modelo, X_train, y_train, X_test, y_test, nodo_id, gen) for modelo in modelos)
        except Exception as e:
            logging.error(f"Error en paralelo Nodo {nodo_id}, Gen {gen+1}: {e}")
            entrenados = [(None, None, None, None, None)] * len(modelos)

        for i, (modelo, acc, tiempo, logloss, f1) in enumerate(entrenados):
            if modelo is not None:
                modelos[i] = modelo
                resultados.append({
                    "nodo": nodo_id,
                    "generación": gen + 1,
                    "precisión": acc,
                    "logloss": logloss,
                    "f1": f1,
                    "tiempo": tiempo
                })
            else:
                logging.warning(f"Modelo {i} en Nodo {nodo_id}, Gen {gen+1} falló")

    return resultados, modelos

# Inicializar modelos jerárquicamente
nodos_modelos = [
    [SGDClassifier(max_iter=20, warm_start=True, random_state=i) for i in range(2)],
    [SGDClassifier(max_iter=20, warm_start=True, penalty='l2', alpha=0.005, random_state=i + 10) for i in range(2)],
    [MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=100, alpha=0.0001, learning_rate_init=0.001, random_state=i + 20, activation='relu', solver='adam', batch_size=64) for i in range(1)],
    [XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=i + 30) for i in range(1)],
    [HistGradientBoostingClassifier(max_iter=8, max_leaf_nodes=8, learning_rate=0.1, early_stopping=True, n_iter_no_change=5, random_state=i + 40) for i in range(1)]
]

# Ejecutar todo el sistema evolutivo
todos_resultados = []
try:
    for nodo_id in range(NUM_NODOS):
        resultado_nodo, modelos_actualizados = evolucionar_nodo(nodo_id, nodos_modelos[nodo_id])
        todos_resultados.extend(resultado_nodo)
        nodos_modelos[nodo_id] = modelos_actualizados
        if nodo_id % 2 == 0:
            nodos_modelos = migracion_genetica(nodos_modelos, memoria_evolutiva)
except Exception as e:
    logging.critical(f"Error crítico durante la evolución: {e}")

# Guardar resultados
if todos_resultados:
    df_resultados = pd.DataFrame(todos_resultados)
    df_resultados.to_csv("resultados_mej_optimizado.csv", index=False)
    print("Evolución completada. Resultados guardados en 'resultados_mej_optimizado.csv'")
else:
    print("No se generaron resultados debido a errores en la ejecución.")
