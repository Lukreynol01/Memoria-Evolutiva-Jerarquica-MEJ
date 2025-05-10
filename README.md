# Memoria Evolutiva Jerárquica (MEJ)

Este proyecto implementa una arquitectura modular de aprendizaje automático inspirada en **evolución genética distribuida**. Utiliza diferentes modelos (SGD, MLP, XGBoost, etc.) organizados jerárquicamente en nodos para realizar entrenamiento, evaluación, migración y mejora de modelos con base en su desempeño.

##  Objetivo del Proyecto

Desarrollar un sistema evolutivo inspirado en la cognición distribuida y la selección natural que:
- Entrene modelos de forma paralela.
- Evolucione sus hiperparámetros por generaciones.
- Migre modelos exitosos entre nodos especializados.
- Permita adaptabilidad y mejoras continuas.

##  Estructura del Proyecto

- `mej_optimizada.py`: núcleo del sistema evolutivo.
- `memoria_evolutiva.pkl`: almacenamiento de la historia de modelos.
- `resultados_mej_optimizado.csv`: log de resultados por generación.

##  Tecnologías Usadas

- `Python 3.10+`
- `scikit-learn`
- `xgboost`
- `joblib`
- `numpy`, `pandas`
- `matplotlib`
- `psutil` (para monitoreo de recursos)

## ⚙ Cómo Funciona

1. **Inicializa 5 nodos jerárquicos**, cada uno con diferentes modelos:
   - Nodo 0: Modelos rápidos (SGD).
   - Nodo 1: SGD regularizado.
   - Nodo 2: MLP (red neuronal pequeña).
   - Nodo 3: XGBoost.
   - Nodo 4: HistGradientBoosting.

2. **Entrena cada nodo por 10 generaciones** de forma paralela.
3. **Evalúa el rendimiento** de cada modelo en precisión, F1, recall, etc.
4. **Guarda modelos destacados** en una memoria evolutiva persistente.
5. **Realiza migración genética**: modelos exitosos se trasladan a otros nodos.
6. **Registra resultados** por nodo y generación.

##  Métricas Calculadas

- Precisión (`accuracy`)
- Pérdida logística (`log_loss`)
- F1 Score (`macro`)
- Precision y Recall (`macro`)
- Tiempo de entrenamiento por modelo

##  Cómo Ejecutar

```bash
pip install -r requirements.txt
python mej_optimizada.py
```

##  Cómo Contribuir

¡Eres bienvenido a contribuir! Algunas ideas:
- Agregar visualización en tiempo real de evolución.
- Implementar mutación genética adaptativa.
- Agregar nuevos tipos de modelos a los nodos.
- Mejorar el sistema de migración o introducir “muerte” de modelos malos.
- Persistencia en base de datos en lugar de archivo `.pkl`.

### Pasos:
1. Haz un fork del repositorio.
2. Crea una rama (`git checkout -b feature-nueva`).
3. Realiza tus cambios.
4. Haz commit (`git commit -m 'Agrega nueva característica'`).
5. Push (`git push origin feature-nueva`).
6. Crea un Pull Request 🎉


##  Créditos

Desarrollado por Luciano Naranjo Altunar y chatgpt con inspiración en sistemas cognitivos distribuidos y evolución natural.
La IA no solo aprende: también **evoluciona**.

