# 📌 EPOCH (Época)
# Una época es una pasada completa por todo el dataset de entrenamiento.
# Afecta al tiempo de entrenamiento: más épocas pueden mejorar el modelo, pero demasiadas pueden causar sobreajuste.
EPOCHS = 10  # Se entrenará el modelo 10 veces sobre todos los datos

# 📌 BATCH (Lote)
# Un batch es la cantidad de muestras procesadas antes de actualizar los pesos del modelo.
# Afecta la velocidad y estabilidad del entrenamiento.
BATCH_SIZE = 32  # Se entrenarán 32 imágenes a la vez antes de actualizar los pesos

# 📌 STEPS PER EPOCH (Pasos por época)
# Número de lotes (batches) procesados en cada época.
# Se calcula como: total de imágenes / BATCH_SIZE
STEPS_PER_EPOCH = train_generator.samples // BATCH_SIZE  

# 📌 VALIDATION STEPS (Pasos de validación)
# Número de lotes usados en cada validación después de una época.
# Un número bajo puede hacer que la validación sea menos precisa.
VALIDATION_STEPS = validation_generator.samples // BATCH_SIZE  

# 📌 IMG_SIZE (Tamaño de imagen)
# Define la resolución de entrada del modelo. 
# Imágenes más grandes capturan más detalles, pero consumen más memoria.
IMG_SIZE = (224, 224)  # Tamaño común para modelos preentrenados

# 📌 LEARNING RATE (Tasa de aprendizaje)
# Controla cuánto se ajustan los pesos del modelo en cada actualización.
# Un valor alto puede hacer que el modelo no aprenda bien, uno muy bajo puede hacer que aprenda muy lento.
LEARNING_RATE = 0.001  

# 📌 OPTIMIZER (Optimizador)
# Define cómo se ajustan los pesos del modelo. Adam es una opción común y estable.
OPTIMIZER = 'adam'  

# 📌 LOSS FUNCTION (Función de pérdida)
# Mide cuán bien está funcionando el modelo. Depende del tipo de problema:
# - 'categorical_crossentropy' para clasificación con múltiples clases.
# - 'binary_crossentropy' para clasificación binaria (dos clases).
LOSS_FUNCTION = 'categorical_crossentropy'  

# 📌 METRICS (Métricas de evaluación)
# Se usan para evaluar el rendimiento del modelo durante el entrenamiento.
# La más común es 'accuracy' (precisión), que mide cuántas predicciones son correctas.
METRICS = ['accuracy']  

# 📌 EARLY STOPPING (Detención temprana)
# Detiene el entrenamiento si la métrica de validación deja de mejorar después de varias épocas.
# Evita entrenar más de lo necesario y reduce el sobreajuste.
EARLY_STOPPING = True  

# 📌 DATA AUGMENTATION (Aumento de datos)
# Genera imágenes modificadas (rotaciones, zoom, etc.) para mejorar la generalización del modelo.
# Es útil cuando tienes pocos datos.
DATA_AUGMENTATION = True  

