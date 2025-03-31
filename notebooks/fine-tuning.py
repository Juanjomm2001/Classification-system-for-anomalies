#https://docs.neptune.ai/integrations/tensorflow/   para el tema de monitorizar online
#https://optuna.org/  para el tema de los mejores hyperparametrosa

# 3. Fine-Tuning del Modelo de Detección de Anomalías
# =============================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, ResNet50, EfficientNetB0
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import datetime
import json

# # Configuración inicial
# %matplotlib inline
# plt.style.use('seaborn-whitegrid')
# np.random.seed(42)
# tf.random.set_seed(42)

# Definir rutas
TRAIN_DIR = '../project3_claud/data/train/'
VALIDATION_DIR = '../project3_claud/data/validation/'
PROCESSED_DIR = '../project3_claud/data/processed/'
MODELS_DIR = '../project3_claud/models/'
FINE_TUNED_DIR = os.path.join(MODELS_DIR, 'fine_tuned')

# Crear directorios si no existen
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FINE_TUNED_DIR, exist_ok=True)

# # 3.1 Cargar configuración
# # ---------------------------------------------

def load_data_config(config_path):
    """Carga la configuración de datos desde un archivo JSON."""
    with open(config_path, 'r') as f:
        return json.load(f)

# Cargar configuración
config_path = os.path.join(PROCESSED_DIR, 'data_config.json')
if os.path.exists(config_path):
    config = load_data_config(config_path)
    print("Configuración cargada desde:", config_path)
    
    # Extraer parámetros de configuración
    IMG_SIZE = tuple(config['image_size'])
    BATCH_SIZE = config['batch_size']
    steps_per_epoch = config['steps_per_epoch']
    validation_steps = config['validation_steps']
    class_indices = config['class_indices']
    print("Configuración cargada:")
    print(f"Tamaño de imagen: {IMG_SIZE}")
    print(f"Tamaño de lote: {BATCH_SIZE}")
    print(f"Pasos por época: {steps_per_epoch}")
    print(f"Índices de clase: {class_indices}")
else:
    print("Archivo de configuración no encontrado. Usando valores predeterminados.")
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    # Calcular parámetros basados en el directorio
    train_samples = len(os.listdir(os.path.join(TRAIN_DIR, 'normal'))) + \
                   len(os.listdir(os.path.join(TRAIN_DIR, 'anomaly')))
    validation_samples = len(os.listdir(os.path.join(VALIDATION_DIR, 'normal'))) + \
                        len(os.listdir(os.path.join(VALIDATION_DIR, 'anomaly')))
    steps_per_epoch = train_samples // BATCH_SIZE
    validation_steps = validation_samples // BATCH_SIZE
    class_indices = {'normal': 0, 'anomaly': 1}

# # 3.2 Preparar generadores de datos
# # ---------------------------------------------

# Generador para entrenamiento con aumento de datos
train_datagen = ImageDataGenerator(
    #rescale=1./255,    si lo has hehco ya ante sno lo hagas otra vez
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Generador para validación (solo rescale)
val_datagen = ImageDataGenerator(
    #rescale=1./255  si ya lo has heho no lo hagas otra vez 
)
# Preparar los generadores de flujo de datos
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True
)

validation_generator = val_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# # 3.3 Definir y construir el modelo base
# # ---------------------------------------------

def build_model(base_model_name='EfficientNetB0', input_shape=(224, 224, 3), 
                trainable_base=False, dropout_rate=0.3):
    """
    Construye un modelo para detección de anomalías basado en redes pre-entrenadas.
    
    Args:
        base_model_name: Nombre del modelo base ('MobileNetV2', 'ResNet50', o 'EfficientNetB0')
        input_shape: Forma de la entrada (altura, ancho, canales)
        trainable_base: Si es True, hace que las capas base sean entrenables
        dropout_rate: Tasa de dropout para regularización
    
    Returns:
        Modelo compilado
    """
    # Crear el modelo base
    if base_model_name == 'MobileNetV2':
        base_model = MobileNetV2(weights='imagenet', include_top=False, 
                               input_shape=input_shape)
    elif base_model_name == 'ResNet50':
        base_model = ResNet50(weights='imagenet', include_top=False, 
                            input_shape=input_shape)
    elif base_model_name == 'EfficientNetB0':
        base_model = EfficientNetB0(weights='imagenet', include_top=False, 
                                  input_shape=input_shape)
    else:
        raise ValueError(f"Modelo base no soportado: {base_model_name}")
    
    # Congelar o descongelar las capas base
    base_model.trainable = trainable_base
    
    # Añadir capas superiores personalizadas
    inputs = Input(shape=input_shape) #entrada del modelo
    x = base_model(inputs, training=False) # modelo preentrenaod como feature extraction
    x = GlobalAveragePooling2D()(x) #Convierte los mapas de características 3D (altura × ancho × canales) a vectores 1D,  Reduce el número de parámetros, evita el sobreajuste y hace que el modelo sea invariante a pequeñas transformaciones espaciales.
    x = BatchNormalization()(x) #normaliza la capa anterior 
    x = Dropout(dropout_rate)(x) #desactiva neuronas por el sobreajuste 
    x = Dense(256, activation='relu')(x) #saca las 256 principales características  
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    
    # Compilar el modelo
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), 
                 tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
    )
    
    return model

# Construir el modelo con EfficientNetB0 como base
print("Construyendo modelo base con EfficientNetB0...")
model = build_model(base_model_name='EfficientNetB0', 
                   input_shape=IMG_SIZE + (3,),
                   trainable_base=False)

# Mostrar el resumen del modelo
model.summary()

# 3.4 Definir callbacks para el entrenamiento
# ---------------------------------------------

# Directorio para logs de TensorBoard
log_dir = os.path.join("../logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(log_dir, exist_ok=True)

# Definir callbacks
callbacks = [
    # Guardar el mejor modelo
    ModelCheckpoint(
        filepath=os.path.join(FINE_TUNED_DIR, 'best_model.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    # Detener el entrenamiento si no hay mejora
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    # Reducir la tasa de aprendizaje cuando el rendimiento se estanca
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        verbose=1
    ),
    # TensorBoard para monitoreo visual
    TensorBoard(log_dir=log_dir, histogram_freq=1)  
    #guspisimo esto es para que te monitorice el eentrenamiento 

#     🛠 ¿Cómo visualizar los datos en TensorBoard?
# Durante o después del entrenamiento, abre una terminal y ejecuta:


# tensorboard --logdir=../logs
# Luego, abre en el navegador:


# http://localhost:6006/
# Ahí podrás ver gráficas detalladas del entrenamiento. 📊    
]

# 3.5 Entrenar el modelo (primera fase - solo capas superiores)
# ---------------------------------------------

# Número de épocas para entrenamiento
EPOCHS = 20

# print("Iniciando entrenamiento de capas superiores...")
# history = model.fit(
#     train_generator,
#     steps_per_epoch=steps_per_epoch,
#     epochs=2,
#     validation_data=validation_generator,
#     validation_steps=validation_steps,
#     callbacks=callbacks,
#     verbose=1
# )

# # 3.6 Fine-tuning (segunda fase - incluir algunas capas del modelo base)
# # ---------------------------------------------

# print("Iniciando fine-tuning con algunas capas del modelo base...")

# # Guardar los pesos entrenados hasta ahora
# model.save(os.path.join(FINE_TUNED_DIR, 'model_phase1.h5'))

# # Ahora vamos a descongelar algunas capas del modelo base
# if isinstance(model.layers[1], tf.keras.Model):  # Si la capa base es un modelo
#     base_model = model.layers[1]
    
#     # Congelar las primeras capas y descongelar las últimas
#     # Para EfficientNetB0, que tiene múltiples bloques
#     # Descongelamos solo los últimos bloques
#     for layer in base_model.layers:
#         layer.trainable = False
    
#     # Descongelar los últimos 20 layers
#     for layer in base_model.layers[-20:]:
#         layer.trainable = True
    
#     # Usar una tasa de aprendizaje más baja para fine-tuning
#     model.compile(
#         optimizer=Adam(learning_rate=1e-5),
#         loss='binary_crossentropy',
#         metrics=['accuracy', tf.keras.metrics.Precision(), 
#                  tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
#     )
    
#     # Mostrar el modelo actualizado
#     model.summary()
    
#     # Entrenar con fine-tuning
#     history_fine = model.fit(
#         train_generator,
#         steps_per_epoch=steps_per_epoch,
#         epochs=10,  # Menos épocas para fine-tuning
#         validation_data=validation_generator,
#         validation_steps=validation_steps,
#         callbacks=callbacks,
#         verbose=1
#     )
    
#     # Combinar historiales de entrenamiento
#     total_history = {}
#     for k in history.history.keys():
#         if k in history_fine.history:
#             total_history[k] = history.history[k] + history_fine.history[k]
# else:
#     print("No se pudo realizar fine-tuning en el modelo base")
#     total_history = history.history

# # 3.7 Visualizar resultados del entrenamiento
# # ---------------------------------------------
# Modificación para asegurar que el historial de entrenamiento se guarde
# Añade esto después de la sección 3.6 o modifica tu sección 3.7

# 3.7 Visualizar resultados del entrenamiento
# ---------------------------------------------

def plot_training_history(history_obj, metrics=['accuracy', 'loss']):
    """
    Visualiza las métricas de entrenamiento.
    
    Args:
        history_obj: Historial de entrenamiento o diccionario con las métricas
        metrics: Lista de métricas para visualizar
    """
    plt.figure(figsize=(15, 10))
    
    # Verificar el tipo de objeto historial
    if hasattr(history_obj, 'history'):
        # Si es un objeto History de Keras
        history_dict = history_obj.history
    else:
        # Si ya es un diccionario
        history_dict = history_obj
    
    print("Métricas disponibles:", list(history_dict.keys()))
    
    for i, metric in enumerate(metrics):
        if metric in history_dict:
            plt.subplot(2, 2, i+1)
            plt.plot(history_dict[metric], label=f'Training {metric}')
            
            # Verificar si existe la métrica de validación
            val_metric = f'val_{metric}'
            if val_metric in history_dict:
                plt.plot(history_dict[val_metric], label=f'Validation {metric}')
            
            plt.title(f'{metric.capitalize()} Over Time')
            plt.xlabel('Epoch')
            plt.ylabel(metric.capitalize())
            plt.legend()
            plt.grid(True)
        else:
            print(f"Advertencia: Métrica '{metric}' no encontrada en el historial")
    
    plt.tight_layout()
    plt.show()

# Intentar visualizar el historial de la primera fase
try:
    print("Visualizando historial de primera fase...")
    plot_training_history(history)
except NameError:
    print("El historial de la primera fase no está disponible.")

# Intentar visualizar el historial de la segunda fase
try:
    print("Visualizando historial de fine-tuning...")
    plot_training_history(history_fine)
except NameError:
    print("El historial de fine-tuning no está disponible.")

# Intentar visualizar el historial combinado
try:
    print("Visualizando historial combinado...")
    plot_training_history(total_history)
except NameError:
    print("El historial combinado no está disponible.")

# Como último recurso, podríamos intentar cargar el modelo y evaluar su rendimiento
print("Evaluando el modelo final guardado...")
try:
    from tensorflow.keras.models import load_model
    
    # Intentar cargar el modelo final
    final_model_path = os.path.join(FINE_TUNED_DIR, 'best_model.h5')
    if os.path.exists(final_model_path):
        final_model = load_model(final_model_path)
        
        # Evaluar en conjunto de validación
        evaluation = final_model.evaluate(validation_generator, verbose=1)
        metrics_names = final_model.metrics_names
        
        print("\nEvaluación del modelo final:")
        for name, value in zip(metrics_names, evaluation):
            print(f"{name}: {value:.4f}")
    else:
        print(f"Modelo final no encontrado en {final_model_path}")
except Exception as e:
    print(f"Error al evaluar el modelo: {e}")
# # 3.8 Guardar el modelo final
# # ---------------------------------------------

# # Guardar el modelo completo
# model.save(os.path.join(FINE_TUNED_DIR, 'final_model.h5'))

# # Guardar también en formato TensorFlow SavedModel para implementación
# model.save(os.path.join(FINE_TUNED_DIR, 'saved_model'))

# # Guardar los mapeos de clase
# class_indices = train_generator.class_indices
# with open(os.path.join(FINE_TUNED_DIR, 'class_indices.json'), 'w') as f:
#     json.dump(class_indices, f)

# print(f"Modelo guardado en {os.path.join(FINE_TUNED_DIR, 'final_model.h5')}")
# print(f"Modelo SavedModel guardado en {os.path.join(FINE_TUNED_DIR, 'saved_model')}")
# print(f"Índices de clase guardados en {os.path.join(FINE_TUNED_DIR, 'class_indices.json')}")

# # Función de utilidad para recargar el modelo
# def load_trained_model(model_path, custom_objects=None):
#     """
#     Carga el modelo entrenado desde un archivo .h5 o directorio SavedModel.
    
#     Args:
#         model_path: Ruta al modelo guardado
#         custom_objects: Diccionario de objetos personalizados (si es necesario)
    
#     Returns:
#         Modelo cargado
#     """
#     try:
#         model = load_model(model_path, custom_objects=custom_objects)
#         print(f"Modelo cargado desde {model_path}")
#         return model
#     except Exception as e:
#         print(f"Error al cargar el modelo: {e}")
#         return None

# # Ejemplo de uso
# # model = load_trained_model(os.path.join(FINE_TUNED_DIR, 'final_model.h5'))

# print("Fine-tuning completado con éxito!")