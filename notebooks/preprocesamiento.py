# 2. Preprocesamiento de Datos () FAKE, este hace el resize y todo y te guarda las iamgenes, no renta mucho
# =============================================


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import glob
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
import shutil

# # Configuración inicial
# #%matplotlib inline
# plt.style.use('seaborn-whitegrid')
# np.random.seed(42)
# tf.random.set_seed(42)

# Definir rutas
RAW_DATA_DIR = '../project3_claud/data/raw/'
PROCESSED_DIR = '../project3_claud/data/processed/'
TRAIN_DIR = '../project3_claud/data/train/'
VALIDATION_DIR = '../project3_claud/data/validation/'

# Crear directorios si no existen
for directory in [PROCESSED_DIR, TRAIN_DIR, VALIDATION_DIR]:
    os.makedirs(directory, exist_ok=True)
    os.makedirs(os.path.join(directory, 'normal'), exist_ok=True)
    os.makedirs(os.path.join(directory, 'anomaly'), exist_ok=True)

# 2.1 Preprocesamiento de Imágenes
# ---------------------------------------------

def preprocess_images(input_dir, output_dir, target_size=(224, 224), normalize=True):   # Le pasas el directorio de entrada, el directorio de salida, el tamaño de la imagen y si normalizar o no
    """
    Preprocesa imágenes: redimensiona y normaliza opcionalmente.
    
    Args:
        input_dir: Directorio con imágenes originales
        output_dir: Directorio para guardar imágenes procesadas
        target_size: Tamaño objetivo (ancho, alto)
        normalize: Si es True, normaliza los valores de píxeles a [0,1]´


    Requisito de las arquitecturas de redes neuronales: Las redes neuronales convolucionales como 
    EfficientNet, ResNet o MobileNet están diseñadas para recibir imágenes de un tamaño específico como entrada.
        EfficientNetB0: 224x224
        MobileNetV2: 224x224
        ResNet50: 224x224
        InceptionV3: 299x299
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Obtener todas las imágenes
    image_paths = glob.glob(os.path.join(input_dir, '*.jpg')) + \
                 glob.glob(os.path.join(input_dir, '*.png'))
    
    print(f"Procesando {len(image_paths)} imágenes de {input_dir}...")
    
    for img_path in tqdm(image_paths): #esto es para crear como un abarra de carga en el terminal
        try:
            # Cargar imagen
            img = cv2.imread(img_path)
            if img is None:
                print(f"Error al cargar la imagen {img_path}. Omitiendo.")
                continue

            # Redimensionar
            img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
            
            # Normalizar si es necesario SI ES FALSE NO LO HACE 
            if normalize:
                img_resized = img_resized.astype(np.float32) / 255.0
                img_resized = (img_resized * 255).astype(np.uint8)  # Reconvertir a uint8 para guardar
            
            # Guardar imagen procesada
            output_path = os.path.join(output_dir, os.path.basename(img_path))
            cv2.imwrite(output_path, img_resized)
            
        except Exception as e:
            print(f"Error al procesar {img_path}: {e}")

# Procesar imágenes normales y con anomalías
preprocess_images(os.path.join(RAW_DATA_DIR, 'normal'), 
                 os.path.join(PROCESSED_DIR, 'normal'), target_size=(224, 224), normalize=False)

preprocess_images(os.path.join(RAW_DATA_DIR, 'anomaly'), 
                 os.path.join(PROCESSED_DIR, 'anomaly'), target_size=(224, 224), normalize=False)

def check_processed_image_sizes(processed_dir):
    image_paths = glob.glob(os.path.join(processed_dir, '*.jpg')) + \
                 glob.glob(os.path.join(processed_dir, '*.png'))
    
    # Cargar imágenes para verificar el tamaño
    img = cv2.imread(image_paths[0])   
    # height, width = img.shape[:2]
    # print(f"Tamaño de las imágenes: {width}x{height} píxeles")
    print("Tamaño de las imágenes:píxeles"+ str(img.shape))

# Verificar el tamaño de las imágenes procesadas
check_processed_image_sizes(os.path.join(PROCESSED_DIR, 'normal'))
check_processed_image_sizes(os.path.join(PROCESSED_DIR, 'anomaly'))


# 2.2 División en conjuntos de entrenamiento y validación
# ---------------------------------------------

def split_dataset(processed_dir, train_dir, val_dir, split_ratio=0.2):  #esto signinfica que el 0.2 es el 20% de las imágenes que se usan para validación
    """
    Divide el conjunto de datos en entrenamiento y validación.
    
    Args:
        processed_dir: Directorio con imágenes procesadas
        train_dir: Directorio para el conjunto de entrenamiento
        val_dir: Directorio para el conjunto de validación
        split_ratio: Proporción del conjunto de validación
    """
    # Procesar cada categoría (normal y anomalía)
    for category in ['normal', 'anomaly']:
        # Obtener  la ruta de las imágenes de la categoría
        image_paths = glob.glob(os.path.join(processed_dir, category, '*.jpg')) + \
                     glob.glob(os.path.join(processed_dir, category, '*.png'))
        #print(image_paths)
        # Dividir en entrenamiento y validación
        train_paths, val_paths = train_test_split(   #esto ya existe la funcion esta 
            image_paths, test_size=split_ratio, random_state=42
        )
        
        print(f"Categoría {category}:")
        print(f"  - Imágenes de entrenamiento: {len(train_paths)}")
        print(f"  - Imágenes de validación: {len(val_paths)}")
        
        # Copiar imágenes a los directorios correspondientes
        for path in train_paths:
            dest_path = os.path.join(train_dir, category, os.path.basename(path))
            shutil.copy(path, dest_path)
        
        for path in val_paths:
            dest_path = os.path.join(val_dir, category, os.path.basename(path))
            shutil.copy(path, dest_path)

# Dividir el conjunto de datos
split_dataset(PROCESSED_DIR, TRAIN_DIR, VALIDATION_DIR, split_ratio=0.2)

# 2.3 Aumento de datos (Data Augmentation)
# ---------------------------------------------

# Definir el generador de datos para aumento de entrenamiento
train_datagen = ImageDataGenerator(
    #rescale=1./255,     YA LO HICISTE ANTES            # Escala los valores de píxeles de [0-255] a [0-1]
    rotation_range=20,              # Rota la imagen aleatoriamente hasta 20 grados
    width_shift_range=0.1,          # Desplaza horizontalmente la imagen hasta 10% del ancho
    height_shift_range=0.1,         # Desplaza verticalmente la imagen hasta 10% del alto
    shear_range=0.2,                # Aplica transformación de cizallamiento (shear) hasta 0.2 radianes
    zoom_range=0.2,                 # Hace zoom aleatorio entre 80% y 120% del tamaño original
    horizontal_flip=True,           # Voltea la imagen horizontalmente con 50% de probabilidad
    fill_mode='nearest'             # Rellena píxeles nuevos con el valor más cercano cuando hay rotaciones o desplazamientos
)
# Generador para validación (solo rescale ya que quieres evaluar con lo mas normal a lo que luego te vas a encontrar)
val_datagen = ImageDataGenerator(rescale=1./255)

# Preparar los generadores de flujo de datos
BATCH_SIZE = 32
IMG_SIZE = (224, 224)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,                # Directorio donde están las imágenes de entrenamiento
    target_size=IMG_SIZE,     # Tamaño al que se redimensionarán las imágenes (ej: (224, 224))
    batch_size=BATCH_SIZE,    # Número de imágenes procesadas en cada lote (ej: 32)
    class_mode='binary',      # Tipo de problema (binario: normal vs anomalía)
    shuffle=True              # Mezclar imágenes aleatoriamente en cada época
)

validation_generator = val_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# # Función para cargar algunas imágenes para visualización
# def load_sample_images(directory, n_samples=3):
#     """Carga algunas imágenes de muestra para visualización."""
#     images = []
#     labels = []
#     paths = []
    
#     for category in ['normal', 'anomaly']:
#         category_dir = os.path.join(directory, category)            
#         # Buscar todas las extensiones comunes
#         image_paths = []
#         for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
#             image_paths.extend(glob.glob(os.path.join(category_dir, ext)))
        
#         # Tomar las primeras n_samples imágenes
#         samples = image_paths[:n_samples]
        
#         for path in samples:
#             try:
#                 img = cv2.imread(path)
#                 if img is None:
#                     print(f"Error al cargar {path}: La imagen es None")
#                     continue
                    
#                 img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#                 images.append(img)
#                 labels.append(category)
#                 paths.append(path)
#                 print(f"Cargada imagen: {os.path.basename(path)}")
#             except Exception as e:
#                 print(f"Error al cargar {path}: {e}")
    
#     return images, labels, paths

# # Función mejorada para visualizar el aumento de datos
# def visualize_augmentation(datagen, images, labels, n_samples=5):
#     """
#     Visualiza ejemplos de aumento de datos.
    
#     Args:
#         datagen: Generador de ImageDataGenerator
#         images: Lista de imágenes originales
#         labels: Lista de etiquetas correspondientes
#         n_samples: Número de muestras aumentadas por imagen
#     """
#     if not images:
#         print("No hay imágenes para visualizar.")
#         return
        
#     plt.figure(figsize=(15, min(3, len(images)) * 3))
    
#     for i, (img, label) in enumerate(zip(images[:3], labels[:3])):
#         # Asegurarse de que la imagen tenga el tamaño correcto
#         print("Size de l aiamgen original; ", img.shape)
#         if img.shape[:2] != IMG_SIZE:
#             print(f"Redimensionando imagen de {img.shape[:2]} a {IMG_SIZE}")
#             img_resized = cv2.resize(img, IMG_SIZE)
            
#         else:
#             img_resized = img.copy()
        
#         # Verificar rango de valores
#         min_val, max_val = img_resized.min(), img_resized.max()
#         print(f"Imagen {i+1}: rango de valores [{min_val}, {max_val}]")
        
#         # Convertir a formato esperado por ImageDataGenerator
#         x = np.expand_dims(img_resized, 0)  # Añadir dimensión de batch
        
#         # Normalizar si los valores están en rango [0-255]
#         if max_val > 1.0:
#             x = x.astype('float32') / 255.
#             print(f"  Normalizado al rango [0,1]")
        
#         # Generar muestras aumentadas
#         augmented_iter = datagen.flow(x, batch_size=1)
        
#         # Mostrar imagen original
#         plt.subplot(min(3, len(images)), n_samples + 1, i * (n_samples + 1) + 1)
#         plt.imshow(img_resized)
#         plt.title(f"Original ({label})")
#         plt.axis('off')
        
#         # Mostrar muestras aumentadas
#         for j in range(n_samples):
#             batch = next(augmented_iter)
#             aug_img = batch[0]
            
#             # Verificar rango de la imagen aumentada
#             aug_min, aug_max = aug_img.min(), aug_img.max()
#             print(f"  Aumentada {j+1}: rango [{aug_min}, {aug_max}]")
            
#             # Preparar para visualización
#             if aug_max <= 1.0:
#                 aug_display = (aug_img * 255).astype('uint8')
#             else:
#                 aug_display = aug_img.astype('uint8')
            
#             plt.subplot(min(3, len(images)), n_samples + 1, i * (n_samples + 1) + j + 2)
#             plt.imshow(aug_display)
#             plt.title(f"Aumentada {j+1}")
#             plt.axis('off')
    
#     plt.tight_layout()
#     plt.show()

# # Cargar y visualizar imágenes de muestra
# print("\nCargando imágenes de muestra para visualización de aumento de datos...")
# sample_images, sample_labels, sample_paths = load_sample_images(PROCESSED_DIR)

# if sample_images:
#     print(f"\nVisualizando aumento de datos con {len(sample_images)} imágenes...")
#     visualize_augmentation(train_datagen, sample_images, sample_labels)
# else:
#     print("No se pudieron cargar imágenes para visualización.")

# # 2.4 Preparación de datos para el entrenamiento
# # ---------------------------------------------

# Calcular el número de pasos por época
steps_per_epoch = train_generator.samples // BATCH_SIZE
validation_steps = validation_generator.samples // BATCH_SIZE

print(f"Configuración de entrenamiento:")
print(f"- Tamaño de lote (batch size): {BATCH_SIZE}") #Es la cantidad de imágenes que se procesan en cada iteración antes de actualizar los pesos del modelo.
print(f"- Tamaño de imagen: {IMG_SIZE}") #Es el tamaño de las imágenes que se usan para entrenar y validar el modelo.
print(f"- Pasos por época: {steps_per_epoch}") #Es el número de veces que el modelo se entrena en el conjunto de entrenamiento.
# ¿Cómo afecta?
# ✅ Si es muy grande

# Se realizan más pasos por cada época, lo que aumenta el tiempo de entrenamiento.

# Puede mejorar la precisión si tienes suficientes datos.

# ✅ Si es muy pequeño

# El modelo ve menos imágenes por época, lo que puede generar un sobreajuste.
print(f"- Pasos de validación: {validation_steps}") #Es el número de veces que el modelo se evalúa en el conjunto de validación.
print(f"- Clases: {train_generator.class_indices}") #Es un diccionario que mapea las etiquetas de las imágenes a números enteros.

# 2.5 Guardar configuración y metadatos
# ---------------------------------------------

# Crear un diccionario con la configuración
config = {
    'image_size': IMG_SIZE,
    'batch_size': BATCH_SIZE,
    'train_samples': train_generator.samples,
    'validation_samples': validation_generator.samples,
    'steps_per_epoch': steps_per_epoch,
    'validation_steps': validation_steps,
    'class_indices': train_generator.class_indices,
    'data_augmentation': {
        'rotation_range': 20,
        'width_shift_range': 0.1,
        'height_shift_range': 0.1,
        'shear_range': 0.2,
        'zoom_range': 0.2,
        'horizontal_flip': True,
        'fill_mode': 'nearest'
    }
}

# Guardar la configuración como JSON
import json

with open(os.path.join(PROCESSED_DIR, 'data_config.json'), 'w') as f:
    json.dump(config, f, indent=4)

print("Configuración guardada en:", os.path.join(PROCESSED_DIR, 'data_config.json'))
print("Preprocesamiento completado con éxito!")

# Función de utilidad para cargar la configuración posteriormente
def load_data_config(config_path):
    """Carga la configuración de datos desde un archivo JSON."""
    with open(config_path, 'r') as f:
        return json.load(f)

# Ejemplo de uso
# config = load_data_config(os.path.join(PROCESSED_DIR, 'data_config.json'))
