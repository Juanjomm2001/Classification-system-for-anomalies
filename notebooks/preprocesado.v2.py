#2 reprocesado sin tener en cuenta el resize y todo eso.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
import glob
from tqdm.notebook import tqdm
import shutil
from collections import Counter

# Configuración inicial
np.random.seed(42)
tf.random.set_seed(42)

# Definir rutas
RAW_DATA_DIR = '../project3_claud/data/raw/'
TRAIN_DIR = '../project3_claud/data/train/'
VALIDATION_DIR = '../project3_claud/data/validation/'
TEST_DIR = '../project3_claud/data/test/'


# Modificar donde se crean los directorios
for directory in [TRAIN_DIR, VALIDATION_DIR, TEST_DIR]:  # Añadir TEST_DIR
    os.makedirs(directory, exist_ok=True)
    os.makedirs(os.path.join(directory, 'normal'), exist_ok=True)
    os.makedirs(os.path.join(directory, 'anomaly'), exist_ok=True)

# ---------------------------------------------
# 1. Análisis exploratorio del dataset
# ---------------------------------------------

def analyze_dataset(data_dir):
    """
    Analiza el conjunto de datos y devuelve estadísticas básicas.
    
    Args:
        data_dir: Directorio con las imágenes
    
    Returns:
        Un diccionario con las estadísticas
    """
    stats = {}
    
    for category in ['normal', 'anomaly']:
        category_dir = os.path.join(data_dir, category)
        if not os.path.exists(category_dir):
            print(f"El directorio {category_dir} no existe.")
            continue
        
        # Contar imágenes
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        image_paths = []
        
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(category_dir, ext)))
        
        num_images = len(image_paths)
        stats[f'{category}_count'] = num_images
        
        # Analizar tamaños y propiedades de las imágenes
        if num_images > 0:
            # Muestrear hasta 20 imágenes para calcular tamaños
            sample_size = min(20, num_images)
            sample_paths = np.random.choice(image_paths, sample_size, replace=False)
            
            image_sizes = []
            image_channels = []
            image_types = []
            
            for path in sample_paths:
                try:
                    img = cv2.imread(path)
                    if img is not None:
                        height, width = img.shape[:2]
                        channels = img.shape[2] if len(img.shape) > 2 else 1
                        image_sizes.append((height, width))
                        image_channels.append(channels)
                        image_types.append(img.dtype)
                except Exception as e:
                    print(f"Error al leer {path}: {e}")
            
            if image_sizes:
                stats[f'{category}_sizes'] = image_sizes
                stats[f'{category}_channels'] = image_channels
                stats[f'{category}_types'] = image_types
                
                # Calcular estadísticas
                heights, widths = zip(*image_sizes)
                stats[f'{category}_height_mean'] = np.mean(heights)
                stats[f'{category}_width_mean'] = np.mean(widths)
                stats[f'{category}_height_min'] = min(heights)
                stats[f'{category}_height_max'] = max(heights)
                stats[f'{category}_width_min'] = min(widths)
                stats[f'{category}_width_max'] = max(widths)
                
                # Obtener los tamaños más comunes
                stats[f'{category}_common_sizes'] = Counter(image_sizes).most_common(3)
    
    return stats

print("Analizando dataset original...")
dataset_stats = analyze_dataset(RAW_DATA_DIR)

# Mostrar estadísticas
total_images = dataset_stats.get('normal_count', 0) + dataset_stats.get('anomaly_count', 0)
print(f"\n--- Estadísticas del Dataset ---")
print(f"Total de imágenes: {total_images}")
print(f"- Imágenes normales: {dataset_stats.get('normal_count', 0)} ({dataset_stats.get('normal_count', 0)/total_images*100:.1f}%)")
print(f"- Imágenes con anomalías: {dataset_stats.get('anomaly_count', 0)} ({dataset_stats.get('anomaly_count', 0)/total_images*100:.1f}%)")

# Comprobar si hay desbalance
if min(dataset_stats.get('normal_count', 0), dataset_stats.get('anomaly_count', 0)) / max(dataset_stats.get('normal_count', 0), dataset_stats.get('anomaly_count', 0)) < 0.5:
    print("\n⚠️ ALERTA: El dataset está desbalanceado. Considera usar class_weights durante el entrenamiento.")

# Mostrar estadísticas de tamaño
print("\n--- Estadísticas de Tamaño ---")
for category in ['normal', 'anomaly']:
    if f'{category}_common_sizes' in dataset_stats:
        print(f"\n{category.capitalize()}:")
        print(f"- Dimensiones promedio: {dataset_stats[f'{category}_height_mean']:.0f}x{dataset_stats[f'{category}_width_mean']:.0f} píxeles")
        print(f"- Rango de altura: {dataset_stats[f'{category}_height_min']} - {dataset_stats[f'{category}_height_max']} píxeles")
        print(f"- Rango de anchura: {dataset_stats[f'{category}_width_min']} - {dataset_stats[f'{category}_width_max']} píxeles")
        print(f"- Tamaños más comunes: {dataset_stats[f'{category}_common_sizes']}")

# ---------------------------------------------
# 2. Visualización de ejemplos de imágenes
# ---------------------------------------------

def load_and_show_examples(data_dir, num_examples=3):
    """Carga y muestra ejemplos de imágenes normales y con anomalías."""
    
    fig = plt.figure(figsize=(15, 8))
    
    for i, category in enumerate(['normal', 'anomaly']):
        category_dir = os.path.join(data_dir, category)
        if not os.path.exists(category_dir):
            continue
            
        # Buscar todas las extensiones comunes
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_paths.extend(glob.glob(os.path.join(category_dir, ext)))
        
        # Seleccionar imágenes aleatorias
        if len(image_paths) > num_examples:
            image_paths = np.random.choice(image_paths, num_examples, replace=False)
        
        for j, path in enumerate(image_paths):
            try:
                img = cv2.imread(path)
                if img is not None:
                    # Convertir de BGR a RGB para matplotlib
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Mostrar imagen
                    ax = fig.add_subplot(2, num_examples, i * num_examples + j + 1)
                    ax.imshow(img_rgb)
                    ax.set_title(f"{category.capitalize()} - {img.shape}")
                    ax.axis('off')
            except Exception as e:
                print(f"Error al cargar {path}: {e}")
    
    plt.tight_layout()
    plt.show()

# Mostrar ejemplos de imágenes
print("\nCargando imágenes de ejemplo del dataset original...")
load_and_show_examples(RAW_DATA_DIR, num_examples=4)

# ---------------------------------------------
# 3. División en conjuntos de entrenamiento y validación
# ---------------------------------------------

def split_dataset(raw_dir, train_dir, val_dir, test_dir, train_ratio=0.7, val_ratio=0.15):
    """
    Divide el conjunto de datos en entrenamiento, validación y prueba.
    
    Args:
        raw_dir: Directorio con imágenes originales
        train_dir: Directorio para el conjunto de entrenamiento
        val_dir: Directorio para el conjunto de validación
        test_dir: Directorio para el conjunto de prueba
        train_ratio: Proporción del conjunto de entrenamiento (por defecto 70%)
        val_ratio: Proporción del conjunto de validación (por defecto 15%)
        (implícitamente, test_ratio = 1 - train_ratio - val_ratio)
    """
    split_stats = {'train': {}, 'validation': {}, 'test': {}}
    
    # Procesar cada categoría (normal y anomalía)
    for category in ['normal', 'anomaly']:
        # Obtener la ruta de las imágenes de la categoría
        image_paths = glob.glob(os.path.join(raw_dir, category, '*.jpg')) + \
                     glob.glob(os.path.join(raw_dir, category, '*.png')) + \
                     glob.glob(os.path.join(raw_dir, category, '*.jpeg')) + \
                     glob.glob(os.path.join(raw_dir, category, '*.JPG')) + \
                     glob.glob(os.path.join(raw_dir, category, '*.PNG')) + \
                     glob.glob(os.path.join(raw_dir, category, '*.JPEG'))
        
        if not image_paths:
            print(f"No se encontraron imágenes para la categoría {category}")
            continue
        
        # Barajar las imágenes para asegurar distribución aleatoria
        np.random.shuffle(image_paths)
        
        # Calcular cantidad para cada conjunto
        total_images = len(image_paths)
        train_size = int(total_images * train_ratio)
        val_size = int(total_images * val_ratio)
        
        # Dividir en tres conjuntos
        train_paths = image_paths[:train_size]
        val_paths = image_paths[train_size:train_size + val_size]
        test_paths = image_paths[train_size + val_size:]
        
        split_stats['train'][category] = len(train_paths)
        split_stats['validation'][category] = len(val_paths)
        split_stats['test'][category] = len(test_paths)
        
        # Copiar imágenes a los directorios correspondientes
        print(f"Copiando imágenes de {category}...")
        
        # Entrenamiendo
        for path in tqdm(train_paths):
            dest_path = os.path.join(train_dir, category, os.path.basename(path))
            shutil.copy(path, dest_path)
        
        # Validación
        for path in tqdm(val_paths):
            dest_path = os.path.join(val_dir, category, os.path.basename(path))
            shutil.copy(path, dest_path)
            
        # Prueba (nuevo)
        for path in tqdm(test_paths):
            dest_path = os.path.join(test_dir, category, os.path.basename(path))
            shutil.copy(path, dest_path)
    
    return split_stats

# Dividir el conjunto de datos en tres partes
print("\nDividiendo el dataset en entrenamiento, validación y prueba...")
split_stats = split_dataset(RAW_DATA_DIR, TRAIN_DIR, VALIDATION_DIR, TEST_DIR, 
                          train_ratio=0.7, val_ratio=0.15)

# Mostrar resultados de la división
print("\n--- Resultados de la División ---")
print("Conjunto de entrenamiento:")
print(f"- Normal: {split_stats['train'].get('normal', 0)} imágenes")
print(f"- Anomalía: {split_stats['train'].get('anomaly', 0)} imágenes")
print(f"- Total: {sum(split_stats['train'].values())} imágenes")

print("\nConjunto de validación:")
print(f"- Normal: {split_stats['validation'].get('normal', 0)} imágenes")
print(f"- Anomalía: {split_stats['validation'].get('anomaly', 0)} imágenes")
print(f"- Total: {sum(split_stats['validation'].values())} imágenes")

print("\nConjunto de prueba:")  # Nuevo
print(f"- Normal: {split_stats['test'].get('normal', 0)} imágenes")
print(f"- Anomalía: {split_stats['test'].get('anomaly', 0)} imágenes")
print(f"- Total: {sum(split_stats['test'].values())} imágenes")

# ---------------------------------------------
# 4. Guardar configuración para fine-tuning
# ---------------------------------------------

# Calcular parámetros para el entrenamiento
BATCH_SIZE = 32
IMG_SIZE = (224, 224)  # Tamaño común para modelos preentrenados

# Calcular pasos por época
train_total = sum(split_stats['train'].values())
val_total = sum(split_stats['validation'].values())

steps_per_epoch = train_total // BATCH_SIZE
validation_steps = val_total // BATCH_SIZE

# Crear un diccionario con la configuración para fine-tuning
# En la parte donde creas el diccionario de configuración
config = {
    'image_size': IMG_SIZE,
    'batch_size': BATCH_SIZE,
    'train_samples': sum(split_stats['train'].values()),
    'validation_samples': sum(split_stats['validation'].values()),
    'test_samples': sum(split_stats['test'].values()),  # Nuevo
    'steps_per_epoch': steps_per_epoch,
    'validation_steps': validation_steps,
    'class_indices': {'anomaly': 0, 'normal': 1},
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
os.makedirs('../project3_claud/data/processed/', exist_ok=True)
config_path = '../project3_claud/data/processed/data_config.json'

with open(config_path, 'w') as f:
    json.dump(config, f, indent=4)

print(f"\nConfiguración guardada en {config_path}")

# Mostrar ejemplos del conjunto de entrenamiento y validación
print("\nMostrando ejemplos del conjunto de entrenamiento:")
load_and_show_examples(TRAIN_DIR, num_examples=3)

print("\nMostrando ejemplos del conjunto de validación:")
load_and_show_examples(VALIDATION_DIR, num_examples=3)

print("\nPreprocesamiento completado con éxito!")