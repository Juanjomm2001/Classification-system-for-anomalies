# 4. Evaluación y Prueba del Modelo
# =============================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
import seaborn as sns
import cv2
import glob
import json
from tqdm.notebook import tqdm
import itertools

# # Configuración inicial
# %matplotlib inline
# plt.style.use('seaborn-whitegrid')
# np.random.seed(42)
# tf.random.set_seed(42)

# Definir rutas
VALIDATION_DIR = '../project3_claud/data/validation/'
FINE_TUNED_DIR = '../project3_claud/models/fine_tuned/'
RESULTS_DIR = '../project3_claud/models/evaluation_results/'

# Crear directorio para resultados si no existe
os.makedirs(RESULTS_DIR, exist_ok=True)

# 4.1 Cargar el modelo entrenado
# ---------------------------------------------

def load_trained_model(model_path):
    """
    Carga el modelo entrenado desde un archivo .h5 o directorio SavedModel.
    
    Args:
        model_path: Ruta al modelo guardado
    
    Returns:
        Modelo cargado
    """
    try:
        model = load_model(model_path)
        print(f"Modelo cargado desde {model_path}")
        return model
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return None

# Cargar el mejor modelo
model_path = os.path.join(FINE_TUNED_DIR, 'best_model.h5')
model = load_trained_model(model_path)

if model is None:
    print("Intentando cargar el modelo final...")
    
if model is None:
    print("No se pudo cargar el modelo. Verifique las rutas y la existencia de los archivos.")
    import sys
    sys.exit(1)

# Cargar índices de clase
class_indices_path = os.path.join(FINE_TUNED_DIR, 'class_indices.json')
if os.path.exists(class_indices_path):
    with open(class_indices_path, 'r') as f:
        class_indices = json.load(f)
    # Invertir el diccionario para obtener etiquetas a partir de índices
    idx_to_class = {v: k for k, v in class_indices.items()}
else:
    print("Archivo de índices de clase no encontrado. Usando valores predeterminados.")
    class_indices = {'normal': 0, 'anomaly': 1}
    idx_to_class = {0: 'normal', 1: 'anomaly'}

print("Índices de clase:", class_indices)

                                                    # def predict_with_custom_threshold(model, image_path, threshold=0.35):
                                                    #     """
                                                    #     Predice la clase de una imagen con un umbral personalizado.
                                                        
                                                    #     Args:
                                                    #         model: Modelo entrenado
                                                    #         image_path: Ruta a la imagen
                                                    #         threshold: Umbral de decisión (default: 0.35)
                                                        
                                                    #     Returns:
                                                    #         Clase predicha y probabilidad
                                                    #     """
                                                    #     # Cargar y preprocesar imagen
                                                    #     img = cv2.imread(image_path)
                                                    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                                    #     img = cv2.resize(img, (224, 224))  # Ajustar al tamaño de entrada del modelo
                                                        
                                                    #     # Normalizar
                                                    #     img = img.astype(np.float32) / 255.0
                                                        
                                                    #     # Ampliar dimensiones para batch
                                                    #     img_array = np.expand_dims(img, axis=0)
                                                        
                                                    #     # Predecir
                                                    #     prediction = model.predict(img_array)[0][0]
                                                        
                                                    #     # Aplicar umbral personalizado
                                                    #     class_name = "anomaly" if prediction > threshold else "normal"
                                                        
                                                    #     return class_name, prediction

                                                    # class_name, prediction = predict_with_custom_threshold(model, '../project3_claud/data/validation/anomaly/1_nano-0000050_jpg.rf.63101af74405d8a22c5fa9816a60ec8a.jpg', threshold=0.35)
                                                    # print(f"Clase: {class_name}, Probabilidad: {prediction:.4f}")
# # # 4.2 Preparar datos de validación
# # # ---------------------------------------------

# # Determinar tamaño de imagen basado en la entrada del modelo
# input_shape = model.input_shape[1:3]
# print(f"Tamaño de entrada del modelo: {input_shape}")

# # Generador para datos de validación
# val_datagen = ImageDataGenerator()

# validation_generator = val_datagen.flow_from_directory(
#     VALIDATION_DIR,
#     target_size=input_shape,
#     batch_size=1,  # Tamaño de lote 1 para procesar imágenes una por una
#     class_mode='binary',
#     shuffle=False   # Importante: No mezclar para mantener el orden
# )

# # 4.3 Evaluación del modelo
# # ---------------------------------------------

# # Evaluar el modelo en el conjunto de validación
# print("Evaluando modelo en conjunto de validación...")
# evaluation = model.evaluate(validation_generator, verbose=1)

# # Mostrar resultados de la evaluación
# metrics = model.metrics_names
# evaluation_results = dict(zip(metrics, evaluation))
# print("\nResultados de la evaluación:")
# for metric, value in evaluation_results.items():
#     print(f"{metric}: {value:.4f}")

# # 4.4 Predicciones y métricas detalladas
# # ---------------------------------------------

# # Obtener predicciones para todas las imágenes de validación
# print("Generando predicciones para todas las imágenes...")

# # Restablecer el generador
# validation_generator.reset()

# # Número total de muestras
# n_samples = validation_generator.samples

# # Arreglos para almacenar predicciones y etiquetas reales
# y_true = np.zeros(n_samples, dtype=int)
# y_pred = np.zeros(n_samples)
# y_pred_binary = np.zeros(n_samples, dtype=int)

# # Umbral para clasificación binaria
# threshold = 0.5

# # Procesar todas las muestras
# for i in tqdm(range(n_samples)):
#     # Obtener una muestra
#     x, y = next(validation_generator)
#     # Almacenar etiqueta real
#     y_true[i] = int(y[0])
#     # Generar predicción
#     pred = model.predict(x, verbose=0)[0][0]
#     y_pred[i] = pred
#     y_pred_binary[i] = 1 if pred >= threshold else 0

# # 4.5 Visualización de resultados
# # ---------------------------------------------

# # Matriz de confusión
# cm = confusion_matrix(y_true, y_pred_binary)

# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#             xticklabels=[idx_to_class[0], idx_to_class[1]],
#             yticklabels=[idx_to_class[0], idx_to_class[1]])
# plt.xlabel('Predicción')
# plt.ylabel('Real')
# plt.title('Matriz de Confusión')
# plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
# plt.show()

# # Informe de clasificación
# print("\nInforme de Clasificación:")
# report = classification_report(y_true, y_pred_binary, 
#                               target_names=[idx_to_class[0], idx_to_class[1]])
# print(report)

# # Guardar reporte como archivo de texto
# with open(os.path.join(RESULTS_DIR, 'classification_report.txt'), 'w') as f:
#     f.write(report)

# # Curva ROC
# fpr, tpr, thresholds = roc_curve(y_true, y_pred)
# roc_auc = auc(fpr, tpr)

# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('Tasa de Falsos Positivos')
# plt.ylabel('Tasa de Verdaderos Positivos')
# plt.title('Curva ROC')
# plt.legend(loc="lower right")
# plt.savefig(os.path.join(RESULTS_DIR, 'roc_curve.png'), dpi=300, bbox_inches='tight')
# plt.show()

# # Curva Precision-Recall
# precision, recall, thresholds_pr = precision_recall_curve(y_true, y_pred)
# pr_auc = auc(recall, precision)

# plt.figure(figsize=(8, 6))
# plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Curva Precision-Recall')
# plt.legend(loc="lower left")
# plt.savefig(os.path.join(RESULTS_DIR, 'pr_curve.png'), dpi=300, bbox_inches='tight')
# plt.show()

# # Distribución de probabilidades de predicción
# plt.figure(figsize=(10, 6))
# sns.histplot(y_pred[y_true==0], color='green', alpha=0.5, bins=30, label='Normal')
# sns.histplot(y_pred[y_true==1], color='red', alpha=0.5, bins=30, label='Anomalía')
# plt.axvline(x=threshold, color='black', linestyle='--', label=f'Umbral ({threshold})')
# plt.xlabel('Probabilidad de Anomalía')
# plt.ylabel('Frecuencia')
# plt.title('Distribución de Probabilidades de Predicción')
# plt.legend()
# plt.savefig(os.path.join(RESULTS_DIR, 'prediction_distribution.png'), dpi=300, bbox_inches='tight')
# plt.show()

# # 4.6 Visualizar predicciones en imágenes individuales
# # ---------------------------------------------

# def visualize_predictions(model, validation_dir, class_indices, n_samples=5):
#     """
#     Visualiza predicciones en imágenes individuales.
    
#     Args:
#         model: Modelo entrenado
#         validation_dir: Directorio con imágenes de validación
#         class_indices: Diccionario con índices de clase
#         n_samples: Número de muestras a visualizar por clase
#     """
#     # Invertir diccionario de índices
#     idx_to_class = {v: k for k, v in class_indices.items()}
    
#     # Obtener rutas de imágenes
#     normal_paths = glob.glob(os.path.join(validation_dir, 'normal', '*'))[:n_samples]
#     anomaly_paths = glob.glob(os.path.join(validation_dir, 'anomaly', '*'))[:n_samples]
    
#     image_paths = normal_paths + anomaly_paths
#     true_labels = ['normal'] * len(normal_paths) + ['anomaly'] * len(anomaly_paths)
    
#     # Preparar figura
#     fig, axes = plt.subplots(2, n_samples, figsize=(15, 6))
    
#     # Procesar cada imagen
#     for i, (img_path, true_label) in enumerate(zip(image_paths, true_labels)):
#         # Determinar índice de subplot
#         row = 0 if true_label == 'normal' else 1
#         col = i if row == 0 else i - n_samples
        
#         # Cargar y preprocesar imagen
#         img = cv2.imread(img_path)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img_resized = cv2.resize(img, model.input_shape[1:3])
#         img_array = img_resized / 255.0
#         img_array = np.expand_dims(img_array, axis=0)
        
#         # Generar predicción
#         pred = model.predict(img_array, verbose=0)[0][0]
#         pred_label = 'anomaly' if pred >= 0.5 else 'normal'
        
#         # Determinar color (verde para correcto, rojo para incorrecto)
#         color = 'green' if pred_label == true_label else 'red'
        
#         # Mostrar imagen
#         axes[row, col].imshow(img_resized)
#         axes[row, col].set_title(f"Real: {true_label}\nPred: {pred_label} ({pred:.2f})", 
#                                  color=color)
#         axes[row, col].axis('off')
    
#     plt.tight_layout()
#     plt.savefig(os.path.join(RESULTS_DIR, 'sample_predictions.png'), dpi=300, bbox_inches='tight')
#     plt.show()

# # Visualizar algunas predicciones
# visualize_predictions(model, VALIDATION_DIR, class_indices, n_samples=5)

# # 4.7 Análisis de umbrales
# # ---------------------------------------------

# def plot_threshold_metrics(y_true, y_pred):
#     """
#     Analiza el efecto de diferentes umbrales en las métricas de rendimiento.
    
#     Args:
#         y_true: Etiquetas reales
#         y_pred: Probabilidades predichas
#     """
#     thresholds = np.linspace(0, 1, 100)
#     accuracy = []
#     precision = []
#     recall = []
#     f1_scores = []
    
#     for threshold in thresholds:
#         y_pred_binary = (y_pred >= threshold).astype(int)
        
#         # Calcular métricas
#         tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
        
#         # Accuracy
#         current_accuracy = (tp + tn) / (tp + tn + fp + fn)
#         accuracy.append(current_accuracy)
        
#         # Precision
#         current_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
#         precision.append(current_precision)
        
#         # Recall
#         current_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
#         recall.append(current_recall)
        
#         # F1 Score
#         current_f1 = 2 * (current_precision * current_recall) / (current_precision + current_recall) \
#             if (current_precision + current_recall) > 0 else 0
#         f1_scores.append(current_f1)
    
#     # Encontrar el mejor umbral para F1 score
#     best_threshold_idx = np.argmax(f1_scores)
#     best_threshold = thresholds[best_threshold_idx]
    
#     # Visualizar métricas según umbral
#     plt.figure(figsize=(10, 6))
#     plt.plot(thresholds, accuracy, label='Accuracy')
#     plt.plot(thresholds, precision, label='Precision')
#     plt.plot(thresholds, recall, label='Recall')
#     plt.plot(thresholds, f1_scores, label='F1 Score')
#     plt.axvline(x=best_threshold, color='black', linestyle='--', 
#                 label=f'Mejor umbral: {best_threshold:.2f}')
#     plt.xlabel('Umbral')
#     plt.ylabel('Valor de métrica')
#     plt.title('Métricas vs. Umbral de Decisión')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(os.path.join(RESULTS_DIR, 'threshold_analysis.png'), dpi=300, bbox_inches='tight')
#     plt.show()
    
#     return best_threshold

# # Analizar y encontrar el mejor umbral
# best_threshold = plot_threshold_metrics(y_true, y_pred)
# print(f"Mejor umbral encontrado: {best_threshold:.4f}")

# # Guardar el mejor umbral
# with open(os.path.join(FINE_TUNED_DIR, 'best_threshold.txt'), 'w') as f:
#     f.write(str(best_threshold))

# # 4.8 Resumen de la evaluación
# # ---------------------------------------------

# # Crear un resumen consolidado de la evaluación
# evaluation_summary = {
#     'accuracy': evaluation_results.get('accuracy', 0),
#     'precision': evaluation_results.get('precision', 0),
#     'recall': evaluation_results.get('recall', 0),
#     'auc': evaluation_results.get('auc', 0),
#     'best_threshold': best_threshold,
#     'roc_auc': roc_auc,
#     'pr_auc': pr_auc,
#     'confusion_matrix': cm.tolist()
# }

# # Guardar resumen como JSON
# with open(os.path.join(RESULTS_DIR, 'evaluation_summary.json'), 'w') as f:
#     json.dump(evaluation_summary, f, indent=4)

# print(f"Resumen de evaluación guardado en {os.path.join(RESULTS_DIR, 'evaluation_summary.json')}")

# # Calcular métricas con el mejor umbral
# y_pred_best = (y_pred >= best_threshold).astype(int)
# cm_best = confusion_matrix(y_true, y_pred_best)

# print(f"\nMatriz de confusión con el mejor umbral ({best_threshold:.4f}):")
# print(cm_best)

# print("\nInforme de clasificación con el mejor umbral:")
# print(classification_report(y_true, y_pred_best, 
#                            target_names=[idx_to_class[0], idx_to_class[1]]))

# print("Evaluación completada con éxito!")
