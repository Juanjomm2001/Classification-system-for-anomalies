# Sistema de Detección de Anomalías para Veolia Kruger

Este proyecto implementa un sistema de detección de anomalías en tiempo real para monitorear la salida de lodos ("sludge") en la planta industrial de Veolia Kruger utilizando técnicas de Computer Vision y TensorFlow.

## Descripción

El sistema detecta anomalías en la salida de lodos, como grietas o partes conglomeradas que no son deseables, mediante un modelo de deep learning con fine-tuning. Cuando se detecta una anomalía, el sistema envía alertas a los operadores para que puedan tomar acciones correctivas.

## Estructura del Proyecto

```
veolia_anomaly_detection/
│
├── data/
│   ├── raw/                # Imágenes originales
│   ├── processed/          # Imágenes preprocesadas
│   ├── train/              # Conjunto de entrenamiento
│   └── validation/         # Conjunto de validación
│
├── models/
│   ├── pretrained/         # Modelos pre-entrenados
│   ├── fine_tuned/         # Modelos después del fine-tuning
│   └── evaluation_results/ # Resultados de evaluación
│
├── notebooks/
│   ├── 01_exploracion_datos.ipynb
│   ├── 02_preprocesamiento.ipynb
│   ├── 03_fine_tuning.ipynb
│   └── 04_evaluacion.ipynb
│
├── src/
│   ├── data/               # Funciones para manejo de datos
│   ├── models/             # Definición y entrenamiento
│   └── utils/              # Utilidades generales
│
├── inference/
│   ├── realtime_detection.py  # Script para detección en tiempo real
│   └── alert_system.py        # Sistema de alertas
│
├── requirements.txt        # Dependencias
└── README.md               # Documentación
```

## Requisitos

- Python 3.7+
- TensorFlow 2.4+ 
- OpenCV 4.5+
- Otras dependencias listadas en `requirements.txt`

## Instalación

1. Clonar el repositorio:
   ```
   git clone https://github.com/veolia-kruger/anomaly-detection.git
   cd anomaly-detection
   ```

2. Crear y activar un entorno virtual:
   ```
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```

3. Instalar dependencias:
   ```
   pip install -r requirements.txt
   ```

## Flujo de Trabajo

### 1. Exploración de Datos

El notebook `01_exploracion_datos.ipynb` permite explorar y visualizar las imágenes para comprender mejor las características de las muestras normales y anómalas.

### 2. Preprocesamiento

El notebook `02_preprocesamiento.ipynb` realiza el preprocesamiento de las imágenes, incluyendo redimensionamiento, normalización y división en conjuntos de entrenamiento y validación.

### 3. Entrenamiento y Fine-Tuning

El notebook `03_fine_tuning.ipynb` implementa el fine-tuning de un modelo pre-entrenado (como EfficientNetB0, MobileNetV2 o ResNet50) para adaptarlo a la detección de anomalías específicas.

### 4. Evaluación

El notebook `04_evaluacion.ipynb` evalúa el rendimiento del modelo mediante métricas como precisión, recall, curvas ROC, y análisis de umbrales óptimos.

### 5. Detección en Tiempo Real

El script `realtime_detection.py` implementa la detección en tiempo real utilizando el modelo entrenado. Puede procesar video de una cámara en vivo o de un archivo de video.

```bash
# Ejemplo de uso con webcam
python inference/realtime_detection.py --source 0

# Ejemplo con stream RTSP
python inference/realtime_detection.py --source rtsp://ejemplo.com/stream

# Ejemplo con archivo de video
python inference/realtime_detection.py --source video.mp4

# Personalizar umbral de detección
python inference/realtime_detection.py --source 0 --threshold 0.75

# Activar sistema de alertas
python inference/realtime_detection.py --source 0 --alerts
```

## Sistema de Alertas

El módulo `alert_system.py` proporciona funcionalidades para notificar cuando se detecta una anomalía. Admite múltiples métodos de notificación:

- Registro local de eventos
- Notificaciones por correo electrónico
- Mensajes de texto (SMS)
- Notificaciones push
- Integración con sistemas de gestión de Veolia Kruger

Para configurar el sistema de alertas:

```bash
# Crear archivo de configuración de ejemplo
python inference/alert_system.py --create-config

# Editar el archivo resultante (alert_config.json) con tus credenciales
```

## Personalización del Modelo

Para personalizar el modelo para una aplicación específica:

1. Recopila imágenes representativas de condiciones normales y anómalas.
2. Organízalas en las carpetas `data/raw/normal` y `data/raw/anomaly`.
3. Ejecuta los notebooks del 1 al 4 para entrenar y evaluar tu modelo personalizado.
4. Utiliza el modelo resultante con el script de detección en tiempo real.

## Contribuciones

Las contribuciones son bienvenidas. Para cambios importantes, por favor abre primero un issue para discutir lo que te gustaría cambiar.

## Licencia

[MIT](https://choosealicense.com/licenses/mit/)
