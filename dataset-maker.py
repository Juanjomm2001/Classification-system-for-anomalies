import os
import shutil
import re
from datetime import datetime

def clasificar_imagenes(directorio_origen, imagen1, imagen2, destino="normal"):
    """
    Busca dos imágenes específicas en un directorio y las mueve a una carpeta de destino.
    
    Args:
        directorio_origen: Ruta al directorio que contiene todas las imágenes
        imagen1: Nombre de la primera imagen a buscar
        imagen2: Nombre de la segunda imagen a buscar
        destino: Carpeta de destino ('normal' o 'anomaly')
    
    Returns:
        bool: True si ambas imágenes fueron encontradas y movidas, False en caso contrario
    """
    # Validar que el destino sea válido
    if destino not in ["normal", "anomaly"]:
        print(f"Error: El destino debe ser 'normal' o 'anomaly', no '{destino}'")
        return False
    
    # Asegurarse de que existan las carpetas de destino
    directorio_destino = os.path.join(directorio_origen, destino)
    if not os.path.exists(directorio_destino):
        os.makedirs(directorio_destino)
        print(f"Se creó la carpeta de destino: {directorio_destino}")
    
    # Verificar si las imágenes existen en el directorio
    ruta_imagen1 = os.path.join(directorio_origen, imagen1)
    ruta_imagen2 = os.path.join(directorio_origen, imagen2)
    
    if not os.path.exists(ruta_imagen1):
        print(f"Error: No se encontró la imagen: {imagen1}")
        return False
    
    if not os.path.exists(ruta_imagen2):
        print(f"Error: No se encontró la imagen: {imagen2}")
        return False
    
    # Mover las imágenes a la carpeta de destino
    try:
        shutil.move(ruta_imagen1, os.path.join(directorio_destino, imagen1))
        print(f"Imagen {imagen1} movida a {destino}")
        
        shutil.move(ruta_imagen2, os.path.join(directorio_destino, imagen2))
        print(f"Imagen {imagen2} movida a {destino}")
        
        return True
    except Exception as e:
        print(f"Error al mover las imágenes: {str(e)}")
        return False

def encontrar_imagen_siguiente(directorio, imagen_base):
    """
    Encuentra la imagen que sigue cronológicamente a la imagen base.
    El formato esperado es: dataset_image_YYYYMMDD-HHMMSS.jpg
    
    Args:
        directorio: Directorio donde buscar
        imagen_base: Nombre de la imagen base
    
    Returns:
        str: Nombre de la siguiente imagen o None si no se encuentra
    """
    # Extraer la fecha y hora de la imagen base
    patron = r"dataset_image_(\d{8})-(\d{6})\.jpg"
    match = re.match(patron, imagen_base)
    
    if not match:
        print(f"Error: La imagen base '{imagen_base}' no tiene el formato esperado")
        return None
    
    fecha_str = match.group(1)
    hora_str = match.group(2)
    
    # Convertir a objeto datetime
    try:
        timestamp_base = datetime.strptime(f"{fecha_str}-{hora_str}", "%Y%m%d-%H%M%S")
    except ValueError:
        print(f"Error: No se pudo convertir la fecha/hora de '{imagen_base}'")
        return None
    
    # Buscar la siguiente imagen
    siguiente_imagen = None
    menor_diferencia = float('inf')
    
    for archivo in os.listdir(directorio):
        if not archivo.endswith('.jpg') or archivo == imagen_base:
            continue
        
        match = re.match(patron, archivo)
        if not match:
            continue
        
        fecha_str = match.group(1)
        hora_str = match.group(2)
        
        try:
            timestamp = datetime.strptime(f"{fecha_str}-{hora_str}", "%Y%m%d-%H%M%S")
        except ValueError:
            continue
        
        # Solo considerar imágenes posteriores a la base
        if timestamp > timestamp_base:
            diferencia = (timestamp - timestamp_base).total_seconds()
            if diferencia < menor_diferencia:
                menor_diferencia = diferencia
                siguiente_imagen = archivo
    
    return siguiente_imagen

# Ejemplo de uso:
if __name__ == "__main__":
    # Directorio donde están todas las imágenes
    directorio_imagenes = "C:\Juanjo\Veolia\sludge_image_viewer (1)\sludge_image_viewer\local_dataset_experiment_20250408"
    
    # Imagen de inicio
    imagen_inicial = "dataset_image_20250408-082602.jpg"
    
    # Encontrar la siguiente imagen cronológicamente
    siguiente = encontrar_imagen_siguiente(directorio_imagenes, imagen_inicial)
    
    if siguiente:
        print(f"La imagen siguiente a {imagen_inicial} es {siguiente}")
        
        # Mover las dos imágenes a la carpeta "anomaly"
        clasificar_imagenes(directorio_imagenes, imagen_inicial, siguiente, "anomaly")
    else:
        print(f"No se encontró una imagen posterior a {imagen_inicial}")