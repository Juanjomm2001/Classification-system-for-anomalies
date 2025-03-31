import os
import cv2
import matplotlib.pyplot as plt

# Ruta absoluta de la carpeta
folder_path = r'C:/Juanjo/Veolia/project3_claud/data/raw/anomaly'

# Verificar si la carpeta existe
if os.path.isdir(folder_path):
    # Listar todos los archivos en la carpeta
    files = os.listdir(folder_path)
    
    # Filtrar solo los archivos .jpg y .png
    image_files = [f for f in files if f.endswith('.jpg') or f.endswith('.png')]
    
    if not image_files:
        print(f"La carpeta {folder_path} no contiene imágenes.")
    else:
        print(f"La carpeta {folder_path} contiene {len(image_files)} imagen(es).")
        
        # Cargar y mostrar una imagen de ejemplo
        img_path = os.path.join(folder_path, image_files[0])
        img = cv2.imread(img_path)
        
        # Convertir la imagen de BGR (OpenCV) a RGB (para matplotlib)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Mostrar la imagen
        plt.imshow(img_rgb)
        plt.title(f"Imagen de ejemplo: {image_files[0]}")
        plt.axis('off')  # No mostrar ejes
        plt.show()

else:
    print(f"La ruta {folder_path} no es una carpeta válida.")
