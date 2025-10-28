import os
import shutil

# Obtener el directorio base del proyecto (2 niveles arriba del script actual)
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# Directorios de origen
cats_dir = os.path.join(base_dir, 'PetImages', 'Cat')
dogs_dir = os.path.join(base_dir, 'PetImages', 'Dog')

# Directorio de destino
destination_dir = os.path.join(base_dir, 'etiquetado')

# Crear directorio de destino si no existe
os.makedirs(destination_dir, exist_ok=True)

# Etiquetar imágenes de gatos
for img in os.listdir(cats_dir):
    shutil.copy(os.path.join(cats_dir, img), os.path.join(destination_dir, f"{img.split('.')[0]}_0.jpg"))

# Etiquetar imágenes de perros
for img in os.listdir(dogs_dir):
    shutil.copy(os.path.join(dogs_dir, img), os.path.join(destination_dir, f"{img.split('.')[0]}_1.jpg"))
