import os
import random
import shutil

# Obtener el directorio base del proyecto (2 niveles arriba del script actual)
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# Directorio de origen
images_dir = os.path.join(base_dir,'etiquetado')  		  # Directorio con imágenes etiquetadas

# Directorio de destino para los conjuntos
train_dir = os.path.join(base_dir, 'conjuntos','train')     # Directorio para entrenamiento
val_dir = os.path.join(base_dir,'conjuntos', 'val')         # Directorio para validación
test_dir = os.path.join(base_dir,'conjuntos','test')       # Directorio para prueba

print("Creando Directorios para entrenamiento, validacion y testeo...")
# Crea los directorios de destino su no existen
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
print("\tCreados correctamente.")

print("Creando una lista de con los nombre de las imagenes...")
# Crea una lista con los nombres de los archivos 
image_files = os.listdir(images_dir)
print("\tLista Creada.")

print("Reordenando de manera aleatoria...")
# Reordena de maenra aleatorio
random.shuffle(image_files)
print("\tLista reordenada.")

print("Calculando tamaño de cada conjunto...\n 70% Entrenamiento.\n 15% Validacion.\n (15%) Resto para Testeo...")
# Se calcula el tamaño de cada conjunto 
train_size = int(len(image_files) * 0.7)
val_size = int(len(image_files) * 0.15)
test_size = len(image_files) - train_size - val_size

print("Copia de archivos en carpeta de Entrenamiento (Puede tomar un momento)...")
# Se copian los archivos a los directorios de destino
for i in range(train_size):
    shutil.copy(os.path.join(images_dir, image_files[i]), os.path.join(train_dir, image_files[i]))
print("Carpeta de Entrenamiento Lista")

print("Copia de archivos en carpeta de Validacion (Puede tomar un momento)...")
for i in range(train_size, train_size + val_size):
    shutil.copy(os.path.join(images_dir, image_files[i]), os.path.join(val_dir, image_files[i]))
print("Carpeta de Validacion Lista")

print("Copia de archivos en carpeta de Testeo (Puede tomar un momento)...")
for i in range(train_size + val_size, len(image_files)):
    shutil.copy(os.path.join(images_dir, image_files[i]), os.path.join(test_dir, image_files[i]))
print("Carpeta de Testeo Lista")
print("Programa finalizado")
