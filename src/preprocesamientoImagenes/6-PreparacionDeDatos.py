import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Obtener el directorio base del proyecto (2 niveles arriba del script actual)
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# Definir parámetros
img_height, img_width = 224, 224
batch_size = 32

# Generador de datos para entrenamiento
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Generador de datos para validación
val_datagen = ImageDataGenerator(rescale=1./255)

# Cargar imágenes desde directorios
train_generator = train_datagen.flow_from_directory(
    os.path.join(base_dir,'conjuntos','train'),
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    color_mode='rgb',  # Asegurarnos de que las imágenes se carguen en color (RGB)
    shuffle=True  # Mezcla las imágenes aleatoriamente
)

validation_generator = val_datagen.flow_from_directory(
    os.path.join(base_dir,'conjuntos','val'),
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    color_mode='rgb',  # Asegurarnos de que las imágenes se carguen en color (RGB)
    shuffle=False  # Para validación, no es necesario mezclar
)


