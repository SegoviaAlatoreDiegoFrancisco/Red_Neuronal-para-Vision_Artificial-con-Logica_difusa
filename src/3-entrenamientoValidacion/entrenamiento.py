import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import skfuzzy as fuzz
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint


#Obtener la base del directorio del proyecto
base_dir=os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# Parametros del modelo
nuevo_alto = 300	
nuevo_ancho = 264	
num_clases = 2
batch_size = 32
num_epocas = 10

# Procesamiento de datos 
datagen = ImageDataGenerator(rescale=1./255)
train_generator = datagen.flow_from_directory(
    os.path.join(base_dir,'conjuntos','train'),
    target_size=(nuevo_alto, nuevo_ancho),
    batch_size=batch_size,
    class_mode='categorical'
)
validation_generator = datagen.flow_from_directory(
    os.path.join(base_dir,'conjuntos','val'),
    target_size=(nuevo_alto, nuevo_ancho),
    batch_size=batch_size,
    class_mode='categorical'
)

# Definicion del modelo 
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(nuevo_alto, nuevo_ancho, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # Añadir regularización
model.add(Dense(num_clases, activation='softmax'))

# Compilación del modelo 
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Guardar el mejor modelo durante el entrenamiento
checkpoint = ModelCheckpoint('modelo_entrenado.h5', save_best_only=True, monitor='val_loss', mode='min')

# Entrenamiento del modelo
history = model.fit(
    train_generator,
    epochs=num_epocas,
    validation_data=validation_generator,
    callbacks=[checkpoint]
)

# Evaluación del modelo
test_generator = datagen.flow_from_directory(
    os.path.join(base_dir,'conjuntos','test'),
    target_size=(nuevo_alto, nuevo_ancho),
    batch_size=batch_size,
    class_mode='categorical'
)
test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Precición en el modelo de pruebas: {test_accuracy}')

# Guarda el modelo
#model.save('modelo_entrenado.h5')
model.save()
# Implementación de Lógica Difusa
def fuzzy_decision(probabilities):
    # Definir funciones de membresía según la probabilidad
    low_confidence = fuzz.trapmf(probabilities, [0, 0, 0.4, 0.6])
    medium_confidence = fuzz.trimf(probabilities, [0.4, 0.5, 0.6])
    high_confidence = fuzz.trapmf(probabilities, [0.6, 0.8, 1, 1])

    # Reglas
    if probabilities[0] > 0.6:
        return "gato"
    elif probabilities[1] > 0.6:
        return "perro"
    else:
        return "Incierto"

# Función de predicción con lógica difusa
def predict_with_fuzzy_logic(image_path):
    # Cargar y preprocesar la imagen
    from tensorflow.keras.preprocessing import image
    img = image.load_img(image_path, target_size=(nuevo_alto, nuevo_ancho))
    img_array = image.img_to_array(img) / 255.0  # Normalizar la imagen
    img_array = np.expand_dims(img_array, axis=0)  # Añadir dimensión de batch
    
    # Realizar la predicción
    probabilities = model.predict(img_array)[0]  
    decision = fuzzy_decision(probabilities)  
    return decision

# Ejemplo de predicción
          #'../conjuntos/val/gatos/18_0.jpg'
image_path =     os.path.join(base_dir,'conjuntos','train','gatos','1_0.jpg')  # Ruta de la imagen que se quiere predecir
result = predict_with_fuzzy_logic(image_path)
print(f'Predicción con lógica difusa: {result}')

