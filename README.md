# Red_Neuronal-para-Vision_Artificial-con-Logica_difusa
Proyecto en Python que implementa una red neuronal convolucional (CNN) para tareas de visión artificial, integrando lógica difusa para decisiones interpretables. Incluye entrenamiento profundo, preprocesamiento de imágenes y evaluación de desempeño.
## 📦 Dataset

Este proyecto utiliza el dataset [Cats vs Dogs de Kaggle](https://www.kaggle.com/datasets/sansin457/cats-vs-dogs) para tareas de clasificación binaria. El dataset no está incluido en este repositorio por motivos de licencia. Para acceder a él, visita el enlace y acepta los términos de uso en Kaggle.
# 🧠 Red Neuronal para Visión Artificial con Lógica Difusa

Proyecto en Python que implementa una red neuronal convolucional (CNN) para clasificación binaria de imágenes (gatos vs perros), integrando lógica difusa para decisiones interpretables.


---

## 🚀 Instalación

1. Clona el repositorio:
```bash
git clone https://github.com/tu_usuario/tu_repositorio.git
cd tu_repositorio
```
2. Crea y activa tu entorno virtual:
❗para este proyecto es necesario usar una version de python compatible con TensorFlow qeu sea estable, como la 3.10.11❗
```bash
   py -3.10 -m venv .venv 
   python --version # verifica que la salida en consola sea 'Python 3.10.11' o algo similar
   source .venv/bin/activate  # o .venv\Scripts\activate en Windows
```
3. Instala las dependencias:
```bash
    pip install -r requirements.txt
```   

## 🔄 Aumento de datos

El proyecto incluye scripts para aplicar técnicas de aumento de datos sobre el conjunto de imágenes, como rotación, volteo horizontal, escalado y ajustes de brillo. Estas transformaciones permiten generar nuevas muestras a partir de las imágenes originales, lo cual es especialmente útil cuando se dispone de una cantidad limitada de datos.

El aumento de datos mejora la capacidad de generalización del modelo, reduce el riesgo de sobreajuste y simula condiciones más variadas del mundo real. Esta etapa se ejecuta antes del entrenamiento y está integrada en el flujo de preprocesamiento del proyecto.


## 🧠 Contexto teórico
# Redes Neuronales Convolucionales(CNN)
Las CNN's con arquitecturas especializadas en procesamiento de imágenes que funcionan mediante:

-Convolución: Operación que permite detectar patrones específicos como bordes o texturas, al multiplicar y sumar valores de los píxeles bajo el filtro mientras este se desliza por toda la imagen. Estos filtros se entrenan para identificar características distintas y aplicar múltiples filtros en capas convolucionales la red puede aprender representaciones jerárquicas de las imágenes. 
Para este proyecto, en Keras, la clase Conv2D, permite definir el número de filtros, el tamaño y su función de activación (ReLU) que en este caso nos ayudó a introducir no linealidades y mejorar la capacidad de nuestra red para aprender patrones mas complicados.

-Pooling: Una práctica de la estadística para agrupar o combinar datos para facilitar su manejo, análisis o uso eficiente. En este caso, esta operación es para reducir dimensionalidad espacial en las imágenes o matrices resultantes tras aplicar convoluciones. No confundir con el redimensionar imágenes. El objetivo principal de hacer esto es agrupar valores numéricos, reducir la cantidad de datos y agilizar los entrenamientos y procesos en los modelos de Deep Learning (DL).

-Capas densas - totalmente conectadas: actúan como la parte de clasificación o regresión de la red. Detrás de las capas convolucionales, que extraen características jerárquicas de los datos de entrada, las capas densas procesan estas características para realizar tareas como la clasificación de imágenes final. Las entradas en estas capas vienen de las capas convolucionales y de pooling, que ya han reducido y resumido la información espacial; antes de llegar a estas capas se suele aplicar un función de aplanamiento que pasa los mapas de características multidimensionales a un vector unidimensional. Cada neurona en una capa densa está conectada a todas las neuronas de la capa anterior, lo que permite que la red combine todas las características extraídas para tomar decisiones sobre el clasificado final.

Las capas convolucionales aprenden jerárquicamente:
-Primeras capas → bordes simples
-Capas intermedias → formas complejas
-Últimas capas → características de alto nivel (ojos, orejas)
    
# Funciones de Activación
1. ReLU (Rectified Linear Unit):
```
f(x) = max(0, x)
```
-Ventaja: Introduce no-linealidad, evita el problema de desvanecimiento de gradientes
-Uso: Capas convolucionales y densas intermedias
2. Softmax:
```
f(x_i) = e^(x_i) / Σ(e^(x_j))
```
-Ventaja: Convierte salidas en probabilidades que suman 1
-Uso: Capa de salida para clasificación multiclase

# Lógica Difusa (Fuzzy Logic)
A diferencia de la lógica booleana (0 o 1), la lógica difusa permite grados de pertenencia (0 a 1):
Función de membresía triangular:
     /\
    /  \
   /    \
  /______\
Se aplica cuando la CNN es incierta (probabilidades cercanas a 0.5), la lógica difusa proporciona un mecanismo para expresar esa incertidumbre.

## Entrenamiento y Validación
# Fase 1: Configuración inicial
El siguiente bloque desactiva optimizaciones de OneDNN qeu causaron conflictos en su momento, asegurando la compatibilidad de Tensorflow y distintos hardware.
```
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
```
Recomiendo ampliamente crear un entorno virutal con python 3.10.+. La variable [base_dir] es usada para guardar el origen del proyecto, y si se siguen los pasos, las carpetas del dataset tendran esa distribución o seguiran ese orden siguiendo rutas relativas más flexibles.
```
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
```
# Fase 2: Parámetros del Modelo
Imágenes pequeñas (300×264) → entrenamiento más rápido. Recordemos que no es un "resize" como podria pensarse. 
Batch de 32 → balance entre estabilidad y velocidad
10 épocas → suficiente para convergencia inicial
```
#Parametros del modelo
nuevo_alto = 300	
nuevo_ancho = 264	
num_clases = 2
batch_size = 32
num_epocas = 10

```
# Fase 3: Procesamiento de datos
En este punto se normalizan los pixeles de un rango de `[0,255]` a un rango de  `[0,1]` facilitando la convergencia dirante el entrenamiento y evita valores númericos extremos.
```
datagen = ImageDataGenerator(rescale=1./255)
```

Ahora, [flow_from_directory]: Lee imágenes directamente desde carpetas, con el proyecto que tiene una estructura esperada como:

conjuntos/
├── train/
│   ├── perros/
│   └── gatos/
├── val/
|   ├── perros/
│   └── gatos/
└── test/
    ├── perros/
    └── gatos/

En [class_mode='categorical']: Codificación one-hot, consiste en una técnica para convertir varias variables categóricas en un formato númerico qeu los algoritmos de aprendizaje automático procesan adecuadamente. Consiste en crear una columna binaria para cada cátegoria, donde cada columna representa una categoria específica y se le atribuye un valor de [1] cuando este pertenece a la cateogoría, [0] en casos contraios (ej: [1,0] para perro, [0,1] para gato).

# Fase 4: Arquitectura del modelo
La línea siguiente, hacemos que las capas se apilen linealmente una detras de otra. a la salida de una = entrada de la siguiente. 
```
model = Sequential()
```
- Capa de la convolución 
Hay distintos elementos, pero en orden, hay 32 filtros que cada uno detecta patrones diferentes, (3,3) es la ventana de convolución de 3x3 píxeles, 'relu' introduce la no linealidad, y en `input_shape` tenemos las dimensiones de 300x264 píxeles y 3 canales, cada uno para el RGB
```
Conv2D(32, (3, 3), activation='relu', input_shape=(300, 264, 3))
```
en términos genrales, la Salida = ReLU(Imagen * Filtro + sesgo)
- Capa 2: Max Pooling
Reduce dimensiones a la mitad (300x264 -> 150x132), pero manteniendo el valor máximo en cada región de 2x2. Esto se refleja en un beneficio de reducir parámetros, acelerar entrenamiento y evitar overfitting.
```
MaxPooling2D(pool_size=(2, 2))
```
- Capa 3: Segunda convolucion
En esta capa 64 filtros detectan patrones más complejos qeu la capa anterior, en lo demas no hay cambios, pero ¿por qué se necesita aumentar el numero de filtros? A medida que se avanza en la red, se necesita capturar características ma abstractas y de mayor nivel. Más filtros permiten aprender una mayor variedad de patrones.
```
Conv2D(64, (3, 3), activation='relu')
```

- Capa 4: Segundo Max Pooling
Se reducen las dimensiones a la mitad nuevamente (150x132 -> 75x66) y se aumenta la invariancia a pequelas traslaciones en la imagen.
```
MaxPooling2D(pool_size=(2, 2))
```
- Capa 5: Flatten()
Convierte la salida 3D en un vector 1D y prepara los datos para las capas densas (fully connected).
```
Flatten()
```
- Capa 6: Capa Densa (Fully Connected)
Con 128 Neuronas en las que cada una se conecta a todas las salidas del Flatten. El propósito es aprender combinaciones complejas de caracteristicas extraidas por als capas convolucionales.
```
Dense(128, activation='relu')
```
- Capa 7: Dropout
En esta capa se regulariza la red, se desactivan aleatoriamente 50% de las neuronas durante el entrenamiento. El beneficio está en evitar el overfitting (memorización del dataset de entrenamiento)
```
Dropout(0.5)
```
- Capa 8: Capa de Salida
Se ingresan las calses (2, para perro y gato), softmax convierte las salidas en probabilidades qeu suman 1; ahora la neurona con la mayor probabilidad indica la clase predicha.
```
Dense(num_clases, activation='softmax')
```
# Fase 5: Compilación del modelo
Se hace uso de un Algoritmo qeu ajusta las tasas de aprendizaje para cada peso. Loss function `categorical_crossentropy` mide la diferencia entre las predicciones y las etiquetas realies, lo cual es ideal para la clasificacion multiclase con la codificación one-hot. Con metrics `accuracy` evaluamos la presición de las predicciones. 
# Fase 6: Guardar el mejor modelo
Se guarda el modelo durante el entrenamiento, gaurdando solo el que tiene la mejor validación, monitoreanoo la pérdida en el conjunto de validación y guarda el modelo cuando la pérdida de la validación es mínima
```
checkpoint = ModelCheckpoint('modelo_entrenado.h5', save_best_only=True, monitor='val_loss', mode='min')
```
# Fase 7: Entrenamiento del modelo
En esta etapa tenemos los datos de entrenamiento como [train_generetor], las epocas definidas que son el número de pasadas sobre el conjunto de entrenamieneto, los datos de validacion para evaluar el rendimiento del modelo en datos no vistos y usamos el modelCheckpoint para guardar el mejor modelo. 
```
history = model.fit(
    train_generator,
    epochs=num_epocas,
    validation_data=validation_generator,
    callbacks=[checkpoint]
)
```
# Fase 8: Evaluación del modelo
En [test_generator] se gaurda un conjunto de daots independiente para evaluar el rendimiento final del modelo (datos de prueba) y en [model.evaluete] se calcula la perdida y la presicion del conjunto de prueba.
```
test_generator = datagen.flow_from_directory(
    os.path.join(base_dir,'conjuntos','test'),
    target_size=(nuevo_alto, nuevo_ancho),
    batch_size=batch_size,
    class_mode='categorical'
)
test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Precisión en el modelo de pruebas: {test_accuracy}')

```
# Fase 9: Guardado del modelo entrenado
En esta fase simplemente se guarda el modelo para su uso posterior.
```
model.save('modelo_entrenado.h5')
```

# Fase 10: Se implementa 
Esta fase introduce un componente adicional para refinar las predicciones de la CNN en situaciones de incertidumbre.

- fuzzy_decision(probabilities)
Función de Membresía: Define cómo las probabilidades de la CNN se mapean a grados de pertenencia en conjuntos difusos:
- - low_confidence: Probabilidades bajas (0-0.4)
- - medium_confidence: Probabilidades medias (0.4-0.6)
- - high_confidence: Probabilidades altas (0.6-1)
Reglas Difusas:
- - Si la probabilidad de "perro" es alta (>0.6), la decisión es "perro".
- - Si la probabilidad de "gato" es alta (>0.6), la decisión es "gato".
En caso contrario (incertidumbre), la decisión es "Incierto".
- predict_with_fuzzy_logic(image_path)
- -Carga y preprocesa la imagen.
- - Realiza la predicción con el modelo CNN.
- - Aplica la lógica difusa para tomar una decisión final.
- Ejemplo de Predicción
Demuestra cómo usar la función predict_with_fuzzy_logic para predecir la clase de una imagen.

## Conclusión
Este código implementa una CNN para la clasificación de imágenes, utilizando técnicas de regularización y guardado de modelos para optimizar el rendimiento y evitar el overfitting. La adición de la lógica difusa permite manejar la incertidumbre en las predicciones, proporcionando una toma de decisiones más robusta.

Y si llegaste hasta este punto, creo está mas decir que este es mas un proyecto que muestra de manera mas didáctica el proceso de hacer una red neuronal mostrando parte de sus conceptos clave. Lo mas improtante es comprender que este proceso puede ser replicado en muchas áreas con problemas especificos, como en un el área de seguridad e higiene en una empresa, para verificar por medio de cámaras que los operadores de maquinaría pesada porten adecuadamente su equipo de protección, o para verificar que no hay personal en espacios delimitados cuando un operario (de grúa por ejemplo) esta trabajando en ese espacio. En fin, espero que este material sea de ayuda o simplemente aporte en el conocimiento de alguien. 