import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

# Cargar el modelo previamente entrenado como variable global
model = tf.keras.models.load_model('modelo_entrenado.h5')

# Función para seleccionar una imagen
def seleccionar_imagen():
    root = tk.Tk()
    root.withdraw()
    # Ocultar la ventana principal
    archivo_seleccionado = filedialog.askopenfilename(
        title="Seleccionar imagen",
        filetypes=[("Archivos de imagen", "*.jpg;*.jpeg;*.png;*.bmp")]
        )
    return archivo_seleccionado

# Preprocesar la imagen (ajustar tamaño, normalizar y convertir a array)
def preprocess_image(img_path, target_size=(300, 264)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = np.expand_dims(image.img_to_array(img) / 255.0, axis=0)
    return img_array

# Predicción utilizando el modelo entrenado
def predict_image(img_path):
    processed_img = preprocess_image(img_path)
    prediction = model.predict(processed_img)[0]
    pred_label = "Gato" if prediction[0] > 0.5 else "Perro"
    print(f"Probabilidades: [Gato: {prediction[0]}, Perro: {prediction[1]}]")
    print(f"La imagen es un {pred_label}.")
    
    # Mostrar imagen y predicción
    plt.imshow(image.load_img(img_path))
    plt.title(f"Predicción: {pred_label}")
    plt.show()
    
# Función de decisión difusa con gráfico de pertenencia
def fuzzy_decision(probabilities):
    # Crear un rango de valores entre 0 y 1
    x = np.linspace(0, 1, 100)

    # Definir las funciones de membresía para baja, media y alta confianza
    low_confidence = fuzz.trapmf(x, [0, 0, 0.4, 0.6])
    medium_confidence = fuzz.trimf(x, [0.4, 0.5, 0.6])
    high_confidence = fuzz.trapmf(x, [0.6, 0.8, 1, 1])

    # Graficar las funciones de membresía
    plt.plot(x, low_confidence, label='Baja Confianza', color='red')
    plt.plot(x, medium_confidence, label='Confianza Media', color='yellow')
    plt.plot(x, high_confidence, label='Alta Confianza', color='green')

    # Calcular la pertenencia de las probabilidades a las funciones de membresía
    low_belong = np.interp(probabilities[0], x, low_confidence)
    medium_belong = np.interp(probabilities[0], x, medium_confidence)
    high_belong = np.interp(probabilities[0], x, high_confidence)

    # Imprimir las pertenencias
    print(f"Pertenencia de Gato (probabilidad {probabilities[0]}): Baja = {low_belong}, Media = {medium_belong}, Alta = {high_belong}")
    
    # Graficar la pertenencia de la probabilidad
    plt.scatter(probabilities[0], low_belong, color='red', zorder=5)
    plt.scatter(probabilities[0], medium_belong, color='yellow', zorder=5)
    plt.scatter(probabilities[0], high_belong, color='green', zorder=5)

    # Título y etiquetas
    plt.title("Funciones de Membresía y Pertenencia de Probabilidad")
    plt.xlabel('Probabilidad')
    plt.ylabel('Grado de Membresía')
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()

    # Decisión difusa basada en las probabilidades
    if probabilities[0] > 0.5:
        return "Gato"
    elif probabilities[1] > 0.5:
        return "Perro"
    else:
        return "Incierto"
# Predicción con lógica difusa
def predict_with_fuzzy_logic(img_path):
    processed_img = preprocess_image(img_path)
    probabilities = model.predict(processed_img)[0]
    decision = fuzzy_decision(probabilities)
    print(f"Predicción con lógica difusa: {decision}")
    return decision


img_path = seleccionar_imagen()
predict_with_fuzzy_logic(img_path)  # Predicción con lógica difusa
predict_image(img_path)  # Predicción normal


