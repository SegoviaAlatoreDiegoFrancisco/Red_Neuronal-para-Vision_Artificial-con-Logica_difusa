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
def select_image():
    root = tk.Tk()
    root.withdraw()
    # Ocultar la ventana principal
    selected_file = filedialog.askopenfilename(
        title="Seleccionar imagen",
        filetypes=[("Archivos de imagen", "*.jpg;*.jpeg;*.png;*.bmp")]
        )
    return selected_file

# Preprocesar la imagen (ajustar tamaño, normalizar y convertir a array)
def preprocess_image(img_path, target_size=(300, 264)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = np.expand_dims(image.img_to_array(img) / 255.0, axis=0)
    return img_array

# Predicción utilizando el modelo entrenado
def predict_image(img_path):
    processed_img = preprocess_image(img_path)
    prediction = model.predict(processed_img)[0]
    pred_label = ("Incierto" if (prediction[0] > 0.45 and prediction[1] > 0.45)
                  else ("Gato" if prediction[0] > 0.55 else "Perro"))
    print(f"Probabilidades: [Gato: {prediction[0]}, Perro: {prediction[1]}]")
    print(f"La imagen es un {pred_label}.")
    
    # Mostrar imagen y predicción
    plt.imshow(image.load_img(img_path))
    plt.title(f"Predicción: {pred_label}")
    plt.show()
    
# Función de decisión difusa con gráfico de pertenencia
def fuzzy_decision(probabilities):
    probs = np.asarray(probabilities)
    if probs.size == 1:
        #si el modelo devuelve una sola probabilidad (sigmoid), se asume qeu es P(Gato)
        probs = np.array([probs.item(),1.0 -probs.item()])
    elif probs.size >= 2:
        probs = probs[:2]
    probs = probs / probs.sum()

    # Crear un rango de valores entre 0 y 1
    x = np.linspace(0, 1, 100)

    # Definir las funciones de membresía para baja, media y alta confianza
    low_confidence = fuzz.trapmf(x, [0, 0, 0.4, 0.6])
    medium_confidence = fuzz.trimf(x, [0.4, 0.5, 0.6])
    high_confidence = fuzz.trapmf(x, [0.6, 0.8, 1, 1])

    #preparar figura con dos g´raficas: membresia y circular o de "pastel"
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12,5))

    #Gráfico de funciones de memebresía de gato
    ax1.plot(x,low_confidence, label='Baja Confianza', color='red')
    ax1.plot(x,medium_confidence, label='Confianza Media', color='orange')
    ax1.plot(x,high_confidence, label='Alta Confianza', color='green')


    #Calcular la pertennecia de la probabilidad de gato a las funciones de membresia
    cat_low_belong= np.interp(probs[0],x,low_confidence)
    cat_medium_belong= np.interp(probs[0],x,medium_confidence)
    cat_high_belong=np.interp(probs[0],x,high_confidence)

    #Marcar las pertenencias en el plot de membresia
    ax1.scatter([probs[0]]*3, [cat_low_belong, cat_medium_belong, cat_high_belong],
                    c=['red','orange','green'],zorder=5)
    ax1.set_title("Funciones de memeresía (Prob.Gato)")
    ax1.set_xlabel("Probabilidad")
    ax1.set_ylabel("Grado de memebresía")
    ax1.grid(True)
    ax1.legend(loc="best")

    #gráfica circular con las probabilidades (Gato | Perro)
    labels = ['Gato','Perro']
    sizes = [probs[0],probs[1]]
    colors = ['#66b3ff','#ff9999']
    explode = (0.05,0.0)

    ax2.pie(sizes,labels=labels,colors=colors,autopct='%1.1f%%',
            startangle=90,explode=explode, shadow=True)
    ax2.axis('equal')
    ax2.set_title("Distribución de probabilidades")

    #gráfica de funciones de memebresia para perro 
    ax3.plot(x,low_confidence,label= 'Baja Confianza', color='red')
    ax3.plot(x,medium_confidence,label='Confianza Media',color='orange')
    ax3.plot(x,high_confidence,label='Alta Confidencia', color='green')

    #Colocar la pertenencia de probabilidad de perro a las funciones
    dog_low_belong = np.interp(probs[1],x,low_confidence)
    dog_medium_belong = np.interp(probs[1],x,medium_confidence)
    dog_high_belong = np.interp(probs[1],x,high_confidence)

    ax3.scatter([probs[1]]*3,[dog_low_belong, dog_medium_belong, dog_high_belong],
                c=['red','orange','green'],zorder=5)
    ax3.set_title("Funciones de membresía (Prob. Perro)")
    ax3.set_xlabel("Probabilidad")
    ax3.set_ylabel("Grado de membresía")
    ax3.grid(True)
    ax3.legend(loc="best")

    plt.suptitle("Análisis difuso y distibución probabilística")
    plt.tight_layout(rect=[0,0.03,1,0.95])
    plt.show()

    #imprimir valores útiles en consola
    print(f"Probabilidades -> Gato: {probs[0]:.4f}, Perro: {probs[1]:.4f}")
    print(f"Pertenencia de Gato: Baja={cat_low_belong:.4f}, Media={cat_medium_belong:.4f}, Alta={cat_high_belong:.4f}")
    print(f"Pertenencia de Perro: Baja={dog_low_belong:.4f}, Media={dog_medium_belong:.4f}, Alta={dog_high_belong:.4f}")

    # Decisión difusa basada en las probabilidades
    if ( 0.45 < probabilities[0]) and ( 0.45 < probabilities[1]):
        return "Incierto"
    elif probabilities[1] > 0.5:
        return "Perro"
    else:
        return "Gato"
    
    '''
def fuzzy_decision_pie(probabilities):
    probs = np.asarray(probabilities)
    if probs.size == 1:
        probs = np.array([probs.item(), 1.0 - probs.item()])
    else:
        pros = probs / probs[:2]
    probs = probs / probs.sum()

    labels = ['Gato',"Perro"]
    colors = ['#66b3ff', '#ff9999']
    explode =(0.05,0.0) # resalta el segmento del gato (primer segment)

    fig, ax = plt.subplots(figsize=(6,6))
    wedges , text, autotext = ax.pie(
        probs
        ,labels = labels
        ,colors = colors
        ,autopct ='%1.1f%%'
        ,startangle =90
        ,explode=explode
        ,shadow=True
    ) 
    ax.axis('equal')
    ax.set_title("Distribución de probabilidades (Gato vs Perro)")
    plt.setp(autotext,size=10,weight='bold')
    plt.show()

    print(f"Probabilidades -> Gato: {probs[0]:.4f}, Perro: {probs[1]:.4f}")
    if probs[0] > probs[1]:
        return 'Gato'
    elif probs[1] > probs[0]:
        return 'Perro'
    else: 
        return "Incierto"
    '''
# Predicción con lógica difusa
def predict_with_fuzzy_logic(img_path):
    if not img_path:
        print("No se seleccionó ninguna imagen")
    processed_img = preprocess_image(img_path)
    probabilities = model.predict(processed_img)[0]
    decision = fuzzy_decision(probabilities)
    print(f"Predicción con lógica difusa: {decision}")
    return decision


img_path = select_image()
predict_with_fuzzy_logic(img_path)  # Predicción con lógica difusa
predict_image(img_path)  # Predicción normal


