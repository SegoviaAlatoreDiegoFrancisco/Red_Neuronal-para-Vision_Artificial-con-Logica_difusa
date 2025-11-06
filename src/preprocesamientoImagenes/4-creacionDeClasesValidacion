import os
import shutil

#Obtener el directorio base de del proyecto
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# Función que clasifica las imágenes según el último número de su nombre
def clasificar_imagenes(directorio_entrada, directorio_salida_0, directorio_salida_1):
    print("Se está ejecutando la clasificacion de las imagenes según su etiqueta (0 ó 1 en su último caracter)")
    # Crear las carpetas de salida si no existen
    if not os.path.exists(directorio_salida_0):
        ("Creando carpeta para gatos...")
        os.makedirs(directorio_salida_0)
    if not os.path.exists(directorio_salida_1):
        ("Creando carpeta para perros...")
        os.makedirs(directorio_salida_1)
    print("Carpetas creadas")
    print("Listando y verificación de los archvivos del directorio de entrada...")
    # Listar todos los archivos en el directorio de entrada
    for archivo in os.listdir(directorio_entrada):
        ruta_archivo = os.path.join(directorio_entrada, archivo)

        # Verificar si es un archivo de imagen
        if os.path.isfile(ruta_archivo):
            # Obtener el último carácter del nombre del archivo (antes de la extensión)
            nombre_sin_extension = os.path.splitext(archivo)[0]
            ultimo_caracter = nombre_sin_extension[-1]

            # Comprobar si el último número es 0 o 1
            if ultimo_caracter == '0':
                # Mover el archivo a la carpeta correspondiente
                shutil.move(ruta_archivo, os.path.join(directorio_salida_0, archivo))
            elif ultimo_caracter == '1':
                shutil.move(ruta_archivo, os.path.join(directorio_salida_1, archivo))
'''
            SE DEBE EDITAR EL DIRECTORIO 
'''
# Ruta de las carpetas
directorio_entrada = os.path.join(base_dir, 'conjuntos','val')  # Ruta de la carpeta que contiene las imágenes
directorio_salida_0 = os.path.join(base_dir, 'conjuntos','val','gatos')  # Carpeta de destino para imágenes cuyo nombre termina en 0 gatos
directorio_salida_1 = os.path.join(base_dir,'conjuntos','val','perros')  # Carpeta de destino para imágenes cuyo nombre termina en 1 perros

# Llamar a la función para clasificar las imágenes
clasificar_imagenes(directorio_entrada, directorio_salida_0, directorio_salida_1)
print("Se terminó la clasificacion.")
