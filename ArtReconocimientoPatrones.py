import tkinter as tk
from tkinter import filedialog, Label, messagebox, Button, Scale, HORIZONTAL, simpledialog
from PIL import Image, ImageTk, ImageDraw
import cv2
import numpy as np
import json
import os

class AplicacionReconocimientoPatrones:
    def __init__(self, root):
        """
        Inicializa la aplicación de reconocimiento de patrones.
        """
        self.root = root
        self.root.title("Reconocimiento de Patrones con ART")
        
        # Crear el frame principal
        self.frame_principal = tk.Frame(self.root)
        self.frame_principal.pack(pady=20, padx=30)
        
        # Frame izquierdo para la imagen en escala de grises y patrones
        self.frame_izquierdo = tk.Frame(self.frame_principal)
        self.frame_izquierdo.pack(side=tk.LEFT, padx=10, pady=(0, 10))  # Añadimos un espacio vertical para separación
        
        # Etiqueta para la sección de la imagen en escala de grises y patrones reconocidos
        self.label_izquierda = Label(self.frame_izquierdo, text="Imagen con patrón más similar")
        self.label_izquierda.pack()

        # Etiqueta para mostrar la imagen en escala de grises
        self.etiqueta_imagen_similar = Label(self.frame_izquierdo)
        self.etiqueta_imagen_similar.pack()

        # Frame derecho para la imagen original cargada
        self.frame_derecho = tk.Frame(self.frame_principal)
        self.frame_derecho.pack(side=tk.LEFT, padx=10, pady=(0, 10))  # Añadimos un espacio vertical para separación

        # Etiqueta para la sección de la imagen original cargada
        self.label_derecha = Label(self.frame_derecho, text="Imagen original o dibujo")
        self.label_derecha.pack()

        # Canvas para dibujar sobre la imagen
        self.canvas = tk.Canvas(self.frame_derecho, width=250, height=250, bg='white')
        self.canvas.pack()

        # Botones
        self.frame_botones = tk.Frame(self.root)
        self.frame_botones.pack(pady=(10, 0))

        self.boton_cargar = tk.Button(self.frame_botones, text="Cargar Imagen", command=self.cargar_imagen)
        self.boton_cargar.pack(side=tk.LEFT, padx=5)

        self.boton_anadir = tk.Button(self.frame_botones, text="Entrenar", command=self.entrenar_imagen)
        self.boton_anadir.pack(side=tk.LEFT, padx=5)

        self.boton_reconocer = tk.Button(self.frame_botones, text="Analizar Patrón", command=self.reconocer_patrones)
        self.boton_reconocer.pack(side=tk.LEFT, padx=5)

        self.boton_limpiar = tk.Button(self.frame_botones, text="Limpiar", command=self.limpiar_canvas)
        self.boton_limpiar.pack(side=tk.LEFT, padx=5)

        # Slider para manejar el parámetro de vigilancia
        self.slider_vigilancia = Scale(self.root, from_=0.0, to=1.0, resolution=0.01, orient=HORIZONTAL, label="Param. Vigilancia:")
        self.slider_vigilancia.set(0.5)
        self.slider_vigilancia.pack(pady=(10, 20))

        # Variable para almacenar la ruta del archivo cargado
        self.ruta_archivo_actual = None
        self.pesos_guardados = "arregloPatrones.json"  # Nombre del archivo para guardar patrones

        # Modelo ART inicializado como None
        self.modelo_art = None

        # Cargar patrones guardados si existen
        if not os.path.exists(self.pesos_guardados):
            self.procesar_imagenes_entrenamiento()
        self.cargar_patrones()

        # Variables para dibujar
        self.dibujando = False
        self.puntos = []

        self.canvas.bind("<Button-1>", self.iniciar_dibujo)
        self.canvas.bind("<B1-Motion>", self.dibujar)
        self.canvas.bind("<ButtonRelease-1>", self.terminar_dibujo)

    def cargar_imagen(self):
        """
        Carga una imagen seleccionada por el usuario.
        """
        ruta_archivo = filedialog.askopenfilename()
        if ruta_archivo:
            self.ruta_archivo_actual = ruta_archivo
            self.imagen_original = Image.open(ruta_archivo)
            self.mostrar_imagen_original()

    def mostrar_imagen_original(self):
        """
        Muestra la imagen original en la interfaz.
        """
        imagen = Image.open(self.ruta_archivo_actual)
        imagen = imagen.resize((250, 250))  # Redimensionar para que se ajuste a la ventana
        self.imagen_tk = ImageTk.PhotoImage(imagen)
        
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.imagen_tk)
        self.canvas.image = self.imagen_tk  # Mantener referencia de la imagen

    def mostrar_imagen_similar(self, ruta_imagen):
        """
        Muestra la imagen con el patrón más similar en la interfaz.
        """
        imagen = Image.open(ruta_imagen)
        imagen = imagen.resize((250, 250))  # Redimensionar para que se ajuste a la ventana
        foto = ImageTk.PhotoImage(imagen)
        
        self.etiqueta_imagen_similar.configure(image=foto)
        self.etiqueta_imagen_similar.image = foto

    def preprocesar_imagen(self, ruta):
        """
        Preprocesa la imagen para su análisis.
        """
        imagen = cv2.imread(ruta)
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        binaria = cv2.threshold(gris, 128, 255, cv2.THRESH_BINARY)[1]
        redimensionada = cv2.resize(binaria, (100, 100))  # Redimensionar para uniformidad
        return redimensionada.flatten()

    def preprocesar_dibujo(self):
        """
        Preprocesa el dibujo realizado en el canvas para su análisis.
        """
        # Crear una imagen en blanco
        imagen = Image.new('RGB', (250, 250), 'white')
        draw = ImageDraw.Draw(imagen)

        # Dibujar los puntos en la imagen
        for i in range(len(self.puntos) - 1):
            draw.line([self.puntos[i], self.puntos[i+1]], fill='black', width=2)
        
        # Guardar la imagen temporalmente y luego procesarla
        imagen.save("dibujo.png")

        return self.preprocesar_imagen("dibujo.png")

    def procesar_imagenes_entrenamiento(self):
        """
        Procesa todas las imágenes en la carpeta 'imagenesEntrenamiento' y guarda sus patrones.
        """
        carpeta_entrenamiento = 'imagenesEntrenamiento'
        patrones = []
        
        for nombre_archivo in os.listdir(carpeta_entrenamiento):
            ruta_archivo = os.path.join(carpeta_entrenamiento, nombre_archivo)
            patron = self.preprocesar_imagen(ruta_archivo)
            patrones.append((patron.tolist(), nombre_archivo))  # Convertir el patrón a lista
        
        with open(self.pesos_guardados, 'w') as archivo:
            json.dump(patrones, archivo)
        print(f"Patrones procesados y guardados en {self.pesos_guardados}")

    def cargar_patrones(self):
        """
        Carga los patrones guardados desde un archivo JSON.
        """
        try:
            with open(self.pesos_guardados, 'r') as archivo:
                arregloPatrones = json.load(archivo)
                print(f"Patrones cargados desde el archivo: {arregloPatrones}")  # Debugging print
                self.modelo_art = ART(rho=self.slider_vigilancia.get())  # Inicializar el modelo ART con el valor del slider
                self.modelo_art.pesos = [(np.array(patron), clase) for patron, clase in arregloPatrones]
                print(f"Pesos del modelo ART después de la carga: {self.modelo_art.pesos}")  # Debugging print
        except FileNotFoundError:
            self.modelo_art = None
            print("Archivo de patrones no encontrado. Modelo ART no inicializado.")  # Debugging print

    def entrenar_imagen(self):
        """
        Toma la imagen o el dibujo actual y lo añade como nuevo patrón al modelo.
        """
        # Preprocesar el dibujo o la imagen cargada
        patron = self.preprocesar_dibujo() if self.ruta_archivo_actual is None else self.preprocesar_imagen(self.ruta_archivo_actual)
        
        # Solicitar al usuario que ingrese el nombre de la clase para la nueva imagen
        nombre_clase = simpledialog.askstring("Input", "¿Qué es la imagen?", parent=self.root)
        
        if nombre_clase:
            # Guardar la imagen en la carpeta 'imagenesEntrenamiento'
            carpeta_entrenamiento = 'imagenesEntrenamiento'
            if not os.path.exists(carpeta_entrenamiento):
                os.makedirs(carpeta_entrenamiento)
            
            extension = ".png" if self.ruta_archivo_actual is None else os.path.splitext(self.ruta_archivo_actual)[1]
            ruta_guardado = os.path.join(carpeta_entrenamiento, nombre_clase + extension)
            
            if self.ruta_archivo_actual is None:
                imagen = Image.new('RGB', (250, 250), 'white')
                draw = ImageDraw.Draw(imagen)
                for i in range(len(self.puntos) - 1):
                    draw.line([self.puntos[i], self.puntos[i+1]], fill='black', width=2)
                imagen.save(ruta_guardado)
            else:
                self.imagen_original.save(ruta_guardado)
            
            nuevo_patron = (patron.tolist(), nombre_clase + extension)
            if os.path.exists(self.pesos_guardados):
                with open(self.pesos_guardados, 'r') as archivo:
                    patrones = json.load(archivo)
            else:
                patrones = []
            patrones.append(nuevo_patron)
            with open(self.pesos_guardados, 'w') as archivo:
                json.dump(patrones, archivo)
            self.cargar_patrones()
            messagebox.showinfo("Imagen Añadida", "La imagen ha sido añadida y el modelo ha sido actualizado.")

    def reconocer_patrones(self):
        """
        Reconoce el patrón de la imagen actual (cargada o dibujada) y muestra la imagen más similar.
        """
        patron = self.preprocesar_dibujo() if self.ruta_archivo_actual is None else self.preprocesar_imagen(self.ruta_archivo_actual)
        mejor_similitud = -1
        mejor_clase = None
        
        for peso, clase in self.modelo_art.pesos:
            similitud = self.modelo_art._calcular_similitud(patron, peso)
            if similitud > mejor_similitud:
                mejor_similitud = similitud
                mejor_clase = clase
        
        if mejor_clase:
            ruta_imagen_similar = os.path.join('imagenesEntrenamiento', mejor_clase)
            self.mostrar_imagen_similar(ruta_imagen_similar)
            messagebox.showinfo("Patrón Reconocido", "Patrón reconocido como: " + mejor_clase)
        else:
            messagebox.showinfo("Patrón no Reconocido", "No se encontró ningún patrón coincidente.")

    def iniciar_dibujo(self, event):
        """
        Inicia el dibujo en el canvas.
        """
        self.dibujando = True
        self.puntos = [(event.x, event.y)]

    def dibujar(self, event):
        """
        Dibuja en el canvas mientras el botón del ratón está presionado.
        """
        if self.dibujando:
            self.puntos.append((event.x, event.y))
            self.canvas.create_line(self.puntos[-2], self.puntos[-1], fill='black', width=2)

    def terminar_dibujo(self, event):
        """
        Finaliza el dibujo en el canvas.
        """
        self.dibujando = False

    def limpiar_canvas(self):
        """
        Limpia el canvas.
        """
        self.canvas.delete("all")
        self.ruta_archivo_actual = None

class ART:
    def __init__(self, rho=0.5):
        """
        Inicializa el modelo ART con el parámetro de vigilancia.
        """
        self.rho = rho  # Parámetro de vigilancia
        self.pesos = []

    def entrenar(self, patron, clase):
        """
        Entrena el modelo ART con un nuevo patrón y su clase correspondiente.
        """
        if not self.pesos:
            self.pesos.append((patron, clase))
        else:
            # Buscar el peso más cercano
            mejor_similitud = -1
            indice_mejor_peso = -1
            
            for i, (peso, _) in enumerate(self.pesos):
                similitud = self._calcular_similitud(patron, peso)
                if similitud > mejor_similitud:
                    mejor_similitud = similitud
                    indice_mejor_peso = i
            
            if mejor_similitud >= self.rho:
                self._actualizar_peso(patron, self.pesos[indice_mejor_peso][0], mejor_similitud, indice_mejor_peso, clase)
            else:
                self.pesos.append((patron, clase))

    def predecir(self, patron):
        """
        Predice la clase de un nuevo patrón basado en los pesos entrenados.
        """
        mejor_similitud = -1
        mejor_clase = None
        
        for peso, clase in self.pesos:
            similitud = self._calcular_similitud(patron, peso)
            if similitud > mejor_similitud:
                mejor_similitud = similitud
                mejor_clase = clase
        
        return mejor_clase if mejor_similitud >= self.rho else None

    def _es_patron_nuevo(self, patron, peso):
        """
        Determina si un patrón es suficientemente diferente para ser considerado nuevo
        bajo el peso y la vigilancia dada.
        """
        return self._calcular_similitud(patron, peso) <= self.rho

    def _actualizar_peso(self, patron, peso, similitud, indice_mejor_peso, clase):
        """
        Actualiza el peso del patrón coincidente con el nuevo patrón.
        """
        alpha = 0.5  # Factor de aprendizaje
        peso_nuevo = peso + alpha * (patron - peso)
        self.pesos[indice_mejor_peso] = (peso_nuevo, clase)

    def _calcular_similitud(self, patron1, patron2):
        """
        Calcula la similitud entre dos patrones.
        """
        return np.dot(patron1, patron2) / (np.linalg.norm(patron1) * np.linalg.norm(patron2))

if __name__ == "__main__":
    root = tk.Tk()
    app = AplicacionReconocimientoPatrones(root)
    root.mainloop()
