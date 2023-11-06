import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from model import SimpleCNN

# Cargar el modelo entrenado
model = SimpleCNN()  # Asegúrate de que SimpleCNN es la misma arquitectura que la que entrenaste
# Cargar los pesos del modelo
model.load_state_dict(torch.load("./model/model1.pth"))
model.eval()

# Función para preprocesar la imagen antes de la inferencia
def preprocess_image(image_path):
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image)
    return image

# Función para clasificar la imagen
def classify_image(image_path):
    image = preprocess_image(image_path)
    image = image.unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    return predicted.item()

# Función para cargar una imagen desde el celular y clasificarla
def load_and_classify_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
    if file_path:
        # Cargar la imagen y realizar la clasificación
        image = Image.open(file_path)
        image = image.resize((224, 224))  # Redimensionar la imagen al tamaño deseado
        img_label.img = ImageTk.PhotoImage(image)
        img_label.create_image(0, 0, anchor=tk.NW, image=img_label.img)
        prediction = classify_image(file_path)
        if prediction == 0:
            prediction_label.config(text="Resultado: Es un gato")
        else:
            prediction_label.config(text="Resultado: Es un conejo")

# Crear la ventana de la interfaz gráfica
window = tk.Tk()
window.title("Clasificador de Conejo o Gato")

# Configurar el tamaño y la posición de la ventana para centrarla en la pantalla
window.geometry("800x600")  # Tamaño de la ventana
window.eval('tk::PlaceWindow %s center' % window.winfo_pathname(window.winfo_id()))

# Botón para cargar una imagen
load_image_button = tk.Button(window, text="Cargar Imagen", command=load_and_classify_image)
load_image_button.pack()

# Canvas para mostrar la imagen
img_label = tk.Canvas(window, width=224, height=224)
img_label.pack()

# Etiqueta para mostrar la predicción
prediction_label = tk.Label(window, text="Resultado: ")
prediction_label.pack()

window.mainloop()



