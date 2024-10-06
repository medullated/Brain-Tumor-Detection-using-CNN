from pathlib import Path
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage, filedialog, Label, END
import tensorflow as tf
from PIL import Image, ImageTk
import numpy as np

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"assets")

MODEL = tf.keras.layers.TFSMLayer(OUTPUT_PATH / Path(r"my_model"), call_endpoint='serving_default')

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

def changeOnHover(button, hover, leave):
    button.bind("<Enter>", func=lambda e: button.config(image=hover))
    button.bind("<Leave>", func=lambda e: button.config(image=leave))

def make_prediction(image, model):
    if image.mode != 'L':
        image = image.convert('L')
    image = image.resize((350, 350))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    image_array = np.expand_dims(image_array, axis=-1)
    

    predictions = model(image_array)
    prediction_value = float(predictions['dense_5'].numpy()[0][0])
    print(predictions)
    predicted_class = 'Tumor' if prediction_value > 0.009 else 'No Tumor'
    return predicted_class

def openfn():
    filename = filedialog.askopenfilename(title='open')
    return filename

def open_img():
    global MODEL, display
    x = openfn()
    img1 = Image.open(x)
    img = img1.resize((350, 350))
    display = ImageTk.PhotoImage(img)
    canvas.itemconfig(image, image=display)
    res = make_prediction(img1, MODEL)
    entry.config(state='normal')
    entry.delete(0, END)
    entry.insert(0, res)
    entry.config(state='disabled', disabledbackground='white', disabledforeground='black')

window = Tk()
window.geometry("850x600")
window.configure(bg = "#FFFFFF")
window.title("Выявление опухолей головного мозга на основе моделей глубокого обучения")

canvas = Canvas(
    window,
    bg = "#FFFFFF",
    height = 600,
    width = 850,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)

canvas.place(x = 0, y = 0)

pimg = PhotoImage(
    file=relative_to_assets("pattern.png"))
pattern = canvas.create_image(
    425.00000000000006,
    300.0000000000002,
    image=pimg
)

limg = PhotoImage(
    file=relative_to_assets("label.png"))
label = canvas.create_image(
    212.0,
    195.00000000000023,
    image=limg
)

bimg1 = PhotoImage(
    file=relative_to_assets("button_1.png"))
bhover = PhotoImage(
    file=relative_to_assets("button_2.png"))
button_1 = Button(
    image=bimg1,
    borderwidth=0,
    highlightthickness=0,
    command=open_img,
    relief="flat"
)
button_1.place(
    x=65.0,
    y=247.00000000000023,
    width=296.0,
    height=53.0
)


eimg = PhotoImage(
    file=relative_to_assets("result.png"))
entry_bg_1 = canvas.create_image(
    212.5,
    363.5000000000002,
    image=eimg
)
entry = Entry(
    bd=0,
    bg="#FFFFFF",
    fg="#000716",
    highlightthickness=0,
    state="disabled",
    disabledbackground="white",
    disabledforeground='black',
    font=("Arial", 18),
    justify="center"

)
entry.place(
    x=88.0,
    y=339.00000000000017,
    width=249.0,
    height=51.00000000000006
)

image = canvas.create_image(
    637.0,
    300.0000000000002,
    image=None,
    tag = "IMG"
)

changeOnHover(button_1, bhover, bimg1)
window.resizable(False, False)
window.mainloop()

