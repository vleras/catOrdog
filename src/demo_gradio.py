import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("./saved_model/cnn_cat_dog.h5")

def predict(img):
    img = img.resize((64, 64))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    result = model.predict(img)[0][0]

    if result >= 0.5:
        return "🐶 Dog"
    else:
        return "🐱 Cat"

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Cat vs Dog Classifier",
    description="Upload an image of a cat or dog."
)

demo.launch()