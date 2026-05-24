import gradio as gr
import cv2
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

model = load_model("garbage_classifier.keras")

class_names = [
   'clothes',
 'shoes',
 'paper',
 'trash',
 'white-glass',
 'brown-glass',
 'biological',
 'green-glass',
 'cardboard',
 'battery',
 'plastic',
 'metal'
]

def predict(image):

    img = cv2.resize(image, (224,224))

    img = preprocess_input(img)

    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    label = class_names[np.argmax(prediction)]

    confidence = np.max(prediction)

    return f"{label} ({confidence:.2f})"

interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(),
    outputs="text",
    title="Garbage Classification AI",
    description="Upload garbage image for classification"
)

interface.launch()