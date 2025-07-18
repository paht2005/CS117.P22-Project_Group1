
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

model = load_model("models/segmenter.h5")
IMG_SIZE = (256, 256)

def segment_fire(image_path):
    img = Image.open(image_path).resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    mask = model.predict(img_array)[0]
    binary_mask = (mask[:, :, 0] > 0.5).astype(np.uint8) * 255
    return binary_mask
