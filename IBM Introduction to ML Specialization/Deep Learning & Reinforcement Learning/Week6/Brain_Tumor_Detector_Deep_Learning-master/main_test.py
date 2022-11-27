import cv2 
import os
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image


model = load_model("BrainTumor10Epochs.h5")

image = cv2.imread("datasets\pred\pred8.jpg")
img = Image.fromarray(image)
img = img.resize((64,64))
img = np.array(img) 
input_img = np.expand_dims(img, axis=0)

result = model.predict_classes(input_img)
print(result)