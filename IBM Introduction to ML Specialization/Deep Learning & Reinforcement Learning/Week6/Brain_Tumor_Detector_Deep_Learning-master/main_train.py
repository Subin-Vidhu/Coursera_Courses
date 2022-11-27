import cv2 
import os
import tensorflow as tf 
from PIL import Image
import numpy as np 
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import normalize 
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D,
     Activation, Dropout, Flatten, Dense)
from tensorflow.keras.utils import to_categorical

image_directory = 'datasets/'
no_tumor_images = os.listdir(image_directory + 'no/')
yes_tumor_images = os.listdir(image_directory + 'yes/')

dataset=[]
label=[]

INPUT_SIZE=64

for i, image_name in enumerate(no_tumor_images):
    if image_name.split(".")[1] == 'jpg':
        image = cv2.imread(image_directory+"no/"+image_name)
        image = Image.fromarray(image, "RGB")
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(0)

for i, image_name in enumerate(yes_tumor_images):
    if image_name.split(".")[1] == 'jpg':
        image = cv2.imread(image_directory+"yes/"+image_name)
        image = Image.fromarray(image, "RGB")
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(1)


dataset= np.array(dataset) 
label= np.array(label)

x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=0)

x_train = normalize(x_train, axis=1)
x_test = normalize(x_test, axis=1)

# Categorical Cross Entropy = 2 | comment these lines if you are using binary cross entropy 
# y_train = to_categorical(y_train, num_classes=2)
# y_test = to_categorical(y_test, num_classes=2)



# Model Building
model = Sequential()

# first layer
model.add(Conv2D(filters= 32, kernel_size=(3,3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

# 1st hidden layer
model.add(Conv2D(filters= 32, kernel_size=(3,3), kernel_initializer="he_uniform"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

# 2nd hidden layer
model.add(Conv2D(filters= 64, kernel_size=(3,3), kernel_initializer="he_uniform"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

# Flatten Layer 
model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))
model.add(Dropout(0.5))

# Binary Cross Entropy = 1, sigmoid 
model.add(Dense(1))
model.add(Activation("sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
 
# Categorical Cross Entropy = 2, softmax   
# model.add(Dense(2))
# model.add(Activation("softmax"))
# model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Fitting the model with the dataset 
model.fit(x_train, y_train, batch_size=32, verbose=True, epochs=10,
          validation_data=(x_test, y_test), shuffle=False)

model.save("BrainTumor10Epochs.h5")