
# coding: utf-8

# # Densenet
# 
# In this week's assignment, you'll be using a pre-trained Densenet model for image classification. 
# 
# Densenet is a convolutional network where each layer is connected to all other layers that are deeper in the network
# - The first layer is connected to the 3rd, 4th etc.
# - The second layer is connected to the 3rd, 4th, 5th etc.
# 
# Like this:
# 
# <img src="images/densenet.png" alt="U-net Image" width="400" align="middle"/>
# 
# For a detailed explanation of Densenet, check out the source of the image above, a paper by Gao Huang et al. 2018 called [Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993.pdf).
# 
# The cells below are set up to provide an exploration of the Keras densenet implementation that you'll be using in the assignment. Run these cells to gain some insight into the network architecture. 

# In[1]:


# Import Densenet from Keras
from keras.applications.densenet import DenseNet121
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras import backend as K

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# For your work in the assignment, you'll be loading a set of pre-trained weights to reduce training time.

# In[2]:


# Create the base pre-trained model
base_model = DenseNet121(weights='./models/nih/densenet.hdf5', include_top=False);


# View a summary of the model

# In[3]:


# Print the model summary
base_model.summary()


# In[4]:


# Print out the first five layers
layers_l = base_model.layers

print("First 5 layers")
layers_l[0:5]


# In[5]:


# Print out the last five layers
print("Last 5 layers")
layers_l[-6:-1]


# In[6]:


# Get the convolutional layers and print the first 5
conv2D_layers = [layer for layer in base_model.layers 
                if str(type(layer)).find('Conv2D') > -1]
print("The first five conv2D layers")
conv2D_layers[0:5]


# In[7]:


# Print out the total number of convolutional layers
print(f"There are {len(conv2D_layers)} convolutional layers")


# In[8]:


# Print the number of channels in the input
print("The input has 3 channels")
base_model.input


# In[9]:


# Print the number of output channels
print("The output has 1024 channels")
x = base_model.output
x


# In[10]:


# Add a global spatial average pooling layer
x_pool = GlobalAveragePooling2D()(x)
x_pool


# In[11]:


# Define a set of five class labels to use as an example
labels = ['Emphysema', 
          'Hernia', 
          'Mass', 
          'Pneumonia',  
          'Edema']
n_classes = len(labels)
print(f"In this example, you want your model to identify {n_classes} classes")


# In[12]:


# Add a logistic layer the same size as the number of classes you're trying to predict
predictions = Dense(n_classes, activation="sigmoid")(x_pool)
print("Predictions have {n_classes} units, one for each class")
predictions


# In[13]:


# Create an updated model
model = Model(inputs=base_model.input, outputs=predictions)


# In[14]:


# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy')
# (You'll customize the loss function in the assignment!)


# #### This has been a brief exploration of the Densenet architecture you'll use in this week's graded assignment!
