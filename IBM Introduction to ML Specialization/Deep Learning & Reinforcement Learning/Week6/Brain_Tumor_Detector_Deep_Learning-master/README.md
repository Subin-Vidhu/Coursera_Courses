# Brain Tumors Detection Using Deep Learning 🧠 

<div align="center">
<p>
<img src="readme_images/project_display.gif" width="700" height="400"/>
</p>
<br>
<div>
</div>
</div>

## Introduction
This repository contains a deep learning model based on a convolutional neural network (CNN) used to detect brain tumors from MRI images. 

There are two pre-trained models in this repo :

1. Binary Cross entropy :  `BrainTumor10Epochs.h5` 
2. Categorical Cross Entropy : `BrainTumorCategorical10Epochs.h5`


## Before you run the model

1. Clone the repository recursively:

    * `git clone --recurse-submodules https://github.com/AI-MOO/Brain_Tumor_Detector_Deep_Learning.git`

2. Make sure that you fulfill all the requirements: Python 3.6.8 or later with all packages in `requirements.txt`

3. Run the model through flask application module : `app.py` in your terminal 

    ```
    python app.py
    ```

## Dataset 

1. Br35H : Brain Tumor Detection 2020
    * [Brain Tumor Detection Download Link](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection)

    * Extract the images in `datasets` folder
 
    * `pip install -r requirements.txt`
2. to train your custom model use `main_train.py` module 


## Credit & Tutorial 

* [KNOWLEDGE DOCTOR on Youtube](https://www.youtube.com/watch?v=pp61TbhJOTg&list=PLWyN7K28ZraStL8fr0eQmr6VwAiahQStd&index=2)


