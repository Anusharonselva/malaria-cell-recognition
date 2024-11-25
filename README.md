# Deep Neural Network for Malaria Infected Cell Recognition

## AIM

To develop a deep neural network for Malaria infected cell recognition and to analyze the performance.

## Problem Statement and Dataset
The problem at hand is the automatic classification of red blood cell images into two categories: parasitized and uninfected. Malaria-infected red blood cells, known as parasitized cells, contain the Plasmodium parasite, while uninfected cells are healthy and free from the parasite. The goal is to build a convolutional neural network (CNN) model capable of accurately distinguishing between these two classes based on cell images.

Traditional methods of malaria diagnosis involve manual inspection of blood smears by trained professionals, which can be time-consuming and error-prone. Automating this process using deep learning can significantly speed up diagnosis, reduce the workload on healthcare professionals, and improve the accuracy of detection.

Our dataset comprises 27,558 cell images, evenly split between parasitized and uninfected cells. These images have been meticulously collected and annotated by medical experts, making them a reliable source for training and testing our deep neural network.
## Neural Network Model
![1](https://github.com/user-attachments/assets/70f59d6b-3408-4dbb-ae0e-452e510e54b8)


## DESIGN STEPS

### STEP 1:
We begin by importing the necessary Python libraries, including TensorFlow for deep learning, data preprocessing tools, and visualization libraries.

### STEP 2:
To leverage the power of GPU acceleration, we configure TensorFlow to allow GPU processing, which can significantly speed up model training.
### STEP 3:
We load the dataset, consisting of cell images, and check their dimensions. Understanding the image dimensions is crucial for setting up the neural network architecture.
### STEP 4:
We create an image generator that performs data augmentation, including rotation, shifting, rescaling, and flipping. Data augmentation enhances the model's ability to generalize and recognize malaria-infected cells in various orientations and conditions.
### STEP 5:
We design a convolutional neural network (CNN) architecture consisting of convolutional layers, max-pooling layers, and fully connected layers. The model is compiled with appropriate loss and optimization functions.
### STEP 6:
We split the dataset into training and testing sets, and then train the CNN model using the training data. The model learns to differentiate between parasitized and uninfected cells during this phase.
### STEP 7:
We visualize the training and validation loss to monitor the model's learning progress and detect potential overfitting or underfitting.
### STEP 8:
We evaluate the trained model's performance using the testing data, generating a classification report and confusion matrix to assess accuracy and potential misclassifications.
### STEP 9:
We demonstrate the model's practical use by randomly selecting and testing a new cell image for classification.


## PROGRAM

### Name: S.ANUSHARON

### Register Number:212222240010
## Import necessary libraries:
```
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras import models
from sklearn.metrics import classification_report,confusion_matrix
import tensorflow as tf
```
## Allow GPU Processing:

```
from tensorflow.compat.v1.keras.backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True # dynamically grow the memory used on the GPU
config.log_device_placement = True # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
set_session(sess)
%matplotlib inline
```
## Read the images:
```
my_data_dir = './dataset/cell_images'
os.listdir(my_data_dir)
test_path = my_data_dir+'/test/'
train_path = my_data_dir+'/train/'

os.listdir(train_path)
len(os.listdir(train_path+'/uninfected/'))
len(os.listdir(train_path+'/parasitized/'))
os.listdir(train_path+'/parasitized')[0]

para_img= imread(train_path+
                 '/parasitized/'+
                 os.listdir(train_path+'/parasitized')[0])
plt.imshow(para_img)
```
## Check Image Dimensions:
```
# Checking the image dimensions
dim1 = []
dim2 = []
for image_filename in os.listdir(test_path+'/uninfected'):
    img = imread(test_path+'/uninfected'+'/'+image_filename)
    d1,d2,colors = img.shape
    dim1.append(d1)
    dim2.append(d2)

sns.jointplot(x=dim1,y=dim2)
image_shape = (130,130,3)
```
## Image Generator:
```
image_gen = ImageDataGenerator(rotation_range=20, # rotate the image 20 degrees
                               width_shift_range=0.10, # Shift the pic width by a max of 5%
                               height_shift_range=0.10, # Shift the pic height by a max of 5%
                               rescale=1/255, # Rescale the image by normalzing it.
                               shear_range=0.1, # Shear means cutting away part of the image (max 10%)
                               zoom_range=0.1, # Zoom in by 10% max
                               horizontal_flip=True, # Allo horizontal flipping
                               fill_mode='nearest' # Fill in missing pixels with the nearest filled value)

image_gen.flow_from_directory(train_path)
image_gen.flow_from_directory(test_path)
```
## Generate the model & compile:
```



## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

Include your plot here

### Classification Report

Include Classification Report here

### Confusion Matrix

Include confusion matrix here

### New Sample Data Prediction

Include your sample cell image input and output of your model.

## RESULT
