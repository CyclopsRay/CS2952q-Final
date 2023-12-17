import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import cv2
import seaborn
seaborn.set_style('darkgrid')
import shutil
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Activation,Dropout,Conv2D, MaxPooling2D,BatchNormalization,ReLU
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import time
from sklearn.metrics import f1_score
import scipy.stats as stats
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

def EfficientNet_B0_Dense1024(img_size, lr, class_count):  
    img_shape=(img_size[0], img_size[1], 3)
    base_model=tf.keras.applications.efficientnet.EfficientNetB0(include_top=False, weights="imagenet",input_shape=img_shape, pooling='max')
    base_model.trainable=True
    x=base_model.output

    #classification head
    x=BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001 )(x)
    x=Dense(1024,activation='relu')(x)
    output=Dense(class_count, activation='sigmoid')(x)
    model=Model(inputs=base_model.input, outputs=output)
    model.compile(Adamax(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy']) 
    return model
