import pandas as pd
import numpy as np
import torch
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
from vonenet import VOneNet
from backend import EfficientNet_B0_Dense1024

from models.sampler import DiffusionSampler
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

model_name = "V+E"
model_name = "E"
model_name = "D+E"
model_name = "D+V+E"

#   ---------load data ---------------------- For VOne and Efficient Net.
# This is the kaggle dataset path
datapath=r'./attack_input/labels.csv'
if model_name == "V+E":
    imgpath=r'./attack_input/vone_eps5'
elif model_name == "E":
    imgpath=r'./attack_input/eps5'
elif model_name == "D+E":
    imgpath=r'./attack_input/diffusion_eps5'
elif model_name == "D+V+E":
    imgpath=r'./attack_input/all_eps5'

# datapath=r'pizza_data/labels.csv'
# imgpath=r'pizza_data/images'
df=pd.read_csv(datapath)
df=df.sample(n=9123, replace=False, random_state = 1)
df['plain']=0
columns=df.columns

df['image_name']=df['image_name'].apply(lambda x: os.path.join(imgpath,x))
df = df.rename(columns={'image_name': 'filepaths'})

for i in range (len(df)):
    label_list=[]
    for j in range (1,len(df.columns)-1):
        column=df.columns[j]        
        label=df.iloc[i][column]
        label_list.append(label)
    max=np.max(label_list)        
    if max == 0:         
        df['plain'].iloc[i]=1
train_df, dummy_df=train_test_split(df, train_size=.8, shuffle=True, random_state=123)
valid_df, test_df=train_test_split(dummy_df, train_size=.5, shuffle=True, random_state=123)
print('train_df length: ', len(train_df), '  test_df length: ', len(test_df), '  valid_df length: ', len(valid_df)) 
img_size=(300,300)
columns=df.columns[1:]
class_count=len(columns)
generator=ImageDataGenerator()
train=generator.flow_from_dataframe(train_df, x_col='filepaths', y_col=columns, target_size=img_size, batch_size=30,shuffle=True,seed=123, class_mode='raw')
val=generator.flow_from_dataframe(valid_df, x_col='filepaths', y_col=columns, target_size=img_size, batch_size=30,shuffle=False,class_mode='raw')
test=generator.flow_from_dataframe(valid_df, x_col='filepaths', y_col=columns, target_size=img_size, batch_size=30,shuffle=False,class_mode='raw')



# ---------------------------- For diffusion model-------------------
# Convert the DataFrame to PyTorch Tensor
def df_to_tensor(df, img_size):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(img_size),
        transforms.ToTensor()
    ])

    images = []
    labels = []

    for i, row in df.iterrows():
        img = cv2.imread(row['filepaths'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = transform(img)

        label = torch.tensor(row[1:-1].values.astype(np.float32))
        images.append(img)
        labels.append(label)

    return torch.stack(images), torch.stack(labels)

# Prepare the dataset and DataLoader for PyTorch
train_images, train_labels = df_to_tensor(train_df, img_size)
valid_images, valid_labels = df_to_tensor(valid_df, img_size)
test_images, test_labels = df_to_tensor(test_df, img_size)

train_dataset = TensorDataset(train_images, train_labels)
valid_dataset = TensorDataset(valid_images, valid_labels)
test_dataset = TensorDataset(test_images, test_labels)

train_loader = DataLoader(train_dataset, batch_size=30, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=30, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=30, shuffle=False)


# ---------------------------------------------------------------------- Now comes something serious --------------------------------------------------------
if model_name == "V+E:
    lr=.001
    if 
    # Test VOneNet and Efficient Net. Those 2 models are writen in Tensorflow.
    model=VOneNet()
    # model=DiffusionSampler()
    # model=VOneNet_with_four_GFBs()
    # model=VOneNet_with_eight_GFBs()
    # model=VOneNet_with_Only_Simple_GFBs()
    # model=EfficientNet_B0_Dense1024(img_size, lr, class_count)
    epoch = 10
    history=model.fit(x=train,  epochs=epochs, verbose=1, validation_data=val, shuffle=True,  initial_epoch=0)
    print(history.history)
    results = model.evaluate(x=test)
    print(results)
elif model_name == "E":
    lr=.001
    if 
    # Test VOneNet and Efficient Net. Those 2 models are writen in Tensorflow.
    # model=VOneNet()
    # model=DiffusionSampler()
    # model=VOneNet_with_four_GFBs()
    # model=VOneNet_with_eight_GFBs()
    # model=VOneNet_with_Only_Simple_GFBs()
    model=EfficientNet_B0_Dense1024(img_size, lr, class_count)
    epoch = 10
    history=model.fit(x=train,  epochs=epochs, verbose=1, validation_data=val, shuffle=True,  initial_epoch=0)
    print(history.history)
    results = model.evaluate(x=test)
    print(results)
# Test Diffusion model. It is writen in Pytorch.
elif model_name = "D+V+E":
    model = DiffusionSampler()

    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(10):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation step
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()

        print(f"Epoch {epoch+1}, Validation Loss: {valid_loss / len(valid_loader)}")

    # Testing the model
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

    print(f"Test Loss: {test_loss / len(test_loader)}")
