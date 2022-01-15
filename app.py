# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 01:27:37 2022

@author: adwaid
"""

# Importing libraries 

import streamlit as st

import matplotlib.pyplot as plt

import seaborn as sns

import cv2

import os

import numpy as np

import pickle

import tensorflow as tf

from tensorflow.keras import layers

from tensorflow.keras import models,utils

import pandas as pd

from tensorflow.keras.models import load_model


from tensorflow.python.keras import utils

from keras.preprocessing import image

from PIL import Image

current_path = os.getcwd()


model1 = load_model(r'models\malaria.h5')



def load(filename):
    img = cv2.imread(filename)
    img = image.load_img(filename, target_size = (64,64))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    index = model1.predict(img)
    index = index.argmax().item()
    if index == 0:
        return 0
    elif index == 1:
        return 1
    else :
        return 
    


sns.set_theme(style="darkgrid")

sns.set()


st.title('Malaria Blood Cell Detection')


def save_uploaded_file(uploaded_file):

    try:

        with open(os.path.join('models/images',uploaded_file.name),'wb') as f:

            f.write(uploaded_file.getbuffer())

        return 1    

    except:

        return 0
    

uploaded_file = st.file_uploader("Upload Image")

# text over upload button "Upload Image"

if uploaded_file is not None:

    if save_uploaded_file(uploaded_file): 

        # display the image

        display_image = Image.open(uploaded_file)

        st.image(display_image)

        prediction = load(os.path.join('models/images',uploaded_file.name))
        
        if prediction==0:
            st.write("The given blood cell is Parasited")
        else :
            st.write("The given blood cell is Uninfected")
        os.remove('models/images/'+uploaded_file.name)

        # deleting uploaded saved picture after prediction

    