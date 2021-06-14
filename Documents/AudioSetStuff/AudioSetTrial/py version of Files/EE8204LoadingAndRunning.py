#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import np_utils # need pip install
from random import randrange
from IPython.display import YouTubeVideo

from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Lambda
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
import tensorflow as tf


import pandas as pd
import numpy as np
import os
from tensorflow import keras
from random import randrange
from IPython.display import YouTubeVideo
import keras_metrics


# In[9]:


CNNModel(100, 5) #batch size, epochs


# In[8]:


ONNModel(20, 10) #batch size, epochs


# In[10]:


multiCNN()


# In[11]:


multiONN()


# In[2]:


def forSpeech(data):    
    
    model=Conv2D(32, (3, 3), padding="same")(data)
    #model=Conv2D(16, kernel_size=(1, 6), strides=(2, 2), activation='relu',input_shape=(10,128,1))(data)
    model=MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(model)
    
    model=Conv2D(32, kernel_size=(1, 6), strides=(2, 2), activation='relu',input_shape=(10,128,1))(model)
    model=MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(model)
    
    model=Conv2D(32, kernel_size=(1, 6), strides=(2, 2), activation='relu',input_shape=(10,128,1))(model)
    
    model=Flatten()(model)
    model=Dense(1000, activation='relu')(model)
    #model=Dense(1, activation='softmax', name="speech_output")(model)
    model=Dense(2, activation='softmax', name="speech_output")(model)

    return model

def forAnimal(data):
    
    model=Conv2D(16, kernel_size=(1, 6), strides=(2, 2), activation='relu',input_shape=(10,128,1))(data)
    model=MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(model)
    
    model=Conv2D(32, kernel_size=(1, 6), strides=(2, 2), activation='relu',input_shape=(10,128,1))(model)
    model=MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(model)
    
    model=Conv2D(32, kernel_size=(1, 6), strides=(2, 2), activation='relu',input_shape=(10,128,1))(model)
    
    model=Flatten()(model)
    model=Dense(1000, activation='relu')(model)
    #model=Dense(1, activation='softmax', name="animal_output")(model)
    model=Dense(2, activation='softmax', name="animal_output")(model)
    

    
    return model

def forMusic(data):
    
    model=Conv2D(16, kernel_size=(1, 6), strides=(2, 2), activation='relu',input_shape=(10,128,1))(data)
    model=MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(model)
    
    model=Conv2D(32, kernel_size=(1, 6), strides=(2, 2), activation='relu',input_shape=(10,128,1))(model)
    model=MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(model)
    
    model=Conv2D(32, kernel_size=(1, 6), strides=(2, 2), activation='relu',input_shape=(10,128,1))(model)
    
    model=Flatten()(model)
    model=Dense(1000, activation='relu')(model)
    #model=Dense(1, activation='softmax', name="music_output")(model)   
    model=Dense(2, activation='softmax', name="music_output")(model)   
    
    return model

def forVehicle(data):
    
    model=Conv2D(16, kernel_size=(1, 6), strides=(2, 2), activation='relu',input_shape=(10,128,1))(data)
    model=MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(model)
    
    model=Conv2D(32, kernel_size=(1, 6), strides=(2, 2), activation='relu',input_shape=(10,128,1))(model)
    model=MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(model)
    
    model=Conv2D(32, kernel_size=(1, 6), strides=(2, 2), activation='relu',input_shape=(10,128,1))(model)
    
    model=Flatten()(model)
    model=Dense(1000, activation='relu')(model)
    #model=Dense(1, activation='softmax', name="vehicle_output")(model)    
    model=Dense(2, activation='softmax', name="vehicle_output")(model)    
    
    return model

def forSpeechONN(data):        
    
    model2 = Dense(10, activation='relu',input_shape=(1280,))(data)
    model2 = Dense(100, activation='relu')(model2)
    model2 = Dense(1000, activation='relu')(model2)
    #model2 = Dense(1, activation='softmax', name="speech_output2")(model2)
    model2 = Dense(2, activation='softmax', name="speech_output2")(model2)

    return model2


def forAnimalONN(data):    
    
    model2 = Dense(10, activation='relu',input_shape=(1280,))(data)
    model2 = Dense(100, activation='relu')(model2)
    model2 = Dense(1000, activation='relu')(model2)
    #model2 = Dense(1, activation='softmax', name="animal_output2")(model2)
    model2 = Dense(2, activation='softmax', name="animal_output2")(model2)

    return model2

def forMusicONN(data):    
    
    model2 = Dense(10, activation='relu',input_shape=(1280,))(data)
    model2 = Dense(100, activation='relu')(model2)
    model2 = Dense(1000, activation='relu')(model2)
    #model2 = Dense(1, activation='softmax', name="music_output2")(model2)
    model2 = Dense(2, activation='softmax', name="music_output2")(model2)

    return model2

def forVehicleONN(data):    
    
    model2 = Dense(10, activation='relu',input_shape=(1280,))(data)
    model2 = Dense(100, activation='relu')(model2)
    model2 = Dense(1000, activation='relu')(model2)
    #model2 = Dense(1, activation='softmax', name="vehicle_output2")(model2)
    model2 = Dense(2, activation='softmax', name="vehicle_output2")(model2)

    return model2

def multiONN():
    X=np.load('XFullForMulti.npy')
    df=pd.read_csv('theLabels.csv')
    
    X2=X.reshape(len(X), 10*128)


    myInputONN=Input(shape=(10*128,))

    spONN = forSpeechONN(myInputONN)
    anONN = forAnimalONN(myInputONN)
    muONN = forMusicONN(myInputONN)
    veONN = forVehicleONN(myInputONN)

    # losses2 = {
    # "speech_output2": "binary_crossentropy",
    # "animal_output2": "binary_crossentropy",
    # "music_output2": "binary_crossentropy",
    # "vehicle_output2": "binary_crossentropy"
    # }

    losses2 = {
    "speech_output2": "sparse_categorical_crossentropy",
    "animal_output2": "sparse_categorical_crossentropy",
    "music_output2": "sparse_categorical_crossentropy",
    "vehicle_output2": "sparse_categorical_crossentropy"
    }

    theClassesONN = [spONN,anONN,muONN,veONN]

    multiModelONN = Model(inputs=myInputONN, outputs=theClassesONN)
    print("[INFO] compiling model for ONN...")
    multiModelONN.compile(optimizer='adam', loss=losses2, metrics=[keras_metrics.precision()])
    multiModelONN.fit(X2,{"speech_output2": df['Speech'], "animal_output2": df['Animal'], "music_output2": df['Music'], "vehicle_output2": df['Vehicle']}, epochs=5)
    


# In[3]:


def multiCNN():
    X=np.load('XFullForMulti.npy')
    df=pd.read_csv('theLabels.csv')
    
    myInput=Input(shape=(10,128,1))

    sp = forSpeech(myInput)
    an = forAnimal(myInput)
    mu = forMusic(myInput)
    ve = forVehicle(myInput)

    theClasses = [sp,an,mu,ve]

    multiModel = Model(inputs=myInput, outputs=theClasses)

    # losses = {
    # "speech_output": "categorical_crossentropy",
    # "animal_output": "categorical_crossentropy",
    # "music_output": "categorical_crossentropy",
    # "vehicle_output": "categorical_crossentropy"
    # }

    # losses = {
    # "speech_output": "binary_crossentropy",
    # "animal_output": "binary_crossentropy",
    # "music_output": "binary_crossentropy",
    # "vehicle_output": "binary_crossentropy"
    # }

    losses = {
    "speech_output": "sparse_categorical_crossentropy",
    "animal_output": "sparse_categorical_crossentropy",
    "music_output": "sparse_categorical_crossentropy",
    "vehicle_output": "sparse_categorical_crossentropy"
    }


    # initialize the optimizer and compile the model
    print("[INFO] compiling model...")
    multiModel.compile(optimizer='adam', loss=losses, metrics=[keras_metrics.precision()])
    
    multiModel.fit(X,{"speech_output": df['Speech'], "animal_output": df['Animal'], "music_output": df['Music'], "vehicle_output": df['Vehicle']}, epochs=5)
    
def multiONN():
    X=np.load('XFullForMulti.npy')
    df=pd.read_csv('theLabels.csv')
    
    X2=X.reshape(len(X), 10*128)


    myInputONN=Input(shape=(10*128,))

    spONN = forSpeechONN(myInputONN)
    anONN = forAnimalONN(myInputONN)
    muONN = forMusicONN(myInputONN)
    veONN = forVehicleONN(myInputONN)

    # losses2 = {
    # "speech_output2": "binary_crossentropy",
    # "animal_output2": "binary_crossentropy",
    # "music_output2": "binary_crossentropy",
    # "vehicle_output2": "binary_crossentropy"
    # }

    losses2 = {
    "speech_output2": "sparse_categorical_crossentropy",
    "animal_output2": "sparse_categorical_crossentropy",
    "music_output2": "sparse_categorical_crossentropy",
    "vehicle_output2": "sparse_categorical_crossentropy"
    }

    theClassesONN = [spONN,anONN,muONN,veONN]

    multiModelONN = Model(inputs=myInputONN, outputs=theClassesONN)
    print("[INFO] compiling model for ONN...")
    multiModelONN.compile(optimizer='adam', loss=losses2, metrics=[keras_metrics.precision()])
    multiModelONN.fit(X2,{"speech_output2": df['Speech'], "animal_output2": df['Animal'], "music_output2": df['Music'], "vehicle_output2": df['Vehicle']}, epochs=5)
    


# In[7]:


def CNNModel(b, e):
    
    xtr = np.load('XTrain.npy')
    ytr=np.load('YTrain.npy')
    xte=np.load('Xtest.npy')
    yte = np.load('Ytest.npy')
    ytidNew=np.load('ytidNew.npy')
    X=np.load('XFull.npy')
    dummy_y=np.load('encoded.npy')
    model = keras.Sequential()

    model.add(keras.layers.Conv2D(16, kernel_size=(1, 6), strides=(2, 2), activation='relu',input_shape=(10,128,1)))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(keras.layers.Conv2D(32, kernel_size=(1, 6), strides=(1, 1), activation='relu',padding="same"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(keras.layers.Conv2D(32, kernel_size=(1, 6), strides=(1, 1), activation='relu',padding="same"))
    #model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))# testing

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1000, activation='relu'))
    model.add(keras.layers.Dense(4, activation='softmax'))
    
    #model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    #model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    ### 
    #batch size 20 with 10 epochs was good
    model.fit(xtr, np.array(ytr), shuffle=True, batch_size=b, epochs=e)

    test_loss, test_acc = model.evaluate(xte, np.array(yte))
    print(test_acc)
    
    # just predicting first 20 videos
    pred = model.predict(X[0:20])
    chosen = []
    ind=0
    for x in pred:
        if np.where(x == (max(x)))[0]==0:
            chosen.append('human speaker')
        if np.where(x == (max(x)))[0]==1:
            chosen.append('animal')
        if np.where(x == (max(x)))[0]==2:
            chosen.append('music')
        if np.where(x == (max(x)))[0]==3:
            chosen.append('vehicle')

    #[human speaker, animal, music, vehicle]  
    print("\nthe order of the Real labels is [speech, animal, music, vehicle]\n")
    for i in ytidNew[0:20]:    
        print('{0}.) youtube id {1} has (a/an) {2} , Real label is {3}'.format(ind, i, chosen[ind], dummy_y[ind]))
        ind+=1

    r = randrange(0,20)
    #https://www.youtube.com/watch?v=
    print('\n ========= \n youtube video to verify prediction no {0} ({1}): \n'.format(r, chosen[r]))
    print("\n to verify the prediction you can go to https://www.youtube.com/watch?v={0}".format(ytidNew[r]))
    YouTubeVideo(ytidNew[r]) # functionality imported from an IPython.display library
    
def ONNModel(b, e):
    
    xtr = np.load('XTrain.npy')
    ytr=np.load('YTrain.npy')
    xte=np.load('Xtest.npy')
    yte = np.load('Ytest.npy')
    ytidNew=np.load('ytidNew.npy')
    X=np.load('XFull.npy')
    dummy_y=np.load('encoded.npy')
    model2 = keras.Sequential()
    model2.add(keras.layers.Dense(10, activation='relu',input_shape=(1280,)))
    model2.add(keras.layers.Dense(100, activation='relu'))
    model2.add(keras.layers.Dense(1000, activation='relu'))
    model2.add(keras.layers.Dense(4, activation='softmax'))

    model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    xtr2=xtr.reshape(len(xtr),10*128)
    xte2=xte.reshape(len(xte),10*128)
    print("\n=======================\n using 1280/10-100-1000 ONN\n=================\n")
    model2.fit(xtr2, np.array(ytr), shuffle=True, batch_size=b, epochs=e)

    test_loss, test_acc = model2.evaluate(xte2, np.array(yte))
    #test_loss, test_acc = model.evaluate(xte, multiLabelsNew[2000:])
    print(test_acc)

    ###### just predicting first 20 videos
    pred = model2.predict(X[0:20].reshape(len(X[0:20]),10*128))
    chosen = []
    ind=0
    for x in pred:
        if np.where(x == (max(x)))[0]==0:
            chosen.append('human speaker')
        if np.where(x == (max(x)))[0]==1:
            chosen.append('animal')
        if np.where(x == (max(x)))[0]==2:
            chosen.append('music')
        if np.where(x == (max(x)))[0]==3:
            chosen.append('vehicle')

    #[human speaker, animal, music, vehicle] 
    print("\n =========================using 1280/10-100-1000 ONN ========================== \n")
    print("\nthe order of the Real labels is [speech, animal, music, vehicle]\n")
    for i in ytidNew[0:20]:    
        print('{0}.) youtube id {1} has (a/an) {2} , Real label is {3}'.format(ind, i, chosen[ind], dummy_y[ind]))
        ind+=1

    r = randrange(0,20)

    print('\n ========= \n youtube video to verify prediction no {0} ({1}): \n'.format(r, chosen[r]))
    print("\n to verify the prediction you can go to https://www.youtube.com/watch?v={0}".format(ytidNew[r]))
    YouTubeVideo(ytidNew[r]) # functionality imported from an IPython.display library


# In[ ]:




