#!/usr/bin/env python
# coding: utf-8

# In[12]:


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


# ## First, change the newPath variable below to point to the location of the tfrecords file, then run all of the cells in order to get the results. You can also comment and uncomment some parts to tune hyperparameters

# In[13]:


newPath = "./audioset_v1_embeddings/myFiltered/"
files = os.listdir( newPath )

df = pd.DataFrame(columns=["ytid", "Speech", "Animal", "Music", "Vehicle"])

oneFile=[]
allFiles=[]
myLabels=[]
myLabelsAll=[]
multiLabels=[]
ytid=[]
ytidMulti=[]
count=0;
ind=0;

for file in files:
    secCount=0
    
    tfrecords_filename = newPath+file

    record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)

    string_record = next(record_iterator)

    example = tf.train.SequenceExample()

    example.ParseFromString(string_record)    
    
    #to store the id of the video for later use
    vidId=example.context.feature['video_id'].bytes_list.value[0].decode()
    
    label=example.context.feature['labels'].int64_list.value[:]       
    
    for a in example.feature_lists.feature_list['audio_embedding'].feature:
        secCount+=1        
    #print(secCount)
    
    if secCount == 10: # make sure all samples are same length
        if 0 in label or 72 in label or 137 in label or 300 in label:
            df.loc[ind, ['ytid']]=vidId
            #df=df.append({"ytid":vidId, "Speech":0, "Animal":0, "Music":0, "Vehicle":0}, ignore_index=True)

            if 0 in label:
                df.loc[ind, ['Speech']]=1
            #    df.iloc[ind,1]=1
            else:
                df.loc[ind, ['Speech']]=0
                #df.iloc[ind,1]=0
            if 72 in label:
                df.loc[ind, ['Animal']]=1
                #df.iloc[ind,2]=1
            else:
                df.loc[ind, ['Animal']]=0
               #df.iloc[ind,2]=0
            if 137 in label:
                df.loc[ind, ['Music']]=1
                #df.iloc[ind,3]=1
            else:
                df.loc[ind, ['Music']]=0            
                #df.iloc[ind,3]=0
            if 300 in label:
                df.loc[ind, ['Vehicle']]=1
                #df.iloc[ind,4]=1
            else:
                df.loc[ind, ['Vehicle']]=0
                #df.iloc[ind,4]=0

            ind+=1

            for a in example.feature_lists.feature_list['audio_embedding'].feature:
                    # 960ms of data
                hexembed = a.bytes_list.value[0].hex()

                arrayembed = [int(hexembed[i:i+2],16) for i in range(0,len(hexembed),2)]         

                allFiles.append(arrayembed)

    


# In[4]:


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


# In[5]:


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


# In[6]:


def shuffling(allSamples,labels,sampleToShuffle, vidId, option):
    X=allSamples    
    Y=labels
    x1=sampleToShuffle
    #y1=sampleLabel
    
    r=[randrange(10) for _ in range(0,10)]
    for num in range(10):
        if(num < int(len(r)/2)):
            #print('1st is {0}, 2nd is {1}'.format(x[(len(x)-1)-num],num))
            temp=x1[num].copy()
            x1[num]=x1[r[num]].copy()#r[num] is the index num in the list r of random numbers between 1 and 10
            x1[r[num]]=temp.copy()    


    X=np.append(X, x1.reshape(1,10,128,1), axis=0) 
    
    if option==1:#value to add to is Animal
        Y=Y.append({"ytid":vidId, "Speech":0, "Animal":1, "Music":0, "Vehicle":0}, ignore_index=True)
    if option==2:#value to add to is Vehicle
        Y=Y.append({"ytid":vidId, "Speech":0, "Animal":0, "Music":0, "Vehicle":1}, ignore_index=True)

    #print(len(X))
    
    return X, Y


# In[7]:


X = np.array(allFiles)
X=X.reshape(int(len(X)/10),10,128,1) # no of 10 s clips, first dimension, 2nd dimension, 1 channel

count72=0
count300=0

while count72<=300: # count72 is for Animal label
    count72=0
    count300=0
    for i, x in enumerate(X):
        if df.iloc[i, 2] == 1 and count72<=300: #df.iloc[i,2] is animal
            count72+=1
            X, df =shuffling(X, df, X[i],df.iloc[i, 0], 1)
            #ytidNew.append(ytidNew[i])
        if df.iloc[i, 4] == 1 and count300<=300:#df.iloc[i,2] is vehicle
            count300+=1
            X, df =shuffling(X, df, X[i], df.iloc[i, 0], 2)
            #ytidNew.append(ytidNew[i])
    if count72>300:
        break

X=X/255 # for kind of normalization

np.save('XFullForMulti',X)
df.to_csv('theLabels.csv')

#Xt = tf.convert_to_tensor(X, dtype=tf.float32)


# In[61]:


myInput=Input(shape=(10,128,1))

#Xt=Input(shape=Xt.shape[1:4])

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


# In[65]:


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


# In[63]:



multiModel.fit(X,{"speech_output": df['Speech'], "animal_output": df['Animal'], "music_output": df['Music'], "vehicle_output": df['Vehicle']}, epochs=5)


# In[66]:


multiModelONN.fit(X2,{"speech_output2": df['Speech'], "animal_output2": df['Animal'], "music_output2": df['Music'], "vehicle_output2": df['Vehicle']}, epochs=5)


# In[69]:


pred = multiModelONN.predict(X2[:10])
print(pred)
print(df['Speech'][0:3])
print(df['Animal'][0:3])
print(df['Music'][0:3])
print(df['Vehicle'][0:3])


# In[ ]:





# In[ ]:




