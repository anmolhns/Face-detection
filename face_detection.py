#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import os
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import cv2
from scipy.spatial import distance


# In[40]:






face_model = cv2.CascadeClassifier('vac_hes/Gaza/mask/haarcascade_frontalface_default.xml')
import matplotlib.pyplot as plt

img = cv2.imread('vac_hes/Gaza/mask/images/GOPR1443.JPG')
img = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)
faces = face_model.detectMultiScale(img,scaleFactor=1.2, minNeighbors=2, minSize=(100, 50))
#returns a list of (x,y,w,h) tuples
img_2 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #converted to colored o/p image
for (x,y,w,h) in faces:
    cv2.rectangle(img_2,(x,y),(x+w,y+h),(250,0,0),40)
    
plt.figure(figsize=(12,12))
plt.imshow(img_2)


# In[60]:


MIN_DISTANCE = 1000
print(len(faces))

if len(faces)>=2:
    label = [0 for i in range (len(faces))]
    for i in range(len(faces)-1):
        for j in range(i+1, len(faces)):
            dist = distance.euclidean(faces[i][:2], faces[j][:2])#calculating the distance between people 
            if dist<MIN_DISTANCE:
                label[i] = 1
                label[j] = 1
                
    new_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for i in range(len(faces)):
        (x,y,w,h)=faces[i]
        if label[i]==1:
            cv2.rectangle(new_img,(x,y), (x+w, y+h),(255,0,0),15)#if distance<MIN_DISTANCE then red box showing violation of social distancing
        else:
            cv2.rectangle(new_img,(x,y),(x+w,y+h),(0,255,0),15)#else green box showing Social distancing followed
    plt.figure(figsize=(10,10))
    plt.imshow(new_img)
else:
    print("No. of faces detected is less than 2")

   




