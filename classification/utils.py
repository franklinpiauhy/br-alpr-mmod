# -*- coding: utf-8 -*-
"""
Created on Wed May  1 14:55:27 2019

@author: lucas
"""

import glob
import keras
from keras import layers
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
import cv2
import numpy as np
import string
#from matplotlib import pyplot as plt

digit_encoder = LabelBinarizer()
digit_encoder.fit(list(string.digits))
letter_encoder = LabelBinarizer()
letter_encoder.fit(list(string.ascii_uppercase))

kernel_sharpening = np.array([[-1,-1,-1], 
                              [-1, 9,-1],
                              [-1,-1,-1]])

def LeNet5(n_classes):
    model = keras.Sequential()
    
    model.add(layers.Conv2D(filters=6, kernel_size=(3, 3), input_shape=(28,28,1), padding='same'))
    model.add(layers.normalization.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.AveragePooling2D())
    
    model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), padding='valid'))
    model.add(layers.normalization.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.AveragePooling2D())
    
    model.add(layers.Flatten())
    
    model.add(layers.Dense(units=120))
    model.add(layers.normalization.BatchNormalization())
    model.add(layers.Activation('relu'))
    
    model.add(layers.Dense(units=84))
    model.add(layers.normalization.BatchNormalization())
    model.add(layers.Activation('relu'))
    
    model.add(layers.Dense(units=n_classes, activation = 'softmax'))
    
    #sgd = keras.optimizers.SGD(lr=0.1, nesterov=True)
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer='sgd', metrics=['accuracy'])
    
    return model 

def load_data(directory):
    X_let = []
    Y_let = []
    X_dig = []
    Y_dig = []
    for f in glob.glob(directory+"**\\*.png", recursive=True):
        metadata = f.split('.')[0] + '.txt'
        with open(metadata, 'r') as file:
            for i in range(0,2): file.readline()
            type_veic = file.readline().split(":")[-1].split()[0]
            if  type_veic != 'car':
                continue
            for i in range(3,6): file.readline()
            plate_text = file.readline().split(":")[-1].split()[0]
            plate_text = plate_text.replace('-', '')
            file.readline()
            chars = [[int(x) for x in file.readline().split(":")[-1].split()] for i in range(0,7)]
        i=0
        for char in chars:
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            x, w, y, h = char[0], char[2], char[1], char[3] # char width
            #y, h = pos[1], pos[3] # char height
            img = img[y:y+h, x:x+w]
            img = cv2.resize(img, (28,28))
            img = cv2.equalizeHist(img)
            #img = cv2.filter2D(img, -1, kernel_sharpening)
            #img = cv2.Laplacian(img,cv2.CV_8U)
            _, img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            #plt.imshow(img, cmap='gray')
            #plt.title(plate_text[i])
            #plt.show()
            if i<3:
                X_let.append(img_to_array(img)/255.0)
                Y_let.append(letter_encoder.transform([plate_text[i]])[0])
            else:
                X_dig.append(img_to_array(img)/255.0)
                Y_dig.append(digit_encoder.transform([plate_text[i]])[0])
            i+=1
    return np.array(X_let), np.array(Y_let), np.array(X_dig), np.array(Y_dig)