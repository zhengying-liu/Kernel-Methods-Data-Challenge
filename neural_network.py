#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 21:58:52 2017

@author: Evariste
"""

if __name__ == "__main__":
    
    from utils import reshape_images
    import keras
    import sklearn
    import pandas as pd
    import numpy as np
    import tensorflow as tf
    
    data = pd.read_csv('data/Xtr.csv', header=None).as_matrix()[:,:-1]
    test = pd.read_csv('data/Xte.csv', header=None).as_matrix()[:,:-1]
    target = pd.read_csv('data/Ytr.csv')["Prediction"].values
    
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
            data, target, test_size=0.15, random_state=37)
    
    X_train_img = reshape_images(X_train)
    X_test_img = reshape_images(X_test)
    
    Y_train = keras.utils.np_utils.to_categorical(y_train)
    
    N = X_train.shape[-1]
    n_classes = 10
    
    model_input = keras.layers.Input(shape=(32,32,3))
    #################### Architecture of the NN ####################
    x = keras.layers.Convolution2D(6, 3, 3, activation='relu',
                                   border_mode='same')(model_input)
    x = keras.layers.MaxPooling2D(pool_size=(2,2))(x)
    x = keras.layers.Convolution2D(12, 3, 3, activation='relu',
                                   border_mode='same')(x)
    x = keras.layers.MaxPooling2D(pool_size=(2,2))(x)
    x = keras.layers.Convolution2D(24, 3, 3, activation='relu',
                                   border_mode='same')(x)
    x = keras.layers.MaxPooling2D(pool_size=(2,2))(x)
    x = keras.layers.Flatten()(x)
    head_classes = keras.layers.Dense(n_classes, activation="softmax", 
                                      name="head_classes")(x)
    model = keras.models.Model(model_input, output=[head_classes])
    ################################################################
    
    model.compile(optimizer="adam",
                  loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    
    model.fit(X_train_img, Y_train, nb_epoch=10, batch_size=50)
    
    y_pred = model.predict(X_test_img).argmax(axis=1)
    print("\naccuracy:", sklearn.metrics.accuracy_score(y_test, y_pred))
    
    