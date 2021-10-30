# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 16:32:10 2021

@author: eugsa
"""
from keras.models import load_model
import tensorflow as tf
import numpy as np
import os
import PIL
import PIL.Image
import keras
from pathlib import Path
import tensorflow_datasets as tfds
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2, ResNet50, InceptionV3 # try to use them and see which is better
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.utils import get_file
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import pathlib
from tensorflow.keras import regularizers 
from tensorflow.keras import backend as K

best_model = load_model("freiburg_groceries_dataset/best_models//InceptionV3_last5-loss-0.78.h5")
best_model.summary()
print(config)

best_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
best_model.optimizer.get_config()
best_model = tf.keras.models.load_model('freiburg_groceries_dataset/best_models/InceptionV3_last5-loss-0.64.h5')
best_model.summary()
best_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(K.eval(best_model.optimizer.lr))

best_model.optimizer.get_config()

best_model.layers[0].input_shape