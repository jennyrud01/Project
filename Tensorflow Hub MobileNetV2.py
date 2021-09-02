# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 21:04:53 2021

@author: eugsa
"""


import tensorflow as tf
import tensorflow_hub as hub

import numpy as np
import os
import PIL
import PIL.Image
import keras
from pathlib import Path
import tensorflow_datasets as tfds
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, GlobalAveragePooling2D
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


batch_size = 64
# 5 types of flowers
num_classes = 25
# training for 10 epochs
epochs = 5
# size of each image
IMAGE_SHAPE = (224, 224, 3)


'''
load dataset
'''

def load_data():
    """This function downloads, extracts, loads, normalizes and one-hot encodes Flower Photos dataset"""
    # download the dataset and extract it
    
    data_dir = pathlib.Path("C:/Users/eugsa/Tensorflow-GPU/freiburg_groceries_dataset/images")
    # count how many images are there
    image_count = len(list(data_dir.glob('*/*.png')))
    print("Number of images:", image_count)
    
    CLASS_NAMES = np.array(sorted([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"]))
      # 20% validation set 80% training set
    image_generator = ImageDataGenerator(rescale=1/255, 
                                         validation_split=0.1)
    # make the training dataset generator
    train_data_gen = image_generator.flow_from_directory(directory=str(data_dir), batch_size=batch_size,
                                                        classes=list(CLASS_NAMES), target_size=(IMAGE_SHAPE[0], IMAGE_SHAPE[1]),
                                                        shuffle=True, subset="training")
    # make the validation dataset generator
    test_data_gen = image_generator.flow_from_directory(directory=str(data_dir), batch_size=batch_size, 
                                                        classes=list(CLASS_NAMES), target_size=(IMAGE_SHAPE[0], IMAGE_SHAPE[1]),
                                                        shuffle=True, subset="validation")
    return train_data_gen, test_data_gen, CLASS_NAMES


feature_extractor_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4"

feature_extractor_layer = hub.KerasLayer(feature_extractor_url, input_shape = IMAGE_SHAPE, trainable = False)

model = tf.keras.Sequential([feature_extractor_layer, tf.keras.layers.Dropout(0.5),
                             tf.keras.layers.Dense(num_classes, activation = 'softmax')])
#hub_layer = hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4",
               #trainable=True, arguments=dict(batch_norm_momentum=0.997))
model.summary()

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
train_generator, validation_generator, class_names = load_data()
training_steps_per_epoch = np.ceil(train_generator.samples / batch_size)
validation_steps_per_epoch = np.ceil(validation_generator.samples / batch_size)

history = model.fit(train_generator, steps_per_epoch=training_steps_per_epoch,
                        validation_data=validation_generator, validation_steps=validation_steps_per_epoch,
                        epochs=epochs)

def display_training_curves(history,title):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(epochs)
    
    plt.plot(epochs_range, acc, label = 'Train Accuracy ')
    plt.plot(epochs_range, val_acc, label = 'Validation Accuracy')
    plt.title(title)
    plt.legend(loc='upper left')
    plt.figure()
    
    plt.show()

display_training_curves(history, 'MobileNetV2 Hub Model Training')


score = model.evaluate(validation_generator, verbose=0)
print('\n', 'Test accuracy:', score[1])

# get a random batch of images
image_batch, label_batch = next(iter(validation_generator))
# turn the original labels into human-readable text
label_batch = [class_names[np.argmax(label_batch[i])] for i in range(batch_size)]
# predict the images on the model
predicted_class_names = model.predict(image_batch)
predicted_ids = [np.argmax(predicted_class_names[i]) for i in range(batch_size)]
# turn the predicted vectors to human readable labels
predicted_class_names = np.array([class_names[id] for id in predicted_ids])
# some nice plotting
plt.figure(figsize=(10,9))
for n in range(30):
    plt.subplot(6,5,n+1)
    plt.subplots_adjust(hspace = 0.3)
    plt.imshow(image_batch[n])
    if predicted_class_names[n] == label_batch[n]:
        color = "blue"
        title = predicted_class_names[n].title()
    else:
        color = "red"
        title = f"{predicted_class_names[n].title()}"
    plt.title(title, color=color)
    plt.axis('off')
_ = plt.suptitle("Model predictions (blue: correct, red: incorrect)")
plt.show()



















