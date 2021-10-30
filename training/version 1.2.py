# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 11:40:00 2021

@author: eugsa
"""


# -*- coding: utf-8 -*-
"""
Created on Sun Aug 15 19:39:45 2021

@author: eugsa
"""


# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 13:29:43 2021

@author: eugsa
"""
'''
What has been performed:
    1) Data augmentation using ImageDataGenerator, 
    2) Baseline model changed with all layers frozen and a classifier head added on top
    
'''

import tensorflow as tf
import numpy as np
import os
import PIL
import PIL.Image

from pathlib import Path
import tensorflow_datasets as tfds
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.utils import get_file
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
import os
import pathlib


'''
The dataset comes with inconsistent image sizes, as a result, 
we gonna need to resize all the images to a shape 
that is acceptable by MobileNet (the model that we gonna use):
    '''

batch_size = 32
# 5 types of flowers
num_classes = 25
# training for 10 epochs
NUM_EPOCHS = 10
# size of each image
IMAGE_SHAPE = (224, 224, 3)


'''
load dataset
'''

def load_data():
    """This function downloads, extracts, loads, normalizes and one-hot encodes Flower Photos dataset"""
    # download the dataset and extract it
    
    data_dir = pathlib.Path("C:/Users/eugsa/Tensorflow-GPU/freiburg_groceries_dataset/images")
    print(data_dir)
    # count how many images are there
    image_count = len(list(data_dir.glob('*/*.png')))
    print("Number of images:", image_count)
    
    CLASS_NAMES = np.array(sorted([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"]))
      # 20% validation set 80% training set
    image_generator = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    # make the training dataset generator
    train_data_gen = image_generator.flow_from_directory(directory=str(data_dir), batch_size=batch_size,
                                                        classes=list(CLASS_NAMES), target_size=(IMAGE_SHAPE[0], IMAGE_SHAPE[1]),
                                                        shuffle=True, subset="training")
    # make the validation dataset generator
    test_data_gen = image_generator.flow_from_directory(directory=str(data_dir), batch_size=batch_size, 
                                                        classes=list(CLASS_NAMES), target_size=(IMAGE_SHAPE[0], IMAGE_SHAPE[1]),
                                                        shuffle=True, subset="validation")
    return train_data_gen, test_data_gen, CLASS_NAMES





def create_model(input_shape):
    # load MobileNetV2
    base_model = MobileNetV2(input_shape=input_shape, weights = 'imagenet', include_top = False)
    
    #freeze all the layers of the base model
    
    base_model.trainable = False
  
    model = Sequential([
        base_model,
        Conv2D(32,3, activation = 'relu'),
        GlobalAveragePooling2D(),
        Dense(num_classes, activation = 'softmax')]
        )

    # print the summary of the model architecture
    model.summary()
    # training the model using adam optimizer
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model



if __name__ == "__main__":
    # load the data generators
    train_generator, validation_generator, class_names = load_data()
    # constructs the model
    model = create_model(input_shape=IMAGE_SHAPE)
    # model name
    model_name = "MobileNetV2_baseline"
    # some nice callbacks
    tensorboard = TensorBoard(log_dir=os.path.join("logs", model_name))
    checkpoint = ModelCheckpoint(os.path.join("results", f"{model_name}" + "-loss-{val_loss:.2f}.h5"),
                                save_best_only=True,
                                verbose=1)
    # make sure results folder exist
    if not os.path.isdir("results"):
        os.mkdir("results")
    # count number of steps per epoch
    training_steps_per_epoch = np.ceil(train_generator.samples / batch_size)
    validation_steps_per_epoch = np.ceil(validation_generator.samples / batch_size)
    # train using the generators
    history = model.fit_generator(train_generator, steps_per_epoch=training_steps_per_epoch,
                        validation_data=validation_generator, validation_steps=validation_steps_per_epoch,
                        epochs=NUM_EPOCHS, verbose=1, callbacks=[tensorboard, checkpoint])
    
    
# load the data generators
train_generator, validation_generator, class_names = load_data()
# constructs the model
model = create_model(input_shape=IMAGE_SHAPE)
# load the optimal weights
model.load_weights("results/MobileNetV2_baseline-loss-1.96.h5")
validation_steps_per_epoch = np.ceil(validation_generator.samples / batch_size)
# print the validation loss & accuracy
evaluation = model.evaluate_generator(validation_generator, steps=validation_steps_per_epoch, verbose=1)
print("Val loss:", evaluation[0])
print("Val Accuracy:", evaluation[1])


#Display training curve

def display_training_curves(history,title):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(NUM_EPOCHS)
    
    plt.plot(epochs_range, acc, label = 'Train Accuracy ')
    plt.plot(epochs_range, val_acc, label = 'Validation Accuracy')
    plt.title(title)
    plt.legend(loc='upper left')
    plt.figure()
    
    plt.show()

display_training_curves(history, 'Baseline MobileNetV2 Model Training')

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




def create_model_fine_tuned(input_shape):
    # load MobileNetV2
    base_model = MobileNetV2(input_shape=input_shape, weights = 'imagenet', include_top = False)
    
    #freeze all the layers of the base model
    
    base_model.trainable = True
    print("Number of layers in the base model: ", len(base_model.layers))
    
    fine_tune_at = 100
    
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
       
    # training the model using adam optimizer
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    # print the summary of the model architecture
    model.summary()
    return model



INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS =10
TOTAL_EPOCHS = INITIAL_EPOCHS + FINE_TUNE_EPOCHS #20

if __name__ == "__main__":
    # load the data generators
    train_generator, validation_generator, class_names = load_data()
    # constructs the model
    model = create_model_fine_tuned(input_shape=IMAGE_SHAPE)
    
    # count number of steps per epoch
    training_steps_per_epoch = np.ceil(train_generator.samples / batch_size)
    validation_steps_per_epoch = np.ceil(validation_generator.samples / batch_size)
    # train using the generators
    history_fine = model.fit_generator(train_generator, epochs=TOTAL_EPOCHS, initial_epoch = INITIAL_EPOCHS,
                                       steps_per_epoch=training_steps_per_epoch,
                        validation_data=validation_generator, validation_steps=validation_steps_per_epoch
                        )

acc = history.history['accuracy'] + history_fine.history['accuracy']
val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']

plt.figure(figsize = (8, 4))

plt.plot(acc, label = 'Train Accuracy')
plt.plot(val_acc, label = 'Validation Accuracy')
plt.ylim([0.5, 1])
plt.plot([NUM_EPOCHS-1, NUM_EPOCHS-1], plt.ylim(ymin = 0.2), label = 'Start Fine Tuning')
plt.title('Fine-tune a Pretrained Model')
plt.legend(loc = 'upper left')

plt.show()


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




















