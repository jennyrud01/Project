# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 16:27:15 2021

@author: eugsa
"""


# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 13:29:43 2021

@author: eugsa
"""


import tensorflow as tf
import numpy as np
import os
import PIL
import PIL.Image

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
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
session = tf.compat.v1.InteractiveSession(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))


gpus = tf.config.list_physical_devices('GPU')

'''
The dataset comes with inconsistent image sizes, as a result, 
we gonna need to resize all the images to a shape 
that is acceptable by MobileNet (the model that we gonna use):
    '''

batch_size = 64
# 5 types of flowers
num_classes = 25
# training for 10 epochs
epochs = 20
# size of each image
IMAGE_SHAPE = (299, 299, 3)


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




def create_model(input_shape):
    # load InceptionV3
    K.set_learning_phase(0)
    base_inception = InceptionV3(weights='imagenet', input_shape=IMAGE_SHAPE, include_top = False)
    nn_inputs = base_inception.input # save input for layer use
   
    for layer in base_inception.layers:
        layer.trainable = False
        
    K.set_learning_phase(1)# switch to training mode
    # build the top layers
    myModelOut=base_inception.output

    myModelOut = GlobalAveragePooling2D()(myModelOut) 
    myModelOut = Dense(1024, activation="relu")(myModelOut) 
    myModelOut = layers.Dropout(0.5)(myModelOut)
    #x = Dense(512, kernel_regularizer = regularizers.l2(l = 0.016),activity_regularizer=regularizers.l1(0.006),
              #bias_regularizer=regularizers.l1(0.006),activation='relu', kernel_initializer= tf.keras.initializers.GlorotUniform(seed=123))(x) 
    myModelOut = Dense(num_classes, activation="softmax")(myModelOut)
        
    model = Model(inputs=nn_inputs, outputs=myModelOut)
    model.summary()
    # training the model using adam optimizer
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()
    
    earlystop_callback = tf.keras.callbacks.EarlyStopping(
                                   monitor='val_accuracy', 
                                   min_delta=0.0001, 
                                   patience=5)
    return model

#model = create_model(input_shape=IMAGE_SHAPE)

if __name__ == "__main__":
    # load the data generators
    train_generator, validation_generator, class_names = load_data()
    # constructs the model
    model = create_model(input_shape=IMAGE_SHAPE)
    # model name
    model_name = "InceptionV3_last5"
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
    
    earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
					 min_delta=0.0001, patience=5)
    # train using the generators
    history = model.fit_generator(train_generator, steps_per_epoch=training_steps_per_epoch,
                        validation_data=validation_generator, validation_steps=validation_steps_per_epoch,
                        epochs=epochs, verbose=1, callbacks=[tensorboard, checkpoint, earlystop_callback])
    
#Display training curve

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

display_training_curves(history, 'Baseline InceptionV3 Model Training')
  
# load the data generators
train_generator, validation_generator, class_names = load_data()
# constructs the model
model = create_model(input_shape=IMAGE_SHAPE)
# load the optimal weights
model.load_weights("results/InceptionV3_last5-loss-1.56.h5")
validation_steps_per_epoch = np.ceil(validation_generator.samples / batch_size)
# print the validation loss & accuracy
evaluation = model.evaluate_generator(validation_generator, steps=validation_steps_per_epoch, verbose=1)
print("Val loss:", evaluation[0])
print("Val Accuracy:", evaluation[1])


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

session.close()
