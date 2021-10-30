# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 16:55:19 2021

@author: eugsa
"""
import tensorflow as tf
import numpy as np
import os
import PIL
import PIL.Image
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from pathlib import Path
import tensorflow_datasets as tfds
from tensorflow.keras import datasets, layers, models
import shutil

#Let's load these images off disk using tf.keras.preprocessing.image_dataset_from_directory.    
image_path = Path("C:/Users/eugsa/Tensorflow-GPU/freiburg_groceries_dataset/images")     

#Define some parameters for the loader:
EPOCHS=200
batch_size = 32
img_height = 224
img_width = 224

img_shape = (224, 224, 3)

#80% of the images for training and 20% for validation
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  image_path,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  image_path,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


# class names are in the class_names attribute on these datasets
class_names = train_ds.class_names
print(class_names)

#Visualise the data

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")


#Configure the dataset for performance
#.cache() keeps the images in memory after they're loaded off disk during the first epoch. 
#This will ensure the dataset does not become a bottleneck while training your model. 
#If your dataset is too large to fit into memory, you can also use this method to create a performant on-disk cache.

#.prefetch() overlaps data preprocessing and model execution while training.
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size = AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size = AUTOTUNE)


#Train a model without fine tuning

num_classes = 25
#we use keras Sequential API for this model
model = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
  tf.keras.layers.Conv2D(32, 3, 3, padding = 'same', activation='relu', input_shape=img_shape),
  tf.keras.layers.MaxPooling2D(pool_size=2),
  tf.keras.layers.Dropout(0.3),
  tf.keras.layers.Conv2D(32, 3, 3, padding = 'same', activation='relu'),
  tf.keras.layers.MaxPooling2D(pool_size  = 2),
  tf.keras.layers.Conv2D(32, 3, padding = 'same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Dropout(0.3),
  tf.keras.layers.Flatten(),#flatten multidimentional tensor into a single-dimentional tensor
  tf.keras.layers.Dense(128, activation = 'softmax'),
  tf.keras.layers.Dense(num_classes)
])



'''
before training a model we need to configure it using model.compile using three parameters:
  loss, optimizer, metrics  
    '''
model.compile(
  optimizer='adam',
  loss="sparse_categorical_crossentropy",
  metrics=['accuracy'])



tf.keras.backend.clear_session()


initial_weights = model.get_weights()
model.set_weights(initial_weights)

# define path to save the mnodel
path_model=os.path.join('model.weights.best.hdf5')
shutil.rmtree(path_model, ignore_errors=True)
checkpointer = ModelCheckpoint(filepath = path_model, verbose = 1, save_best_only = True)


history=model.fit(train_ds,
  validation_data=val_ds,
  batch_size=batch_size,
  epochs=EPOCHS,
  callbacks=[checkpointer])




model.fit(
  train_ds,
  validation_data=val_ds,
  batch_size = batch_size,
  epochs=EPOCHS
)


#inspect the model
model.summary()

#fine tuning the model

image_count = len(list(image_path.glob('*/*.png')))
print(image_count)

list_ds = tf.data.Dataset.list_files(str(image_path/'*/*'), shuffle=False)

list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)

for f in list_ds.take(25):
    print(f.numpy())
    
    
#The tree structure of the files can be used to compile a class_names list.
class_names = np.array(sorted([item.name for item in image_path.glob('*') if item.name != "LICENSE.txt"]))
print(class_names)

#split into train and validation datasets

val_size = int(image_count*0.2)
train_ds = list_ds.skip(val_size)
val_ds= list_ds.take(val_size)

#check the length of each dataset

print(tf.data.experimental.cardinality(train_ds).numpy())
print(tf.data.experimental.cardinality(val_ds).numpy())

#short function that converts a file path to an (img, label) pair:
    
def get_label(file_path):
    # convert the path to a list of path components
     parts = tf.strings.split(file_path, os.path.sep)
     # The second to last is the class-directory
     one_hot = parts[-2] == class_names
     # Integer encode the label
     return tf.argmax(one_hot)


def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.io.decode_jpeg(img, channels=3)
  # resize the image to the desired size
  return tf.image.resize(img, [img_height, img_width])


def process_path(file_path):
  label = get_label(file_path)
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label

#Use Dataset.map to create a dataset of image, label pairs:

# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.

train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)

val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)


for image, label in train_ds.take(1):
    print("Image shape:", image.numpy().shape)
    print("Lable:", label.numpy())
    


#Configure dataset for performance
#To train a model with this dataset you will want the data:

#To be well shuffled.
#To be batched.
#Batches to be available as soon as possible.
#These features can be added using the tf.data API


def configure_for_performance(ds):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size = 1000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size = AUTOTUNE)
    return ds

train_ds = configure_for_performance(train_ds)
val_ds = configure_for_performance(val_ds)

#visualize the data

image_batch, label_batch = next(iter(train_ds))
plt.figure(figsize=(10, 10))
for i in range(9):
  ax = plt.subplot(3, 3, i + 1)
  plt.imshow(image_batch[i].numpy().astype("uint8"))
  label = label_batch[i]
  plt.title(class_names[label])
  plt.axis("off")


#train a model with a few epochs to keep the training time short

model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=5
)

import time

def benchmark(dataset, num_epochs=2):
    start_time = time.perf_counter()
    for epoch_num in range(num_epochs):
        for sample in dataset:
            # Performing a training step
            time.sleep(0.01)
    print("Execution time:", time.perf_counter() - start_time)

#model overfits


benchmark(train_ds)


#model performance visualisation

history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=5
)

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.0, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(val_ds, verbose=2)












