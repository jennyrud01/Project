# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 15:56:00 2021

@author: eugsa
"""

import tensorflow as tf
from functools import partial
import matplotlib.pyplot as plt


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
from PIL import Image, ImageFont, ImageDraw

AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 64
IMAGE_SIZE = [256, 256]

PATH = 'C:/Users/eugsa/Tensorflow-GPU/Object_Detection_Imgs1-TFRecords-export'


FILENAMES = tf.io.gfile.glob(PATH + "/*.tfrecord")
split_ind = int(0.9 * len(FILENAMES))
TRAINING_FILENAMES, VALID_FILENAMES = FILENAMES[:split_ind], FILENAMES[split_ind:]

print("Train TFRecord Files:", len(TRAINING_FILENAMES))
print("Validation TFRecord Files:", len(VALID_FILENAMES))

import shutil
source_dir = 'C:/Users/eugsa/Tensorflow-GPU/Object_Detection_Imgs1-TFRecords-export'
target_dir = 'C:/Users/eugsa/Tensorflow-GPU/Object_Detection_Imgs1-TFRecords-export/train'

for file_name in TRAINING_FILENAMES:
    shutil.move(os.path.join(source_dir, file_name), target_dir)

raw_dataset = tf.data.TFRecordDataset(f'C:/Users/eugsa/Tensorflow-GPU/Object_Detection_Imgs1-TFRecords-export/BEANS0000.tfrecord')
for raw_record in raw_dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    print(example)





fontname = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
font = ImageFont.truetype(fontname, 40) if os.path.isfile(fontname) else ImageFont.load_default()


def bbox(img, xmin, ymin, xmax, ymax, color, width, label, score):
    draw = ImageDraw.Draw(img)
    xres, yres = img.size[0], img.size[1]
    box = np.multiply([xmin, ymin, xmax, ymax], [xres, yres, xres, yres]).astype(int).tolist()
    txt = " {}: {}%" if score >= 0. else " {}"
    txt = txt.format(label, round(score, 1))
    ts = draw.textsize(txt, font=font)
    draw.rectangle(box, outline=color, width=width)
    if len(label) > 0:
        if box[1] >= ts[1]+3:
            xsmin, ysmin = box[0], box[1]-ts[1]-3
            xsmax, ysmax = box[0]+ts[0]+2, box[1]
        else:
            xsmin, ysmin = box[0], box[3]
            xsmax, ysmax = box[0]+ts[0]+2, box[3]+ts[1]+1
        draw.rectangle([xsmin, ysmin, xsmax, ysmax], fill=color)
        draw.text((xsmin, ysmin), txt, font=font, fill='white')


labels = "beans"


def plot_img(img, axes, xmin, ymin, xmax, ymax, classes, class_label):
    for i in range(len(xmin)):
       color="green"
       bbox(img, xmin[i], ymin[i], xmax[i], ymax[i], color, 5, classes[i].decode(), -1)
    plt.setp(axes, xticks=[], yticks=[])
    axes.set_title(class_label)
    plt.imshow(img)
    
from io import BytesIO

dataset = tf.data.TFRecordDataset(FILENAMES)
img_example = next(iter(dataset)) 
img_parsed = tf.train.Example.FromString(img_example.numpy())
# only extract features we will actually use
xmin=img_parsed.features.feature['image/object/bbox/xmin'].float_list.value[:]
xmax=img_parsed.features.feature['image/object/bbox/xmax'].float_list.value[:]
ymin=img_parsed.features.feature['image/object/bbox/ymin'].float_list.value[:]
ymax=img_parsed.features.feature['image/object/bbox/ymax'].float_list.value[:]
#by=img_parsed.features.feature['image/by'].bytes_list.value[0].decode()
classes=img_parsed.features.feature['image/object/class/text'].bytes_list.value[:]
class_label=img_parsed.features.feature['image/object/class/label'].int64_list.value[:]
img_encoded=img_parsed.features.feature['image/encoded'].bytes_list.value[0]

fig = plt.figure(figsize=(10,10))
axes = axes = fig.add_subplot(1, 1, 1)
img = Image.open(BytesIO(img_encoded))
plot_img(img, axes, xmin, ymin, xmax, ymax, classes, class_label)



def read_tfrecord(example, labeled):
    tfrecord_format = (
        {
            "image": tf.io.FixedLenFeature([], tf.string),
            "target": tf.io.FixedLenFeature([], tf.int64),
        }
        if labeled
        else {"image": tf.io.FixedLenFeature([], tf.string),}
    )
    example = tf.io.parse_single_example(example, tfrecord_format)
    image = decode_image(example["image"])
    if labeled:
        label = tf.cast(example["target"], tf.int32)
        return image, label
    return image

def load_dataset(filenames, labeled=True):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False  # disable order, increase speed
    dataset = tf.data.TFRecordDataset(
        filenames
    )  # automatically interleaves reads from multiple files
    dataset = dataset.with_options(
        ignore_order
    )  # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(partial(read_tfrecord, labeled=labeled), num_parallel_calls=AUTOTUNE
    )
    # returns a dataset of (image, label) pairs if labeled=True or just images if labeled=False
    return dataset


def get_dataset(filenames, labeled=True):
    dataset = load_dataset(filenames, labeled=labeled)
    dataset = dataset.shuffle(2048)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    return dataset


train_dataset = get_dataset(TRAINING_FILENAMES)
valid_dataset = get_dataset(VALID_FILENAMES)


initial_learning_rate = 0.01
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=20, decay_rate=0.96, staircase=True
)

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    "object_model.h5", save_best_only=True
)

early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    patience=10, restore_best_weights=True
)


def make_model():
    base_model = tf.keras.applications.Xception(
        input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet"
    )

    base_model.trainable = False

    inputs = tf.keras.layers.Input([*IMAGE_SIZE, 3])
    x = tf.keras.applications.xception.preprocess_input(inputs)
    x = base_model(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(8, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.7)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss="binary_crossentropy",
        metrics=tf.keras.metrics(name="accuracy"),
    )

    return model

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print("Device:", tpu.master())
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
except:
    strategy = tf.distribute.get_strategy()
print("Number of replicas:", strategy.num_replicas_in_sync)
with strategy.scope():
    model = make_model()

history = model.fit(
    train_dataset,
    epochs=2,
    validation_data=valid_dataset,
    callbacks=[checkpoint_cb, early_stopping_cb],
)





gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
        
tf.keras.backend.clear_session()

policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
tf.keras.mixed_precision.experimental.set_policy(policy)

import numpy as np
tfrds = tf.data.TFRecordDataset(TRAINING_FILENAMES)

help(tfrds)

for elem in tfrds:
    print( type(elem))
    break
    
for npelem in tfrds.as_numpy_iterator():
    print( type(npelem))
    barr = np.frombuffer(npelem, dtype=np.byte )
    print( barr.shape)
    break
    
oneelem = tfrds.take(1)
print( type(oneelem))



class DataRead():
    def __init__(self):
        self.feature_description = {         
            'image_raw': tf.io.VarLenFeature( dtype=tf.float32),
            'img_shape': tf.io.FixedLenFeature([], tf.string),
            'poslabel': tf.io.VarLenFeature( dtype=tf.float32),
            'collabel': tf.io.VarLenFeature( dtype=tf.float32)            
        }  

    def prepdata( self, fmap):
        pmap = tf.io.parse_single_example(fmap, self.feature_description)

        imgraw = tf.sparse.to_dense(pmap['image_raw'])
        imshape =  tf.io.decode_raw(pmap['img_shape'], tf.uint8)
        poslabel = tf.sparse.to_dense(pmap['poslabel'])
        collabel = tf.one_hot( tf.cast( tf.sparse.to_dense(pmap['collabel']), tf.uint8),  tf.constant(3))[0]
                                    
        return (tf.reshape( imgraw, tf.cast(imshape, tf.int32)),
                tf.concat( [poslabel,collabel], axis=-1))

AUTOTUNE = tf.data.AUTOTUNE
BATCHSIZE = 64
IMAGE_SIZE = [256, 256]

datar = DataRead()

traindat = tfrds.map(datar.prepdata, num_parallel_calls=tf.data.experimental.AUTOTUNE)
traindat = traindat.cache()
traindat = traindat.shuffle(1000, seed=1234, reshuffle_each_iteration=True)
traindat = traindat.batch(BATCHSIZE, drop_remainder=True)
traindat = traindat.prefetch( tf.data.experimental.AUTOTUNE)


def genmodel():

    minput = layers.Input( shape=(112,112,3,), dtype="float32")
    bmod = minput
    bmod = layers.Conv2D(filters=16, kernel_size= (3, 3), padding = 'same')(bmod)
    bmod = layers.ReLU()(bmod)
    bmod = layers.MaxPooling2D(pool_size=(3, 3), strides=(3,3), padding = 'same')(bmod) 
    bmod = layers.Conv2D(filters=32, kernel_size= (3, 3), padding = 'same')(bmod)
    bmod = layers.ReLU()(bmod)
    bmod = layers.MaxPooling2D(pool_size=(3, 3), strides=(3,3), padding = 'same')(bmod) 
    bmod = layers.Flatten()(bmod)
    bmod = layers.Dropout(0.1)(bmod)    
    bmod = layers.Dense(128, dtype="float32")(bmod)
    bmod = layers.ReLU()(bmod)
    bmod = layers.Dense(64, dtype="float32")(bmod)
    bmod = layers.ReLU()(bmod)
    bmod = layers.BatchNormalization()(bmod)
    mout = layers.Dense(5, activation='sigmoid', dtype="float32")(bmod)

    sdetect = tf.keras.Model( inputs=minput, outputs=mout)

    sdetect.build(input_shape=(112,112,3,))
    sdetect.summary()
    
    return sdetect

sdetect = genmodel()

    
sdetect.compile(
    optimizer = tf.keras.optimizers.SGD(learning_rate=3e-3, momentum=0.9, nesterov=True) ,
    loss = tf.keras.losses.MeanSquaredError()
) 

history = sdetect.fit(
    traindat,
    epochs=10
)