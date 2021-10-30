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
from keras.models import load_model
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
session = tf.compat.v1.InteractiveSession(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))


gpus = tf.config.list_physical_devices('GPU')

'''
The dataset comes with inconsistent image sizes, as a result, 
we gonna need to resize all the images to a shape 
that is acceptable by MobileNet (the model that we gonna use):
    '''

batch_size = 64

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
    image_generator = ImageDataGenerator(rescale=1./255, 
                                         
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
    
    # TODO:uncomment to see what happens
   
# add a global spatial average pooling layer
    myModelOut = GlobalAveragePooling2D()(myModelOut) 
      
    #myModelOut = tf.keras.layers.Dropout(0.5, seed = 1000)(myModelOut)
 # let's add a fully-connected layer   
    myModelOut = Dense(1024, activation="relu")(myModelOut)
    #myModelOut = Dense(512, activation="relu")(myModelOut)
    myModelOut = Dense(512, activation="relu")(myModelOut)
    myModelOut = layers.Dropout(0.5)(myModelOut) 
    myModelOut = Dense(num_classes, activation="softmax", name = 'predictions')(myModelOut)
  # this is the model we will train      
    model = Model(inputs=nn_inputs, outputs=myModelOut)
    model.summary()
    # training the model using adam optimizer and learning rate
    #opt = tf.keras.optimizers.Adam(learning_rate=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    opt = tf.keras.optimizers.Adam(learning_rate = 0.0001, decay = 1e-3)
  # If you need to load weights from previous training, do so here:
    model_path = 'C:/Users/eugsa/Tensorflow-GPU/freiburg_groceries_dataset/best_models/InceptionV3_last5-loss-0.64.h5'
    model.load_weights(model_path, by_name=True)       
   # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])          
    model.summary()
    return model



#model = create_model(input_shape=IMAGE_SHAPE)
import shutil

if __name__ == "__main__":
    # load the data generators
    train_generator, validation_generator, class_names = load_data()
    # constructs the model
    model = create_model(input_shape=IMAGE_SHAPE)
    # model name
    model_name = "InceptionV3_last5"
    # some nice callbacks
    shutil.rmtree("logs", ignore_errors=True)
    tensorboard = TensorBoard(log_dir=os.path.join("logs", model_name))
    checkpoint = ModelCheckpoint(os.path.join("results", f"{model_name}" + "-loss-{val_loss:.2f}.h5"),
                                save_best_only=True,
                                verbose=1)
    # make sure results folder exist
    
    shutil.rmtree("results", ignore_errors=True)
    
    #if not os.path.isdir("results"):
    
    os.mkdir("results")
    # count number of steps per epoch
    training_steps_per_epoch = np.ceil(train_generator.samples / batch_size)
    validation_steps_per_epoch = np.ceil(validation_generator.samples / batch_size)
    
    earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
					 min_delta=0.0001, patience=5)
        
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

display_training_curves(history, 'Fine Tuned InceptionV3 Model Training')
  
# load the data generators
train_generator, validation_generator, class_names = load_data()
# constructs the model
model = create_model(input_shape=IMAGE_SHAPE)
# load the optimal weights
model.load_weights("freiburg_groceries_dataset/best_models//InceptionV3_last5-loss-0.64.h5")
validation_steps_per_epoch = np.ceil(validation_generator.samples / batch_size)
# print the validation loss & accuracy
evaluation = model.evaluate(validation_generator, steps=validation_steps_per_epoch, verbose=1)
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

base_inception = InceptionV3(weights='imagenet', input_shape=IMAGE_SHAPE, include_top = False)
    
for layer in base_inception.layers[:249]:
    layer.trainable = False
for layer in base_inception.layers[249:]:
    layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
opt = tf.keras.optimizers.Adam(learning_rate = 0.0001, decay = 1e-3)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])


history_fine_tuned = model.fit_generator(train_generator, steps_per_epoch=training_steps_per_epoch,
                        validation_data=validation_generator, validation_steps=validation_steps_per_epoch,
                        epochs=epochs, verbose=1, callbacks=[tensorboard, checkpoint, earlystop_callback])


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

display_training_curves(history_fine_tuned, 'Fine Tuned InceptionV3 Model Training')


def display_loss_curves(history,title):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(epochs) 
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


display_loss_curves(history, 'Fine Tuned InceptionV3 Model Training')



saved_keras_model = 'InceptionV3_model'
model.save(saved_keras_model)

keras_model = tf.keras.models.load_model('InceptionV3_model')
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('mobilenet_v2_1.0_224.tflite', 'wb') as f:
  f.write(tflite_model)


# A generator that provides a representative dataset
  
data_dir = pathlib.Path("C:/Users/eugsa/Tensorflow-GPU/freiburg_groceries_dataset/images")
    # count how many images are there
image_count = len(list(data_dir.glob('*/*.png')))

IMAGE_SIZE= 299
def representative_data_gen():
  dataset_list = tf.data.Dataset.list_files(str(data_dir) + '/*/*')
  for i in range(100):
    image = next(iter(dataset_list))
    image = tf.io.read_file(image)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image = tf.cast(image / 255., tf.float32)
    image = tf.expand_dims(image, 0)
    yield [image]

saved_keras_model = 'InceptionV3_model'
model.save(saved_keras_model)

keras_model = tf.keras.models.load_model('InceptionV3_model')
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# This ensures that if any ops can't be quantized, the converter throws an error
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# These set the input and output tensors to uint8
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
# And this sets the representative dataset so we can quantize the activations
converter.representative_dataset = representative_data_gen
tflite_model = converter.convert()

interpreter = tf.lite.Interpreter(model_content=tflite_model)
input_type = interpreter.get_input_details()[0]['dtype']
print('input: ', input_type)
output_type = interpreter.get_output_details()[0]['dtype']
print('output: ', output_type)



with open('Inception3model_quant.tflite', 'wb') as f:
  f.write(tflite_model)


batch_images, batch_labels = next(validation_generator)

logits = model(batch_images)
prediction = np.argmax(logits, axis=1)
truth = np.argmax(batch_labels, axis=1)


keras_accuracy = tf.keras.metrics.Accuracy()
keras_accuracy(prediction, truth)

print("Raw model accuracy: {:.2%}".format(keras_accuracy.result()))


def set_input_tensor(interpreter, input):
  input_details = interpreter.get_input_details()[0]
  tensor_index = input_details['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  # Inputs for the TFLite model must be uint8, so we quantize our input data.
  # NOTE: This step is necessary only because we're receiving input data from
  # ImageDataGenerator, which rescaled all image data to float [0,1]. When using
  # bitmap inputs, they're already uint8 [0,255] so this can be replaced with:
  #   input_tensor[:, :] = input
  scale, zero_point = input_details['quantization']
  input_tensor[:, :] = np.uint8(input / scale + zero_point)

def classify_image(interpreter, input):
  set_input_tensor(interpreter, input)
  interpreter.invoke()
  output_details = interpreter.get_output_details()[0]
  output = interpreter.get_tensor(output_details['index'])
  # Outputs from the TFLite model are uint8, so we dequantize the results:
  scale, zero_point = output_details['quantization']
  output = scale * (output - zero_point)
  top_1 = np.argmax(output)
  return top_1

interpreter = tf.lite.Interpreter('Inception3model_quant.tflite')
interpreter.allocate_tensors()

# Collect all inference predictions in a list
batch_prediction = []
batch_truth = np.argmax(batch_labels, axis=1)

for i in range(len(batch_images)):
  prediction = classify_image(interpreter, batch_images[i])
  batch_prediction.append(prediction)

# Compare all predictions to the ground truth
tflite_accuracy = tf.keras.metrics.Accuracy()
tflite_accuracy(batch_prediction, batch_truth)
print("Quant TF Lite accuracy: {:.2%}".format(tflite_accuracy.result()))

tf.download('Inception3model_quant_edgetpu.tflite')
