# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 11:23:51 2021

@author: eugsa
"""

import tensorflow as tf
import numpy as np
import os
import PIL
import PIL.Image

from pathlib import Path
import tensorflow_datasets as tfds

x = tf.random.uniform([3, 3])
print("Is there a GPU available: "),
print(tf.config.list_physical_devices("GPU"))

print("Is the Tensor on GPU #0:  "),
print(x.device.endswith('GPU:0'))