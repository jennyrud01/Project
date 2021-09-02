# -*- coding: utf-8 -*-
"""
Created on Sat May 29 12:49:17 2021

@author: eugsa
"""

import pandas as pd
import re
import os
import cv2
import matplotlib.pyplot as plt

os.chdir('C:\\Users\\eugsa\\Tensorflow-GPU\\grocerydataset')
annotation = pd.read_table('annotation.txt', sep = '\n', header = None)
ann_df = pd.DataFrame()
idd = []
num_prod = []
coord = []



for i in range(len(annotation[0])):
    idd.append(annotation[0].str.split(' ')[i][0])
    num_prod.append(annotation[0].str.split(' ')[i][1])
    coord.append(annotation[0].str.split(' ')[i][2:])
    
ann_df['id']=idd
ann_df['num_prod'] = num_prod
ann_df['coord'] = coord
ann_df.head()

os.chdir('C:\\Users\\eugsa\\Tensorflow-GPU')
print(os.listdir('C://Users//eugsa//Tensorflow-GPU//ShelfImages'))
def test_train_split(data_part):
    ids = os.listdir('C://Users//eugsa//Tensorflow-GPU//ShelfImages//' + data_part)
    #print(ids)
    filename = []
    width = []
    height = []
    label = []
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    for idd in ids:
        h,w,c = cv2.imread('C://Users//eugsa//Tensorflow-GPU//ShelfImages//' + data_part+'//'+idd).shape
        bbox = ann_df[ann_df['id']==idd]['coord'].values[0]
        for j in range(0,len(bbox), 5):
            xminn = int(bbox[j])
            yminn = int(bbox[j+1])
            xmaxx = int(bbox[j+2])+xminn
            ymaxx = int(bbox[j+3])+yminn
            
            filename.append(idd)
            width.append(w)
            height.append(h)
            label.append('product')
            xmin.append(xminn)
            ymin.append(yminn)
            xmax.append(xmaxx)
            ymax.append(ymaxx)
    df = pd.DataFrame()
    df['filename']=filename
    df['width'] = width
    df['height']= height
    df['class'] = label
    df['xmin'] = xmin
    df['ymin'] = ymin
    df['xmax'] = xmax
    df['ymax'] = ymax
    df.to_csv(data_part+'_labels.csv', index = False)
    
    
test_train_split('test')
test_train_split('train')

#Training 
## TfRecord generation

import tensorflow as tf




    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    