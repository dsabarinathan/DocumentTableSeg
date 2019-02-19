# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 22:13:38 2019

@author: sabari
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from models import tableSeg
from preprocessing import sauvola_thresholding

from scipy import ndimage
from keras.preprocessing import image
from keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
import cv2
import tensorflow as tf
# Set some parameters
IMG_WIDTH = 320
IMG_HEIGHT = 384
IMG_CHANNELS = 3

seed = 42
#random.seed = seed
np.random.seed=seed



MASK_PATH="./train/Mask/"
TRAIN_PATH="./train/image/"
train_ID=os.listdir(TRAIN_PATH)
MODEL_PATH="./model/"




X_train = np.zeros((len(train_ID),IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_ID), IMG_HEIGHT,IMG_WIDTH, 1), dtype=np.uint8)

count_img=0
for i in range(len(train_ID)):
    imr=cv2.imread(TRAIN_PATH+train_ID[i])
    
    grayImage=cv2.cvtColor(imr,cv2.COLOR_BGR2GRAY)

    thresholdedImage=sauvola_thresholding(grayImage)
    
    """
    Apply the distance transform to identify the high frequency
    
    """
    distTransformImage=cv2.resize(ndimage.distance_transform_edt(thresholdedImage),(IMG_WIDTH,IMG_HEIGHT))
    Lineardist_transform1 = cv2.resize(cv2.distanceTransform(grayImage,cv2.DIST_L1,5),(IMG_WIDTH,IMG_HEIGHT))
    Lineardist_transform2 = cv2.resize(cv2.distanceTransform(grayImage,cv2.DIST_L2,5),(IMG_WIDTH,IMG_HEIGHT))
    

    X_train[i,:,:,0]=distTransformImage
    X_train[i,:,:,1]=distTransformImage
    X_train[i,:,:,2]=distTransformImage


    maskPath=MASK_PATH+train_ID[i][0:len(train_ID[i])-4]+'_groundTruth.jpg'
    
    imr0=cv2.imread(maskPath)
    resized1=cv2.resize(imr0,( IMG_WIDTH,IMG_HEIGHT))>0
    Y_train[i]=np.reshape(np.uint8(resized1[:,:,0]),(384,320,1))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
def dice_coe(output, target, loss_type='jaccard', axis=(1, 2, 3), smooth=1e-5):

    inse = tf.reduce_sum(output * target, axis=axis)
    if loss_type == 'jaccard':
        l = tf.reduce_sum(output * output, axis=axis)
        r = tf.reduce_sum(target * target, axis=axis)
    elif loss_type == 'sorensen':
        l = tf.reduce_sum(output, axis=axis)
        r = tf.reduce_sum(target, axis=axis)
    else:
        raise Exception("Unknow loss_type")
    ## old axis=[0,1,2,3]
    # dice = 2 * (inse) / (l + r)
    # epsilon = 1e-5
    # dice = tf.clip_by_value(dice, 0, 1.0-epsilon) # if all empty, dice = 1
    ## new haodong
    dice = (2. * inse + smooth) / (l + r + smooth)
    ##
    dice = tf.reduce_mean(dice)

    return dice


BATCH_SIZE=4
seed=42
# Creating the training Image and Mask generator
image_datagen = image.ImageDataGenerator(shear_range=0.5, rotation_range=45, zoom_range=0.2, width_shift_range=0.2, height_shift_range=0.2, fill_mode='reflect',horizontal_flip=True,vertical_flip=True)
mask_datagen = image.ImageDataGenerator(shear_range=0.5, rotation_range=45, zoom_range=0.2, width_shift_range=0.2, height_shift_range=0.2, fill_mode='reflect',horizontal_flip=True,vertical_flip=True)
# Keep the same seed for image and mask generators so they fit together

image_datagen.fit(X_train[:int(X_train.shape[0]*0.7)], augment=True, seed=seed)
mask_datagen.fit(Y_train[:int(Y_train.shape[0]*0.7)], augment=True, seed=seed)

x=image_datagen.flow(X_train[:int(X_train.shape[0]*0.7)],batch_size=BATCH_SIZE,shuffle=True, seed=seed)
y=mask_datagen.flow(Y_train[:int(Y_train.shape[0]*0.7)],batch_size=BATCH_SIZE,shuffle=True, seed=seed)



# Creating the validation Image and Mask generator
image_datagen_val = image.ImageDataGenerator()
mask_datagen_val = image.ImageDataGenerator()

image_datagen_val.fit(X_train[int(X_train.shape[0]*0.7):], augment=True, seed=seed)
mask_datagen_val.fit(Y_train[int(Y_train.shape[0]*0.7):], augment=True, seed=seed)

x_val=image_datagen_val.flow(X_train[int(X_train.shape[0]*0.7):],batch_size=BATCH_SIZE,shuffle=True, seed=seed)
y_val=mask_datagen_val.flow(Y_train[int(Y_train.shape[0]*0.7):],batch_size=BATCH_SIZE,shuffle=True, seed=seed)



train_generator = zip(x, y)
val_generator = zip(x_val, y_val)



model=tableSeg(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)



filepath=MODEL_PATH+"weights-table-segmentation.hdf5"

model_checkpoint = ModelCheckpoint(filepath,monitor='val_loss', 
                                   mode = 'min', save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', mode = 'min',factor=0.5, patience=5, min_lr=0.00001, verbose=1)

#model.load_weights(filepath=MODEL_PATH+"weights-new_loss_keras_lovasz_hinge_improvement.hdf5")
results = model.fit(X_train, Y_train, validation_split=0.2, batch_size=16, epochs=5000,callbacks=[model_checkpoint,reduce_lr],shuffle=True)
