import cv2, os, random
import numpy as np
import shutil
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Dropout, Flatten,concatenate,Input,UpSampling2D, Reshape,Permute, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D,Conv1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.layers import BatchNormalization


from numpy import asarray
from numpy import zeros
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from time import time
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as K

def IoU(y_val, y_pred):
    class_iou = []
    n_classes = 7

    y_predi = np.argmax(y_pred, axis=3)
    y_truei = np.argmax(y_val, axis=3)

    for c in range(n_classes):
        TP = np.sum((y_truei == c) & (y_predi == c))
        FP = np.sum((y_truei != c) & (y_predi == c))
        FN = np.sum((y_truei == c) & (y_predi != c))
        IoU = TP / float(TP + FP + FN)
        if(float(TP + FP + FN) == 0):
          IoU=TP/0.001
        class_iou.append(IoU)
    MIoU=sum(class_iou)/n_classes
    return MIoU

def Dice(y_val, y_pred):
    class_dic = []
    n_classes = 7

    y_predi = np.argmax(y_pred, axis=3)
    y_truei = np.argmax(y_val, axis=3)

    for c in range(n_classes):
        TP = np.sum((y_truei == c) & (y_predi == c))
        FP = np.sum((y_truei != c) & (y_predi == c))
        FN = np.sum((y_truei == c) & (y_predi != c))
        dic = 2*TP / float(2*TP + FP + FN)
        if(float(TP + FP + FN) == 0):
          dic=TP/0.001
        class_dic.append(dic)
    Mdic=sum(class_dic)/n_classes
    return Mdic

def Sens(y_val, y_pred):
    class_sen = []
    n_classes = 7

    y_predi = np.argmax(y_pred, axis=3)
    y_truei = np.argmax(y_val, axis=3)

    for c in range(n_classes):
        TP = np.sum((y_truei == c) & (y_predi == c))
        FN = np.sum((y_truei == c) & (y_predi != c))
        sen = TP / float(TP + FN)
        if(float(TP + FN) == 0):
          sen=TP/0.001
        class_sen.append(sen)
    Msen=sum(class_sen)/n_classes
    return Msen

def Spec(y_val, y_pred):
    class_spe = []
    n_classes = 7

    y_predi = np.argmax(y_pred, axis=3)
    y_truei = np.argmax(y_val, axis=3)

    for c in range(n_classes):
        FN = np.sum((y_truei == c) & (y_predi != c))
        TN = np.sum((y_truei != c) & (y_predi != c))
        spe = TN / float(TN + FN)
        if(float(TN + FN) == 0):
          spe=TN/0.001
        class_spe.append(spe)
    Mspe=sum(class_spe)/n_classes
    return Mspe

def miou( y_true, y_pred ) :
    score = tf.py_function( lambda y_true, y_pred : IoU( y_true, y_pred).astype('float32'), [y_true, y_pred], 'float32')
    return score


def unet(pretrained_weights = None):
    inputs = Input(shape=(256, 320,3))
    print(inputs , inputs.shape)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = keras.initializers.he_normal(seed=24))(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = keras.initializers.he_normal(seed=24))(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = keras.initializers.he_normal(seed=24))(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = keras.initializers.he_normal(seed=24))(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = keras.initializers.he_normal(seed=24))(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = keras.initializers.he_normal(seed=24))(conv3)
    drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)

    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = keras.initializers.he_normal(seed=24))(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = keras.initializers.he_normal(seed=24))(conv4)
    drop4 = Dropout(0.5)(conv4)

    up5 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = keras.initializers.he_normal(seed=24))(UpSampling2D(size = (2,2))(drop4))
    merge5 = concatenate([conv3,up5], axis = 3)
    conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = keras.initializers.he_normal(seed=24))(merge5)
    conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = keras.initializers.he_normal(seed=24))(conv5)

    up6 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = keras.initializers.he_normal(seed=24))(UpSampling2D(size = (2,2))(conv5))
    merge6 = concatenate([conv2,up6], axis = 3)
    conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = keras.initializers.he_normal(seed=24))(merge6)
    conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = keras.initializers.he_normal(seed=24))(conv6)

    up7 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = keras.initializers.he_normal(seed=24))(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv1,up7], axis = 3)
    conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = keras.initializers.he_normal(seed=24))(merge7)
    conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = keras.initializers.he_normal(seed=24))(conv7)

    conv8 = Conv2D(7, 3, activation = 'relu', padding = 'same', kernel_initializer = keras.initializers.he_normal(seed=24))(conv7)
    out = (Activation('softmax'))(conv8)

    model = Model(inputs,out)
    return model

from tensorflow.keras.applications.vgg16 import VGG16
encoder_vgg16 = VGG16(input_shape =  (256, 320,3), include_top = False, weights = 'imagenet')
encoder_vgg16.trainable = False

for l in encoder_vgg16.layers:
    l.trainable = False

conv1 = encoder_vgg16.get_layer("block1_conv2").output
conv2 = encoder_vgg16.get_layer("block2_conv2").output
conv3 = encoder_vgg16.get_layer("block3_conv3").output
conv4 = encoder_vgg16.get_layer("block4_conv3").output

def vgg_unet1():

    up5 = Conv2D(256, 2, activation = 'relu', padding = 'same',
                   kernel_initializer = keras.initializers.glorot_normal(seed=58))(UpSampling2D(size = (2,2))(conv4))
    merge5 = concatenate([conv3,up5], axis = 3)
    conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = keras.initializers.glorot_normal(seed=58))(merge5)
    conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = keras.initializers.glorot_normal(seed=58))(conv5)

    up6 = Conv2D(128, 2, activation = 'relu', padding = 'same',
                 kernel_initializer = keras.initializers.glorot_normal(seed=58))(UpSampling2D(size = (2,2))(conv5))
    merge6 = concatenate([conv2,up6], axis = 3)
    conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = keras.initializers.glorot_normal(seed=58))(merge6)
    conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = keras.initializers.glorot_normal(seed=58))(conv6)

    up7 = Conv2D(64, 2, activation = 'relu', padding = 'same',
                 kernel_initializer = keras.initializers.glorot_normal(seed=58))(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv1,up7], axis = 3)
    conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = keras.initializers.glorot_normal(seed=58))(merge7)
    conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = keras.initializers.glorot_normal(seed=58))(conv7)

    conv8 = Conv2D(7, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = keras.initializers.glorot_normal(seed=58))(conv7)
    out = (Activation('softmax'))(conv8)

    model = Model(encoder_vgg16.input,out)

    return model

def segnet1():
    inputs = Input(shape=(256, 320,3))

    # Encoder
    conv1 = Conv2D(64, 3, padding="same", kernel_initializer =
                          keras.initializers.glorot_normal(seed=84))(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation("relu")(conv1)
    conv2 = Conv2D(64, 3, padding="same", kernel_initializer =
                          keras.initializers.glorot_normal(seed=84))(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation("relu")(conv2)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, 3, padding="same", kernel_initializer =
                          keras.initializers.glorot_normal(seed=84))(pool1)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation("relu")(conv3)
    conv4 = Conv2D(128, 3, padding="same", kernel_initializer =
                          keras.initializers.glorot_normal(seed=84))(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation("relu")(conv4)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(256, 3, padding="same", kernel_initializer = keras.initializers.glorot_normal(seed=84))(pool2)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation("relu")(conv5)
    conv6 = Conv2D(256, 3, padding="same", kernel_initializer = keras.initializers.glorot_normal(seed=84))(conv5)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation("relu")(conv6)
    conv7 = Conv2D(256, 3, padding="same", kernel_initializer = keras.initializers.glorot_normal(seed=84))(conv6)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation("relu")(conv7)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv7)

    conv8 = Conv2D(512, 3, padding="same", kernel_initializer = keras.initializers.glorot_normal(seed=84))(pool3)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation("relu")(conv8)
    conv9 = Conv2D(512, 3, padding="same", kernel_initializer = keras.initializers.glorot_normal(seed=84))(conv8)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation("relu")(conv9)
    conv10 = Conv2D(512, 3, padding="same", kernel_initializer = keras.initializers.glorot_normal(seed=84))(conv9)
    conv10 = BatchNormalization()(conv10)
    conv10 = Activation("relu")(conv10)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv10)

    # Decoder
    up1 = UpSampling2D(size=(2, 2))(pool4)
    conv11 = Conv2D(512, 3, padding="same", kernel_initializer = keras.initializers.glorot_normal(seed=84))(up1)
    conv11 = BatchNormalization()(conv11)
    conv11 = Activation("relu")(conv11)
    conv12 = Conv2D(512, 3, padding="same", kernel_initializer = keras.initializers.glorot_normal(seed=84))(conv11)
    conv12 = BatchNormalization()(conv12)
    conv12 = Activation("relu")(conv12)
    conv13 = Conv2D(512, 3, padding="same", kernel_initializer = keras.initializers.glorot_normal(seed=84))(conv12)
    conv13 = BatchNormalization()(conv13)
    conv13 = Activation("relu")(conv13)

    up2 = UpSampling2D(size=(2, 2))(conv13)
    conv14 = Conv2D(256, 3, padding="same", kernel_initializer = keras.initializers.glorot_normal(seed=84))(up2)
    conv14 = BatchNormalization()(conv14)
    conv14 = Activation("relu")(conv14)
    conv15 = Conv2D(256, 3, padding="same", kernel_initializer = keras.initializers.glorot_normal(seed=84))(conv14)
    conv15 = BatchNormalization()(conv15)
    conv15 = Activation("relu")(conv15)
    conv16 = Conv2D(256, 3, padding="same", kernel_initializer = keras.initializers.glorot_normal(seed=84))(conv15)
    conv16 = BatchNormalization()(conv16)
    conv16 = Activation("relu")(conv16)

    up3 = UpSampling2D(size=(2, 2))(conv16)
    conv17 = Conv2D(128, 3, padding="same", kernel_initializer = keras.initializers.glorot_normal(seed=84))(up3)
    conv17 = BatchNormalization()(conv17)
    conv17 = Activation("relu")(conv17)
    conv18 = Conv2D(128, 3, padding="same", kernel_initializer = keras.initializers.glorot_normal(seed=84))(conv17)
    conv18 = BatchNormalization()(conv18)
    conv18 = Activation("relu")(conv18)

    up4 = UpSampling2D(size=(2, 2))(conv18)
    conv19 = Conv2D(64, 3, padding="same", kernel_initializer = keras.initializers.glorot_normal(seed=84))(up4)
    conv19 = BatchNormalization()(conv19)
    conv19 = Activation("relu")(conv19)
    conv20 = Conv2D(64, 3, padding="same", kernel_initializer = keras.initializers.glorot_normal(seed=84))(conv19)
    conv20 = BatchNormalization()(conv20)
    conv20 = Activation("relu")(conv20)

    conv21 = Conv2D(7, 3, padding="same", kernel_initializer = keras.initializers.glorot_normal(seed=84))(conv20)
    out = Activation("softmax")(conv21)
    model = Model(inputs,out)
    return model

vgg_encoder = encoder_vgg16.get_layer("block5_conv3").output
def vgg_segnet1():
    # Decoder
    up1 = UpSampling2D(size=(2, 2))(vgg_encoder)
    conv11 = Conv2D(512, 3, padding="same", kernel_initializer = keras.initializers.he_normal(seed=84))(up1)
    conv11 = BatchNormalization()(conv11)
    conv11 = Activation("relu")(conv11)
    conv12 = Conv2D(512, 3, padding="same", kernel_initializer = keras.initializers.he_normal(seed=84))(conv11)
    conv12 = BatchNormalization()(conv12)
    conv12 = Activation("relu")(conv12)
    conv13 = Conv2D(512, 3, padding="same", kernel_initializer = keras.initializers.he_normal(seed=84))(conv12)
    conv13 = BatchNormalization()(conv13)
    conv13 = Activation("relu")(conv13)

    up2 = UpSampling2D(size=(2, 2))(conv13)
    conv14 = Conv2D(256, 3, padding="same", kernel_initializer = keras.initializers.he_normal(seed=84))(up2)
    conv14 = BatchNormalization()(conv14)
    conv14 = Activation("relu")(conv14)
    conv15 = Conv2D(256, 3, padding="same", kernel_initializer = keras.initializers.he_normal(seed=84))(conv14)
    conv15 = BatchNormalization()(conv15)
    conv15 = Activation("relu")(conv15)
    conv16 = Conv2D(256, 3, padding="same", kernel_initializer = keras.initializers.he_normal(seed=84))(conv15)
    conv16 = BatchNormalization()(conv16)
    conv16 = Activation("relu")(conv16)

    up3 = UpSampling2D(size=(2, 2))(conv16)
    conv17 = Conv2D(128, 3, padding="same", kernel_initializer = keras.initializers.he_normal(seed=84))(up3)
    conv17 = BatchNormalization()(conv17)
    conv17 = Activation("relu")(conv17)
    conv18 = Conv2D(128, 3, padding="same", kernel_initializer = keras.initializers.he_normal(seed=84))(conv17)
    conv18 = BatchNormalization()(conv18)
    conv18 = Activation("relu")(conv18)

    up4 = UpSampling2D(size=(2, 2))(conv18)
    conv19 = Conv2D(64, 3, padding="same", kernel_initializer = keras.initializers.he_normal(seed=84))(up4)
    conv19 = BatchNormalization()(conv19)
    conv19 = Activation("relu")(conv19)
    conv20 = Conv2D(64, 3, padding="same", kernel_initializer = keras.initializers.he_normal(seed=84))(conv19)
    conv20 = BatchNormalization()(conv20)
    conv20 = Activation("relu")(conv20)

    conv21 = Conv2D(7, 3, padding="same", kernel_initializer = keras.initializers.he_normal(seed=84))(conv20)
    out = Activation("softmax")(conv21)
    model = Model(encoder_vgg16.input,out)
    return model