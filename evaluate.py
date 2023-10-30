import cv2, os, random, time
import numpy as np
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import argparse

from numpy import asarray
from numpy import zeros

from data import unet, vgg_unet1, segnet1, vgg_segnet1, IoU, Dice, Sens, Spec, miou

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
import keras.backend as K

from imagecorruptions.imagecorruptions import corruptions


def parse_args():
    """
  Parse input arguments
  """
    parser = argparse.ArgumentParser(description='Training with Consistency loss')
    parser.add_argument('--data_path', help='root folder', default='/content/drive/MyDrive/idd20k_lite/', type=str)
    parser.add_argument('--aug', help='augmentations', default='none', type=str)
    parser.add_argument('--model', help='type of model', default='vgg_unet1', type=str)

    args = parser.parse_args()
    return args
    
args = parse_args()
print('Called with args:')
print(args)

data = args.data_path
img_test = data + 'leftImg8bit/val/'
seg_test = data + 'gtFine/val/'
img_test_files = sorted(os.listdir(img_test))
seg_test_files = sorted(os.listdir(seg_test))
test_img = os.listdir(img_test)
test_img.sort()
test_seg = os.listdir(seg_test)
test_seg.sort()

img_test_ = []
seg_test_ = []

height=256
width=320
n_classes=7

def prepare_corr_image_data(path, data, corr):
    src=path+data
    img = cv2.imread(src)
    img=cv2.resize(img,(width,height))
    img = corruptions.corrupt(img, corruption_number=corr, severity=3)
    img = np.float32(img)  / 255.
    return img

def prepare_label_data(path,data):
    label = np.zeros((height, width, n_classes))
    src=path+data
    img = cv2.imread(src)
    img=cv2.resize(img,(width,height))
    img1=img[:,:,0]
    for i in range(n_classes):
        label[:,:,i] = (img1==i).astype(int)
    return label

for img_folder in test_img:
    lid = sorted(os.listdir(img_test+img_folder))
    lid_ = []
    for ld in lid:
      lid_.append(img_folder+'/'+ld)
    img_test_.extend(sorted(lid_))

for seg_folder in test_seg:
    lid = sorted(os.listdir(seg_test+seg_folder))
    lid_ = []
    for ld in lid:
      lid_.append(seg_folder+'/'+ld)
    seg_test_.extend(sorted(lid_))

test_img = img_test_
test_seg = seg_test_

test_seg_label=[]
test_inst_seg=[]
for i in range(len(test_seg)):
    if(i%2 !=0):
        test_seg_label.append(test_seg[i])
    else:
        test_inst_seg.append(test_seg[i])

test_seg_label = sorted(test_seg_label)
test_inst_seg = sorted(test_inst_seg)

aug = args.aug
f_aug = ''

if args.model == 'unet':
    unet_model1 = keras.models.load_model(f'/content/drive/MyDrive/ood_seg/m1_woc_{aug}_{f_aug}.h5')
elif args.model == 'vgg_unet1':
    unet_model1 = keras.models.load_model(f'/content/drive/MyDrive/ood_seg/m2_woc_{aug}_{f_aug}.h5')
elif args.model == 'segnet1':
    unet_model1 = keras.models.load_model(f'/content/drive/MyDrive/ood_seg/m3_woc_{aug}_{f_aug}.h5')
elif args.model == 'vgg_segnet1':
    unet_model1 = keras.models.load_model(f'/content/drive/MyDrive/ood_seg/m4_woc_{aug}_{f_aug}.h5')
else:
    raise Exception

t = 0.0
t1 = 0.0
t2 = 0.0
t3 = 0.0

for corr in range(15):
    X_test_c, y_test_c = [], []

    for i in range(len(test_img)):
        ta = prepare_corr_image_data(img_test,test_img[i], corr)
        X_test_c.append(ta)

    for i in range(len(test_seg_label)):
        y_test_c.append(prepare_label_data(seg_test,test_seg_label[i]))

    # High RAM
    X_test_c=np.array(X_test_c)
    y_test_c=np.array(y_test_c)
    print('Corrupt Test Data : ')
    print('Images-',X_test_c.shape)
    print('Labels-',y_test_c.shape)
    print('='*40)

    sio = 0.0
    sdi = 0.0
    ssen = 0.0
    sspec = 0.0
    c = 0
    for i in range(len(X_test_c)):
        y_pred = unet_model1.predict(X_test_c[i:(i+1)])
        io = IoU(y_test_c[i:(i+1)], y_pred)
        sio += io
        di = Dice(y_test_c[i:i+1], y_pred)
        sdi += di
        sen = Sens(y_test_c[i:i+1], y_pred)
        ssen += sen
        spec = Spec(y_test_c[i:i+1], y_pred)
        sspec += spec
        c += 1
    print(f'OOD IoU Metric for corr {corr}: ',sio/c)
    print(f'OOD Dice Metric for corr {corr}: ',sdi/c)
    print(f'OOD Sens Metric for corr {corr}: ',ssen/c)
    print(f'OOD Spec Metric for corr {corr}: ',sspec/c)
    t += sio/c
    t1 += sdi/c
    t2 += ssen/c
    t3 += sspec/c

print('Mean iou: ', t/15)
print('Mean dice: ', t1/15)
print('Mean sens: ', t2/15)
print('Mean spec: ', t3/15)
print('\nEvaluation Done...')
