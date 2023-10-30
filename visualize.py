import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import cv2, os, random, time
import shutil
import pandas as pd
from numpy import asarray
from numpy import zeros
import argparse

from data import unet, vgg_unet1, segnet1, vgg_segnet1, IoU, Dice, Sens, Spec, miou
from imagecorruptions.imagecorruptions import corruptions

import tensorflow as tf
from tensorflow import keras
import keras.backend as K

def parse_args():
    """
  Parse input arguments
  """
    parser = argparse.ArgumentParser(description='Training with Consistency loss')
    parser.add_argument('--data_path', help='root folder', default='/content/drive/MyDrive/idd20k_lite/', type=str)
    parser.add_argument('--aug', help='augmentations', default='none', type=str)
    parser.add_argument('--model', help='type of model', default='vgg_unet1', type=str)
    parser.add_argument('--corr', help='use corrupton', default=0, type=int)
    parser.add_argument('--is_pred', help='predictions or gt', default=0, type=int)
    parser.add_argument('--img_num', help='image number to viz', default=5, type=int)
    parser.add_argument('--is_woc', help='model trained without cl', default=1, type=int)
    parser.add_argument('--is_plot_model', help='model plotting', default=0, type=int)

    args = parser.parse_args()
    return args

args = parse_args()
print('Called with args:')
print(args)

image_paths = glob(args.data_path + 'leftImg8bit/val/*/*_image.jpg')
label_paths = [p.replace('leftImg8bit', 'gtFine').replace('_image.jpg', '_label.png') for p in image_paths]

# Assigning some RGB colors for the 7 + 1 (Misc) classes
colors = np.array([
    [128, 64, 18],      # Drivable
    [244, 35, 232],     # Non Drivable
    [220, 20, 60],      # Living Things
    [0, 0, 230],        # Vehicles
    [220, 190, 40],     # Road Side Objects
    [70, 70, 70],       # Far Objects
    [70, 130, 180],     # Sky
    [0, 0, 0]           # Misc
], dtype=np.int)

aug = args.aug
f_aug = '' 
if args.model == 'unet':
    if args.is_woc:
        unet_model1 = keras.models.load_model(f'/content/drive/MyDrive/ood_seg/m1_woc_{aug}_{f_aug}.h5')
    else:
        unet_model1 = keras.models.load_model(f'/content/drive/MyDrive/ood_seg/m1_{aug}_{f_aug}.h5')
    if args.is_plot_model:
        if args.is_plot_model:
            tf.keras.utils.plot_model(
            unet_model1,
            to_file="/content/drive/MyDrive/ood_seg/m1.png",
            show_shapes=False,
            show_dtype=False,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=False,
            dpi=96,
            layer_range=None,
            show_layer_activations=False,
            show_trainable=False,
        )
elif args.model == 'vgg_unet1':
    if args.is_woc:
        unet_model1 = keras.models.load_model(f'/content/drive/MyDrive/ood_seg/m2_woc_{aug}_{f_aug}.h5')
    else:
        unet_model1 = keras.models.load_model(f'/content/drive/MyDrive/ood_seg/m2_{aug}_{f_aug}.h5')
    if args.is_plot_model:
        if args.is_plot_model:
            tf.keras.utils.plot_model(
            unet_model1,
            to_file="/content/drive/MyDrive/ood_seg/m2.png",
            show_shapes=False,
            show_dtype=False,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=False,
            dpi=96,
            layer_range=None,
            show_layer_activations=False,
            show_trainable=False,
        )
elif args.model == 'segnet1':
    if args.is_woc:
        unet_model1 = keras.models.load_model(f'/content/drive/MyDrive/ood_seg/m3_woc_{aug}_{f_aug}.h5')
    else:
        unet_model1 = keras.models.load_model(f'/content/drive/MyDrive/ood_seg/m3_{aug}_{f_aug}.h5')
    if args.is_plot_model:
        if args.is_plot_model:
            tf.keras.utils.plot_model(
            unet_model1,
            to_file="/content/drive/MyDrive/ood_seg/m3.png",
            show_shapes=False,
            show_dtype=False,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=False,
            dpi=96,
            layer_range=None,
            show_layer_activations=False,
            show_trainable=False,
        )
elif args.model == 'vgg_segnet1':
    if args.is_woc:
        unet_model1 = keras.models.load_model(f'/content/drive/MyDrive/ood_seg/m4_woc_{aug}_{f_aug}.h5')
    else:
        unet_model1 = keras.models.load_model(f'/content/drive/MyDrive/ood_seg/m4_{aug}_{f_aug}.h5')
    if args.is_plot_model:
        tf.keras.utils.plot_model(
            unet_model1,
            to_file="/content/drive/MyDrive/ood_seg/m4.png",
            show_shapes=False,
            show_dtype=False,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=False,
            dpi=96,
            layer_range=None,
            show_layer_activations=False,
            show_trainable=False,
        )
else:
    raise Exception

if args.is_plot_model:
    print('Model plotted..')
    raise Exception
    
corr = args.corr
is_pred = args.is_pred
height=256
width=320
n_classes=7
print(len(image_paths))

for i in range(args.img_num, len(image_paths)):
    print(image_paths[i], label_paths[i])
    image_frame = cv2.imread(image_paths[i])
    image_frame = image_frame.astype('uint8')
    cv2.imwrite('/content/drive/MyDrive/ood_seg/img.jpg', image_frame)
    if not args.is_pred:
        label_map = cv2.imread(label_paths[i])
        label_map = label_map[:,:,-1]
        label_map = label_map.astype('uint8')
        color_image = np.zeros((label_map.shape[0], label_map.shape[1], 3), dtype=np.int)
        for j in range(7):
            color_image[label_map == j] = colors[j]
        color_image[label_map == 255] = colors[7]
        color_image = color_image.astype('uint8')
        cv2.imwrite(f'/content/drive/MyDrive/ood_seg/fig_{i}_gt.jpg', color_image)
    
    for cor in range(15):
        image_frame = cv2.imread(image_paths[i])
        image_frame = image_frame.astype('uint8')
        if args.corr:
            image_frame = corruptions.corrupt(image_frame, corruption_number=cor,  severity=3)
            image_frame = image_frame.astype('uint8')
            cv2.imwrite(f'/content/drive/MyDrive/ood_seg/corr_img_{i}_{cor}.jpg', image_frame)

        if is_pred:
            image_frame=cv2.resize(image_frame,(width,height))
            image_frame = image_frame[np.newaxis, :]
            image_frame = image_frame/255
            y_pred = unet_model1.predict(image_frame)
            label_map = np.argmax(y_pred, axis=3)
            label_map = label_map[0]
        label_map = label_map.astype('uint8')
        color_image = np.zeros((label_map.shape[0], label_map.shape[1], 3), dtype=np.int)
        for j in range(7):
            color_image[label_map == j] = colors[j]
        color_image[label_map == 255] = colors[7]
        color_image = color_image.astype('uint8')

        cv2.imwrite(f'/content/drive/MyDrive/ood_seg/fig_{i}_{cor}.jpg', color_image)
        if not args.corr:
            break
    break