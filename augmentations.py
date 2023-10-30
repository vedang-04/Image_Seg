import numpy as np
import cv2
import random
import math
import tensorflow as tf
from autoaugment import ImageNetPolicy, CIFAR10Policy, SVHNPolicy, SubPolicy
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import torch

def flipped(img, label):
    img = np.fliplr(img)
    label = np.fliplr(label)
    return img, label

def color(x):
    x = tf.image.random_hue(x, 0.08)  
    x = tf.image.random_saturation(x, 0.6, 1.6)
    x = tf.image.random_brightness(x, 0.05) 
    x = tf.image.random_contrast(x, 0.7, 1.3)
    return x

def random_perspective(im, label, degrees=10, translate=.1, scale=.1,
                       shear=10, perspective=0.0, border=(0, 0)):

    height = im.shape[0] + border[0] * 2  # shape(h,w,c)
    width = im.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -im.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -im.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            im = cv2.warpPerspective(im, M, dsize=(width, height), borderValue=(114, 114, 114))
        else: 
            im = cv2.warpAffine(im, M[:2], dsize=(width, height), borderValue=(114, 114, 114))
        if perspective:
            label = cv2.warpPerspective(label, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  
            label = cv2.warpAffine(label, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    return im, label

def geo(img, label):
    # img,label = random_perspective(img,label)
    img,label = flipped(img,label)
    return img,label

def mixup(im1, label1, im2, label2):
    r = np.random.beta(32.0,32.0)
    im1 = (im1 * r + im2 * (1 - r))
    label1 = (label1 * r + label2 * (1 - r))
    return im1, label1

def autoaug(im1, label1, policy_name='imagenet'):
    if policy_name == 'imagenet':
        policy = ImageNetPolicy()
    elif policy_name == 'cifar':
        policy = CIFAR10Policy()
    elif policy_name == 'svhn':
        policy = SVHNPolicy()
    else:
        raise Exception
    
    im1 = Image.fromarray((im1*255).astype('uint8'))
    im1 = policy(im1)
    im1 = np.asarray(im1).astype('float32')/255.
    return im1, label1

def np_style(im, feat):
    im = torch.unsqueeze(torch.tensor(im).permute(2,0,1).cuda(),0)
    feat = torch.tensor(feat).permute(0,3,1,2).cuda()
    feat_mean = feat.mean((2, 3), keepdim=True)  # size: B, C, 1, 1
    ones_mat = torch.ones_like(feat_mean)
    alpha = torch.normal(ones_mat, 0.75 * ones_mat)  # size: B, C, → 1, 1
    beta = torch.normal(ones_mat, 0.75 * ones_mat)  # size: B, C, → 1, 1
    output = alpha * im - alpha * feat_mean + beta * feat_mean
    output = output[0].detach().cpu().numpy().transpose(1, 2, 0)
    return output