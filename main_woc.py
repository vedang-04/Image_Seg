import cv2, os, random, time
import numpy as np
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import argparse

from numpy import asarray
from numpy import zeros

from data import unet, vgg_unet1, segnet1, vgg_segnet1, IoU, miou
from augmentations import geo, mixup, autoaug, color, np_style

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
    parser.add_argument('--bs', help='batch size', default=8, type=int)
    parser.add_argument('--lr', help='learning rate', default=0.0001, type=float)
    parser.add_argument('--epochs', help='max epochs', default=30, type=int)
    parser.add_argument('--model', help='type of model', default='vgg_unet1', type=str)

    args = parser.parse_args()
    return args

args = parse_args()
print('Called with args:')
print(args)

data = args.data_path
img_train = data + 'leftImg8bit/train/'
seg_train = data + 'gtFine/train/'
img_test = data + 'leftImg8bit/val/'
seg_test = data + 'gtFine/val/'
img_train_files = sorted(os.listdir(img_train))
seg_train_files = sorted(os.listdir(seg_train))
img_test_files = sorted(os.listdir(img_test))
seg_test_files = sorted(os.listdir(seg_test))
train_img = os.listdir(img_train)
train_img.sort()
train_seg = os.listdir(seg_train)
train_seg.sort()
test_img = os.listdir(img_test)
test_img.sort()
test_seg = os.listdir(seg_test)
test_seg.sort()

img_train_ = []
img_test_ = []
seg_train_ = []
seg_test_ = []

height=256
width=320
n_classes=7
def prepare_image_data(path,data):
    src=path+data
    img = cv2.imread(src)
    img=cv2.resize(img,(width,height))
    img = np.float32(img)  / 255
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

for img_folder in train_img:
    lid = sorted(os.listdir(img_train+img_folder))
    lid_ = []
    for ld in lid:
      lid_.append(img_folder+'/'+ld)
    img_train_.extend(sorted(lid_))

for img_folder in test_img:
    lid = sorted(os.listdir(img_test+img_folder))
    lid_ = []
    for ld in lid:
      lid_.append(img_folder+'/'+ld)
    img_test_.extend(sorted(lid_))

for seg_folder in train_seg:
    lid = sorted(os.listdir(seg_train+seg_folder))
    lid_ = []
    for ld in lid:
      lid_.append(seg_folder+'/'+ld)
    seg_train_.extend(sorted(lid_))

for seg_folder in test_seg:
    lid = sorted(os.listdir(seg_test+seg_folder))
    lid_ = []
    for ld in lid:
      lid_.append(seg_folder+'/'+ld)
    seg_test_.extend(sorted(lid_))

train_img = img_train_
test_img = img_test_
train_seg = seg_train_
test_seg = seg_test_

train_seg_label=[]
train_inst_seg=[]
for i in range(len(train_seg)):
    if(i%2 !=0):
        train_seg_label.append(train_seg[i])
    else:
        train_inst_seg.append(train_seg[i])

test_seg_label=[]
test_inst_seg=[]
for i in range(len(test_seg)):
    if(i%2 !=0):
        test_seg_label.append(test_seg[i])
    else:
        test_inst_seg.append(test_seg[i])

train_seg_label = sorted(train_seg_label)
train_inst_seg = sorted(train_inst_seg)

test_seg_label = sorted(test_seg_label)
test_inst_seg = sorted(test_inst_seg)

X_train, y_train, X_test, y_test = [], [], [], []

for i in range(len(train_img)):
    X_train.append(prepare_image_data(img_train,train_img[i]))

for i in range(len(train_seg_label)):
    y_train.append(prepare_label_data(seg_train,train_seg_label[i]))

for i in range(len(test_img)):
    X_test.append(prepare_image_data(img_test,test_img[i]))

for i in range(len(test_seg_label)):
    y_test.append(prepare_label_data(seg_test,test_seg_label[i]))

# High RAM
X_train=np.array(X_train)
y_train=np.array(y_train)
X_train_ = X_train.copy()
y_train_ = y_train.copy()
X_test=np.array(X_test)
y_test=np.array(y_test)
print('Train Data : ')
print('Images-',X_train.shape)
print('Labels-',y_train.shape)
print('='*40)
print('Test Data : ')
print('Images-',X_test.shape)
print('Labels-',y_test.shape)
print('='*40)

# Data aug
p = 0.5
aug = args.aug
f_aug = ''
bs = args.bs
for i in range(len(X_train)):
    if random.random()>=p:
        j = np.random.randint(0,len(X_train))
        if j==i:
            j+=1
        if aug == 'geo':
            if f_aug == 'f_mixup':
                xt = X_train[i].copy()
                yt = y_train[i].copy()
            X_train[i], y_train[i] = geo(X_train[i].copy(), y_train[i].copy())
            if f_aug == 'f_mixup':
                X_train[i], y_train[i] = mixup(xt, yt, X_train[i].copy(), y_train[i].copy())

        if aug == 'mixup':
            X_train[i], y_train[i] = mixup(X_train[i].copy(), y_train[i].copy(), X_train[j].copy(), y_train[j].copy())
        
        if aug == 'autoaug':
            if f_aug == 'f_mixup':
                xt = X_train[i].copy()
                yt = y_train[i].copy()
            X_train[i], y_train[i] = autoaug(X_train[i], y_train[i])
            if f_aug == 'f_mixup':
                X_train[i], y_train[i] = mixup(xt, yt, X_train[i].copy(), y_train[i].copy())

        if aug == 'color':
            if f_aug == 'f_mixup':
                xt = X_train[i].copy()
                yt = y_train[i].copy()
            X_train[i] = color(X_train[i])
            if f_aug == 'f_mixup':
                X_train[i], y_train[i] = mixup(xt, yt, X_train[i].copy(), y_train[i].copy())

        if aug == 'np_style':
            if f_aug == 'f_mixup':
                xt = X_train[i].copy()
                yt = y_train[i].copy()
            if (i+bs) >= len(X_train):
                f = i-bs
                X_train[i] = np_style(X_train[i], X_train[f:i])
            else:
                f = i+bs
                X_train[i] = np_style(X_train[i], X_train[i:f])
            if f_aug == 'f_mixup':
                X_train[i], y_train[i] = mixup(xt, yt, X_train[i].copy(), y_train[i].copy())

# unet, vgg_unet1, segnet1, vgg_segnet1
if args.model == 'unet':
    unet_model1 = unet()
elif args.model == 'vgg_unet1':
    unet_model1 = vgg_unet1()
elif args.model == 'segnet1':
    unet_model1 = segnet1()
elif args.model == 'vgg_segnet1':
    unet_model1 = vgg_segnet1()
else:
    raise Exception
unet_model1.summary()

class CustomAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name='custom_accuracy', **kwargs):
        super(CustomAccuracy, self).__init__(name=name, **kwargs)
        self.total = self.add_weight('total', initializer='zeros')
        self.count = self.add_weight('count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Calculate custom accuracy and update the total and count
        accuracy = miou(y_true, y_pred)
        self.total.assign_add(tf.reduce_sum(accuracy))
        self.count.assign_add(tf.cast(tf.shape(y_true)[0], dtype=tf.float32))

    def result(self):
        # Compute and return the final result
        return self.total / self.count

    def reset_states(self):
        # Reset the accumulated values
        self.total.assign(0)
        self.count.assign(0)

def custom_loss(y_true, y_pred):
    net_loss = CategoricalCrossentropy()(y_true, y_pred) 
    return net_loss

loss_fn = custom_loss
lr = args.lr #0.0001 ##0.001 # change if possible
optimizer = keras.optimizers.Adam(learning_rate=lr)

train_loss_metric = tf.metrics.Mean()
train_accuracy_metric = CustomAccuracy()
val_acc_metric = CustomAccuracy()
num_epochs = args.epochs #30

best = 0
lmod = []
train_loss_list = []
val_loss_list = []

for epoch in range(num_epochs):
    print("\nStart of epoch %d" % (epoch,))
    start_time = time.time()

    t_l = 0.0
    v_l = 0.0

    trlim = len(X_train)//bs
    if int(trlim)!=trlim:
        trlim += 1

    for i in range(trlim):  # Iterate through batches
        with tf.GradientTape() as tape:
            x_batch_c = X_train[bs*i:bs*(i+1)]
            if x_batch_c.shape[0]==0:
                continue
            y_batch = y_train[bs*i:bs*(i+1)]
            x_batch = X_train_[bs*i:bs*(i+1)]

            logits = unet_model1(x_batch_c, training=True)
            loss_value = loss_fn(y_batch, logits)
            t_l += loss_value
         
        grads = tape.gradient(loss_value, unet_model1.trainable_variables)
        optimizer.apply_gradients(zip(grads, unet_model1.trainable_variables))
        
        train_loss_metric.update_state(loss_value)
        train_accuracy_metric.update_state(y_batch, logits)
        
    t_l = t_l/trlim
    train_loss_list.append(t_l)
       
    print(f"Epoch {epoch+1}: Loss={train_loss_metric.result()}, Accuracy={train_accuracy_metric.result()}")
    
    for j in range(len(X_test)):
        x_batch_test = X_test[j:j+1]
        y_batch_test = y_test[j:j+1]
        val_logits = unet_model1(x_batch_test, training=False)
        val_acc_metric.update_state(y_batch_test, val_logits)
        val_loss_value = loss_fn(y_batch_test, val_logits)
        v_l += val_loss_value

    v_l = v_l/len(X_test)
    val_loss_list.append(v_l)

    val_acc = val_acc_metric.result()
    val_acc_metric.reset_states()
    print("Validation acc: %.4f" % (float(val_acc),))
    if val_acc>best:
        best = val_acc
        lmod.append(unet_model1)

    print("Time taken: %.2fs" % (time.time() - start_time))

if args.model == 'unet':
    lmod[-1].save(f'/content/drive/MyDrive/ood_seg/m1_woc_{aug}_{f_aug}.h5') # change if possible
elif args.model == 'vgg_unet1':
    lmod[-1].save(f'/content/drive/MyDrive/ood_seg/m2_woc_{aug}_{f_aug}.h5') # change if possible
elif args.model == 'segnet1':
    lmod[-1].save(f'/content/drive/MyDrive/ood_seg/m3_woc_{aug}_{f_aug}.h5') # change if possible
elif args.model == 'vgg_segnet1':
    lmod[-1].save(f'/content/drive/MyDrive/ood_seg/m4_woc_{aug}_{f_aug}.h5') # change if possible
else:
    raise Exception
print('Model saved and Training finished...')

fig, ax = plt.subplots()
ax.plot(list(np.arange(1,len(train_loss_list)+1)), train_loss_list, label='train_loss');
ax.plot(list(np.arange(1,len(val_loss_list)+1)), val_loss_list, label='val_loss');
ax.set_title('Epoch vs Loss')
ax.set_ylabel('loss')
ax.set_xlabel('epoch')
ax.legend(loc='best')
if args.model == 'unet':
    plt.savefig(f'/content/drive/MyDrive/ood_seg/m1_woc_{aug}_{f_aug}.png') # change if possible
elif args.model == 'vgg_unet1':
    plt.savefig(f'/content/drive/MyDrive/ood_seg/m2_woc_{aug}_{f_aug}.png') # change if possible
elif args.model == 'segnet1':
    plt.savefig(f'/content/drive/MyDrive/ood_seg/m3_woc_{aug}_{f_aug}.png') # change if possible
elif args.model == 'vgg_segnet1':
    plt.savefig(f'/content/drive/MyDrive/ood_seg/m4_woc_{aug}_{f_aug}.png') # change if possible
else:
    raise Exception
print("Plotting done")