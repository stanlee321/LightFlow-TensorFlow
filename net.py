import os
import shutil
import random
import struct
import cv2
import numpy as np

import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import EarlyStopping 

from model.lightflow import LightFlow


path = "model"
dir0 = '20181119_1'
net_name = 'lightflow'
dir_restore = os.path.join(path, 
                            net_name,
                            dir0, 
                            'model-6250')

dir_data = '/media/csc105/Data/dataset/FlyingChairs/data/'

lr_base = 1e-3              # initial learning rate
epoch_lr_decay = 500        # every # epoch, lr will decay 0.1
epoch_max = 5               # max epoch
max_to_keep = 5             # number of model to save
batch_size = 32             # bs
train_pairs_number = 20000  # number of train samples
val_iter = 2                # validation batch
use_gpu_1 = False
W, H = 512, 384
val_pairs_number = batch_size * val_iter
iter_per_epoch = train_pairs_number // batch_size
epoch_save = epoch_max // max_to_keep

########################################
dir_models = path + net_name
dir_logs = 'log/' + net_name
dir_model = dir_models + dir0
dir_log_train = dir_logs + dir0 + '_train'
dir_log_test = dir_logs + dir0 + '_test'

weights_path = path + net_name + '.h5'

if not os.path.exists(dir_models):
    os.mkdir(dir_models)
if not os.path.exists(dir_logs):
    os.mkdir(dir_logs)
if os.path.exists(dir_model):
    shutil.rmtree(dir_model)
if os.path.exists(dir_log_train):
    shutil.rmtree(dir_log_train)
if os.path.exists(dir_log_test):
    shutil.rmtree(dir_log_test)

os.mkdir(dir_model)
os.mkdir(dir_log_train)
os.mkdir(dir_log_test)
########################################


def remove_file(directory_list):
    if '.directory' in directory_list:
        directory_list.remove('.directory')
    return directory_list


def load_data():
    img1_list_t = []
    img2_list_t = []
    flow_list_t = []
    img1_list_v = []
    img2_list_v = []
    flow_list_v = []
    namelist = remove_file(os.listdir(dir_data))
    namelist.sort()
    for i in range(train_pairs_number+val_pairs_number):
        if i < train_pairs_number:
            flow_list_t.append(dir_data + namelist[3*i])
            img1_list_t.append(dir_data + namelist[3*i+1])
            img2_list_t.append(dir_data + namelist[3*i+2])
        else:
            flow_list_v.append(dir_data + namelist[3*i])
            img1_list_v.append(dir_data + namelist[3*i+1])
            img2_list_v.append(dir_data + namelist[3*i+2])

    assert len(img1_list_t) == len(img2_list_t)
    assert len(img1_list_t) == len(flow_list_t)
    assert len(img1_list_v) == len(img2_list_v)
    assert len(img1_list_v) == len(flow_list_v)
    return img1_list_t, img2_list_t, flow_list_t, img1_list_v, img2_list_v, flow_list_v


class Data(object):
    def __init__(self, list1, list2, list3, bs=batch_size, shuffle=True, minus_mean=True):
        self.list1 = list1
        self.list2 = list2
        self.list3 = list3
        self.bs = bs
        self.index = 0
        self.number = len(self.list1)
        self.index_total = range(self.number)
        self.shuffle = shuffle
        self.minus_mean = minus_mean
        if self.shuffle:
            random.shuffle(self.index_total)

    def read_flow(self, name):
        f = open(name, "rb")
        data = f.read()
        f.close()
        width = struct.unpack('@i', data[4:8])[0]
        height = struct.unpack('@i', data[8:12])[0]
        flowdata = np.zeros((height, width, 2))
        for i in range(width*height):
            data_u = struct.unpack('@f', data[12+8*i:16+8*i])[0]
            data_v = struct.unpack('@f', data[16+8*i:20+8*i])[0]
            n = int(i / width)
            k = np.mod(i, width)
            flowdata[n, k, :] = [data_u, data_v]
        return flowdata

    def next_batch(self):
        start = self.index
        self.index += self.bs
        if self.index > self.number:
            if self.shuffle: random.shuffle(self.index_total)
            self.index = 0
            start = self.index
            self.index += self.bs
        end = self.index
        img1_batch = []
        img2_batch = []
        flow_batch = []
        for i in range(start, end):
            img1 = cv2.imread(self.list1[self.index_total[i]]).astype(np.float32)
            img1_batch.append(img1)
            img2 = cv2.imread(self.list2[self.index_total[i]]).astype(np.float32)
            img2_batch.append(img2)
            flow = self.read_flow(self.list3[self.index_total[i]])
            flow_batch.append(flow)

        return np.array(img1_batch), np.array(img2_batch), np.array(flow_batch)

def main():
    # Prepare Data
    
    list1_t, list2_t, list3_t, list1_v, list2_v, list3_v = load_data()
    dataset_t = Data(list1_t, list2_t, list3_t, shuffle=True, minus_mean=False)
    dataset_v = Data(list1_v, list2_v, list3_v, shuffle=True, minus_mean=False)
    x1_v = []
    x2_v = []
    x3_v = []

    for j in range(val_iter):
        x1_b, x2_b, x3_b = dataset_v.next_batch()
        x1_v.append(x1_b)
        x2_v.append(x2_b)
        x3_v.append(x3_b)


    epochs_arr  = [   20,      5,      5]
    learn_rates = [0.001, 0.0003, 0.0001, 0.00001]

    for learn_rate, epochs in zip(learn_rates, epochs_arr):
        optimizer = SGD(nesterov=True, lr=learn_rate, momentum=0.1, decay=0.001)
        INPUT_SHAPE = (512,384,6)
        model = LightFlow.build(input_shape=INPUT_SHAPE)
        model.compile(optimizer=optimizer,loss='mean_squared_error')

        callbacks = [EarlyStopping(monitor='val_loss', patience=2, verbose=1),
                 ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True, verbose=2)]


        x1_t, x2_t, x3_t = dataset_t.next_batch()

        X_train = tf.concat([x1_t, x2_t], axis=3)
        Y_train = x3_t

        X_val = tf.concat([x1_v, x2_v], axis=3)
        Y_val = x3_v

        model.fit(x = X_train, y= Y_train, validation_data=(X_val, Y_val),
            batch_size=256, verbose=2, epochs=epochs, callbacks=callbacks, shuffle=True)
