import os, random
import ntpath
import SimpleITK
from matplotlib import pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
from itertools import product

import keras
from keras import backend as K


class BatchGenerator(keras.utils.Sequence):

    def __init__(self, data, patch_size, step_size=(10,10,10), batch_size=6, sampling=None):
        self.patch_size,self.data,self.batch_size,self.sampling = patch_size,data,batch_size,sampling
        self.output_size = get_output_size(patch_size)
        self.images = [readImg(path) for path in data['image'].values]
        self.fissuremasks = [readImg(path) for path in data['fissuremask'].values]
        self.lungmasks = [readImg(path) for path in data['lungmask'].values]
        self.indices, self.labels = self.__generate_indices(self.images, patch_size, step_size)
        self.samples = []

    def __len__(self):
        return int(np.floor(len(self.samples) / self.batch_size))

    def __getitem__(self, index):
        idxs = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        X = np.empty((self.batch_size, *self.patch_size, 1))
        y = np.empty((self.batch_size, *self.output_size, 1))
        for i, idx in enumerate(idxs):
            X[i, ] = self.get_patch(idx[1:], self.images[idx[0]], self.patch_size)[:, :, :, np.newaxis]
            y[i, ] = self.get_patch(idx[1:], self.fissuremasks[idx[0]], self.output_size)[:, :, :, np.newaxis]
        return X, y

    def on_epoch_end(self):
        if self.sampling:
            self.samples, _ = self.sampling(self.indices, self.labels)
        else:
            self.samples = self.indices

    @staticmethod
    def get_patch(idx, image, patch_size):
        z, y, x = idx
        c, h, w = patch_size
        return image[int(z-(c/2)):int(z+(c/2)),int(y-(h/2)):int(y+(h/2)),int(x-(w/2)):int(x+(w/2))]

    def __generate_indices(self, images, patch_size, step):
        indices, labels = [], []
        zh, yh, xh = tuple((np.array(patch_size)/2).astype(int))
        for i, image in enumerate(images):
            c, h, w = image.shape
            for z, y, x in product(range(zh,c-zh,step[0]), range(yh,h-yh,step[1]),range(xh,w-xh,step[2])):
                indices.append([i,z,y,x])
                labels.append(np.amax(self.get_patch((z,y,x), self.fissuremasks[i], self.output_size)))
        return np.array(indices), np.array(labels)


def get_output_size(input_size):
    output_size = (((((((np.floor((np.floor((np.floor((np.array(input_size)-4)/2)-4)/2)-4)/2)-4)*2)-4)*2)-4)*2)-4)
    return tuple(output_size.astype(int))


def readImg(img_path):
    img = SimpleITK.ReadImage(img_path)
    img_array = SimpleITK.GetArrayFromImage(img)
    return img_array


def load_unique_image_names(folder):
    uniqueimglist = []
    for file in os.listdir(folder):
        file = file.replace(".mhd", "")
        file = file.replace("_lm", "")
        file = file.replace("_fm", "")
        file = file.replace(".zraw", "")
        if ".csv" not in file:
            uniqueimglist.append(file)
    uniqueimglist = list(set(uniqueimglist))
    return uniqueimglist


def load_training_set(folder):
    ''' Load training data from a folder'''
    fileList = load_unique_image_names(folder)

    trainSet = []
    for file in fileList:
        filePath = folder + '/' + file
        image = lungMask = fissureMask = None
        try:
            image = SimpleITK.ReadImage(filePath + '.mhd')
            lungMask = SimpleITK.ReadImage(filePath + '_lm.mhd')
            fissureMask = SimpleITK.ReadImage(filePath + '_fm.mhd')
            label = file[0]
            trainSet.append({'name': file,
                             'image': image,
                             'lungmask': lungMask,
                             'fissuremask': fissureMask,
                             'label': label})
        except:
            print("Error reading file: " + file)

    return trainSet


def get_exact_csv_set(folder, label):
    return pd.read_csv(folder + '/LUT-' + label + '.csv')


# Loss calculation for 3D U-net
def dice_coefficient(y_true, y_pred, smooth=1.):
    bglabel = 0
    cflabel = 2
    iflabel = 4
    importance_factors = [0.001, 0.5, 0.5]
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    penalty_mask = (y_true_f == bglabel) * importance_factors[0]
    + (y_true_f == cflabel) * importance_factors[1]
    + (y_true_f == iflabel) * importance_factors[2]

    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coefficient_loss(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)