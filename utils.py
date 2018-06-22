import os, random
import ntpath
import SimpleITK
from matplotlib import pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
from itertools import product

import keras
from keras.utils import to_categorical
from keras import backend as K


class BatchGenerator(keras.utils.Sequence):
    """Generator for keras that provides patches of images with corresponding fissure masks."""

    def __init__(self, data, patch_size, step=(10, 10, 10), batch_size=8, sampling=None, test=False):
        """Load all images and generate a list of indices for patch locations."""
        self.patch_size, self.data, self.batch_size, self.sampling = patch_size, data, batch_size, sampling
        self.output_size = get_output_size(patch_size)
        self.images = [read_img(path) for path in data['image'].values]
        self.test = test
        if not self.test:
            self.fissure_masks = [read_img(path) for path in data['fissuremask'].values]
        # Generate coordinates at every step leaving a border of half the patch size
        indices, labels = [], []
        zh, yh, xh = int(patch_size[0]/2), int(patch_size[1]/2), int(patch_size[2]/2)
        for i, image in enumerate(self.images):
            c, h, w = image.shape
            for z, y, x in product(range(zh, c-zh, step[0]), range(yh, h-yh, step[1]), range(xh, w-xh, step[2])):
                indices.append([i, z, y, x])  # The label of a patch is the highest value occurring in the fissure mask
                if not self.test:
                    labels.append(np.amax(self.get_patch((z, y, x), self.fissure_masks[i], self.output_size)))
        self.indices, self.labels = np.array(indices), np.array(labels)
        self.samples = self.indices
        self.on_epoch_end()

    def __getitem__(self, index):
        """Get a new batch of input images and corresponding fracture masks."""
        idxs = self.samples[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = np.empty((self.batch_size, *self.patch_size, 1)), np.empty((self.batch_size, *self.output_size, 1))
        for i, idx in enumerate(idxs):
            X[i, ] = self.get_patch(idx[1:], self.images[idx[0]], self.patch_size)[:, :, :, np.newaxis]
            if not self.test:
                y[i, ] = self.get_patch(idx[1:], self.fissure_masks[idx[0]], self.output_size)[:, :, :, np.newaxis]
                return X, to_categorical(y, 5)
            else:
                return X

    def __len__(self):
        """Amount of batches per epoch."""
        return int(np.floor(len(self.samples) / self.batch_size))

    def on_epoch_end(self):
        """If sample function is provided apply sampling function to indices and labels to get new samples."""
        if self.sampling:
            self.samples, _ = self.sampling(self.indices, self.labels)

    @staticmethod
    def get_patch(location, image, patch_size):
        """Get part of image array centered on location with patch_size as size."""
        z, y, x = location
        c, h, w = patch_size
        return image[int(z-(c/2)):int(z+(c/2)), int(y-(h/2)):int(y+(h/2)), int(x-(w/2)):int(x+(w/2))]


def batch_balance_sampling(X, y):
    ids_per_label = {}
    for c in np.unique(y):
        ids_per_label[c] = np.where(y==c)
        np.random.shuffle(ids_per_label)
    print(ids_per_label)

    return X, y


def get_output_size(input_size):
    """Calculation to get the output shape of U-Net given the input shape."""
    output_size = (((((((np.floor((np.floor((np.floor((np.array(input_size)-4)/2)-4)/2)-4)/2)-4)*2)-4)*2)-4)*2)-4)
    return tuple(output_size.astype(int))


class PatchExtractor:
    """PatchExtractor: class used to extract and possibly augment patches from images."""
    def __init__(self, patch_size, output_shape):
        self.patch_size = patch_size
        self.output_shape = output_shape

    def get_patch(self, image, location, isOutput):
        """
        image: a numpy array representing the input image
        location: a tuple with an z, y, and x coordinate

        return a 3D patch from the image at 'location', representing the center of the patch
        """

        z, y, x = location
        c, h, w = self.patch_size
        patch = np.zeros(self.patch_size + (1,))
        if isOutput:
            patch = np.zeros(self.output_shape + (1,))
            c, h, w = self.output_shape
        try:
            # patch = image[int(z-(c/2)):int(z+(c/2)),int(y-(h/2)):int(y+(h/2)),int(x-(w/2)):int(x+(w/2))]
            patch[:, :, :, 0] = image[int(z - (c / 2)):int(z + (c / 2)), int(y - (h / 2)):int(y + (h / 2)),
                                int(x - (w / 2)):int(x + (w / 2))]
        except:
            print("Patch out of boundary, please make sure that the patch location is not out of boundary.")
        return patch


class BatchCreator:

    def __init__(self, patch_extractor, dataset, sampleLocations, batch_division, nr_samples):
        self.patch_extractor = patch_extractor
        self.patch_size = self.patch_extractor.patch_size

        self.dataset = dataset

        self.img_list = dataset['image'].values
        self.lbl_list = dataset['fissuremask'].values
        self.msk_list = dataset['lungmask'].values

        self.img_indices = dataset.index.values.tolist()

        self.bSamples, self.fcSamples, self.fiSamples = sampleLocations

        self.batch_division = batch_division

        self.nr_samples = nr_samples

        self.counter = 0

        self.examined_images = []

    def create_batch(self, batch_size, img_index):

        if len(self.examined_images) == len(self.img_indices):
            self.examined_images = []

        x_data, y_data, fissure_data = self.initializeOutputArrays(batch_size)

        b_samples = self.bSamples[img_index]
        fc_samples = self.fcSamples[img_index]
        fi_samples = self.fiSamples[img_index]

        img_array, lbl_array, msk_array = self.img2array(img_index)

        fc_nr, fi_nr = self.checkEmpty(b_samples, fc_samples, fi_samples, self.batch_division)

        for i in range(batch_size):
            if i < fc_nr:
                (z, y, x) = random.choice(fc_samples)
                x_data[i] = self.patch_extractor.get_patch(img_array, (z, y, x), False)
                y_data[i, 0, 0, 0, 2] = 1
                fissure_data[i] = self.patch_extractor.get_patch(lbl_array, (z, y, x), True)
            elif ((i >= fc_nr) and (i < (fc_nr + fi_nr))):
                (z, y, x) = random.choice(fi_samples)
                x_data[i] = self.patch_extractor.get_patch(img_array, (z, y, x), False)
                y_data[i, 0, 0, 0, 1] = 1
                fissure_data[i] = self.patch_extractor.get_patch(lbl_array, (z, y, x), True)
            else:
                (z, y, x) = random.choice(b_samples)
                x_data[i] = self.patch_extractor.get_patch(img_array, (z, y, x), False)
                y_data[i, 0, 0, 0, 0] = 1
                fissure_data[i] = self.patch_extractor.get_patch(lbl_array, (z, y, x), True)

        self.updateCounter(batch_size)

        self.examined_images.append(img_index)

        return x_data, fissure_data

    def pickImage(self):
        img_index = self.img_indices[len(self.examined_images)]
        self.examined_images.append(img_index)
        return img_index

    def initializeOutputArrays(self, batch_size):
        # patch array
        x_data = np.zeros((batch_size, *self.patch_extractor.patch_size, 1))
        # label array (one-hot structure)
        y_data = np.zeros((batch_size, 1, 1, 1, 3))
        # fissure mask patch array
        fissure_data = np.zeros((batch_size, *self.patch_extractor.output_shape, 1))

        return x_data, y_data, fissure_data

    def img2array(self, img_index):
        # compute numpy array from image
        img_path = self.dataset.iloc[self.img_indices.index(img_index)]['image']
        img_array = read_img(img_path)

        # compute numpy array from fissure mask
        lbl_path = self.dataset.iloc[self.img_indices.index(img_index)]['fissuremask']
        lbl_array = read_img(lbl_path)

        # compute numpy array from lung mask
        msk_path = self.dataset.iloc[self.img_indices.index(img_index)]['lungmask']
        msk_array = read_img(msk_path)
        return img_array, lbl_array, msk_array

    def checkEmpty(self, b_samples, fc_samples, fi_samples, batch_division):
        fc_nr, fi_nr = batch_division

        if len(fc_samples) == 0:
            if len(fi_samples) == 0:
                fc_nr = 0
                fi_nr = 0
            else:
                fi_nr = fi_nr + fc_nr
                fc_nr = 0
        else:
            if len(fi_samples) == 0:
                fc_nr = fc_nr + fi_nr
                fi_nr = 0

        return fc_nr, fi_nr

    def updateCounter(self, batch_size):
        self.counter += batch_size
        if self.counter > self.nr_samples:
            self.counter = 0

    def counterReset(self):
        if self.counter == 0:
            return True
        else:
            return False

    def get_generator(self, batch_size):
        """returns a generator that will yield batches infinitely"""
        img_index = self.pickImage()
        while True:
            if self.counterReset:
                img_index = self.pickImage()
            print(img_index)
            yield self.create_batch(batch_size, img_index)


def read_img(img_path):
    """Read image from file location and returns it as an array."""
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
    """Load training data from a folder"""
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

def load_test_set(folder):
    file_list = load_unique_image_names(folder)
    
    test_set = []
    for file in file_list:
        file_path = folder + '/' + file
        image = lung_mask = None
        try:
            image = file_path + '.mhd'
            lung_mask = file_path + '_lm.mhd'
            test_set.append({'name': file, 
                             'image': image,
                             'lungmask': lung_mask})
        except :
            print("Error reading file: " + file)

    return test_set

def get_exact_csv_set(folder, label):
    return pd.read_csv(folder + '/LUT-' + label + '.csv')

    #function taken from https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/python/keras/_impl/keras/backend.py
    #adapted to weigh categories


"""
def weighted_categorical_crossentropy(target, output):
    #decide on weights
    bglabel = 0
    cflabel = 2
    iflabel = 4
    
    weight_values_per_class = [0.001, 0.4, 0.6]
    #weight_values_per_class = compute_loss_weights(target)
    weights = (target == bglabel) * weight_values_per_class[0] 
    + (target == cflabel) * weight_values_per_class[1] 
    + (target == iflabel) * weight_values_per_class[2] 
    
    # scale preds so that the class probas of each sample sum to 1
    output = output / np.sum(output)
    # manual computation of crossentropy
    
    #epsilon_ = _to_tensor(epsilon(), output.dtype.base_dtype)
    output = K.clip(output, K.epsilon(), 1. - K.epsilon())
    
    return -K.sum(target * K.log(output) * weights)
"""

"""
A weighted version of categorical_crossentropy for keras (2.0.6). This lets you apply a weight to unbalanced classes.
@url: https://gist.github.com/wassname/ce364fddfc8a025bfab4348cf5de852d
@author: wassname
"""
def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes

    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """

    weights = K.variable(weights)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss

    return loss
	
	
def dice_coefficient(y_true, y_pred, smooth=1.):
    """Loss calculation for 3D U-net"""

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coefficient_loss(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)
