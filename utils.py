import os, random
import ntpath
import SimpleITK
from matplotlib import pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import pickle
import math

from keras import backend as K
from keras.engine import Input, Model
from keras.layers import Conv3D, MaxPooling3D, Activation, Cropping3D, UpSampling3D
from keras.optimizers import Adam

K.set_image_data_format("channels_first")

from keras.layers.merge import concatenate


class PatchExtractor:
    """
    PatchExtractor: class used to extract and possibly augment patches from images.
    """

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
            patch[:,:,:,0] = image[int(z - (c / 2)):int(z + (c / 2)), int(y - (h / 2)):int(y + (h / 2)),
                    int(x - (w / 2)):int(x + (w / 2))]
        except:
            print("Patch out of boundary, please make sure that the patch location is not out of boundary.")
        return patch


class BatchCreator:

    def __init__(self, patch_extractor, dataset, patch_indices, batch_division):
        self.patch_extractor = patch_extractor
        self.patch_size = self.patch_extractor.patch_size

        self.img_list = dataset['image'].values
        self.lbl_list = dataset['fissuremask'].values
        self.msk_list = dataset['lungmask'].values

        self.a_indices = dataset.index[dataset['label'] == "a"].tolist()
        self.b_indices = dataset.index[dataset['label'] == "b"].tolist()
        self.c_indices = dataset.index[dataset['label'] == "c"].tolist()

        self.img_indices = self.a_indices + self.b_indices + self.c_indices

        self.fc_indices = patch_indices[0]
        self.fi_indices = patch_indices[1]

        self.batch_division = batch_division

        self.examined_images = []

    def create_batch(self, batch_size):

        if len(self.examined_images) == len(self.a_indices + self.b_indices + self.c_indices):
            self.examined_images = []
            self.img_indices = self.a_indices + self.b_indices + self.c_indices

        img_index = self.pickImage()

        x_data, y_data, fissure_data = self.initializeOutputArrays(batch_size)

        fc_slices_dict = self.fc_indices[img_index]
        fi_slices_dict = self.fi_indices[img_index]

        img_array, lbl_array, msk_array = self.img2array(img_index)

        (fc_nr, fi_nr) = self.batch_division
        b_nr = batch_size - (fc_nr + fi_nr)

        if len(list(fi_slices_dict.keys())) == 0:
            fc_nr = fc_nr + fi_nr
            fi_nr = 0

        fc_grid, fc_grid_size = self.fissureGrid(fc_slices_dict)
        fi_grid, fi_grid_size = self.fissureGrid(fi_slices_dict)
        b_grid, b_grid_dict, b_grid_size = self.backgroundGrid(img_array.shape, int(b_nr / 4))

        background_counter = 0
        background_index = 0

        for i in range(batch_size):
            if i < fc_nr:
                z = fc_grid[i % fc_grid_size]
                (z, y, x) = self.getCoordinates(fc_slices_dict, z, img_array)
                x_data[i] = self.patch_extractor.get_patch(img_array, (z, y, x), False)
                y_data[i, 0, 0, 0, 2] = 1
                fissure_data[i] = self.patch_extractor.get_patch(lbl_array, (z, y, x), True)
            elif ((i >= fc_nr) and (i < (fc_nr + fi_nr))):
                z = fi_grid[i % fi_grid_size]
                (z, y, x) = self.getCoordinates(fi_slices_dict, z, img_array)
                x_data[i] = self.patch_extractor.get_patch(img_array, (z, y, x), False)
                y_data[i, 0, 0, 0, 1] = 1
                fissure_data[i] = self.patch_extractor.get_patch(lbl_array, (z, y, x), True)
            else:
                if background_counter == 4:
                    background_index += 1
                    background_counter = 0
                z = b_grid[background_index]
                grid = b_grid_dict[z][background_counter]
                (z, y, x) = self.getBackground(grid, msk_array, z)
                x_data[i] = self.patch_extractor.get_patch(img_array, (z, y, x), False)
                y_data[i, 0, 0, 0, 0] = 1
                fissure_data[i] = self.patch_extractor.get_patch(lbl_array, (z, y, x), True)
                background_counter += 1

        self.examined_images.append(img_index)

        return x_data, y_data

    def pickImage(self):
        index = np.random.randint(0, len(self.img_indices) - 1)
        img_index = self.img_indices[index]
        self.examined_images.append(img_index)
        self.img_indices = np.delete(self.img_indices, index)
        return img_index

    def initializeOutputArrays(self, batch_size):
        # patch array
        x_data = np.zeros((batch_size, *self.patch_extractor.patch_size,1))
        # label array (one-hot structure)
        y_data = np.zeros((batch_size, 1, 1, 1, 3))
        # fissure mask patch array
        fissure_data = np.zeros((batch_size, *self.patch_extractor.output_shape,1))

        return x_data, y_data, fissure_data

    def img2array(self, img_index):
        # compute numpy array from image
        img_path = self.img_list[img_index]
        img_array = readImg(img_path)

        # compute numpy array from fissure mask
        lbl_path = self.lbl_list[img_index]
        lbl_array = readImg(lbl_path)

        # compute numpy array from lung mask
        msk_path = self.msk_list[img_index]
        msk_array = readImg(msk_path)
        return img_array, lbl_array, msk_array

    def fissureGrid(self, slicesDict):
        z_size, _, _ = self.patch_extractor.patch_size
        slices = sorted(list(slicesDict.keys()))
        z_grid = list(self.chunks(slices, int(z_size * 1.5)))
        z_medians = [int(np.median(chunk)) for chunk in z_grid]
        grid_size = len(z_medians)
        return z_medians, grid_size

    def backgroundGrid(self, img_shape, b_nr):
        z_max, y_max, x_max = img_shape
        z_size, y_size, x_size = self.patch_extractor.patch_size
        slices = list(range(z_max))
        z_grid = list(self.chunks(slices, int(len(slices) / b_nr)))
        z_medians = [int(np.median(chunk)) for chunk in z_grid]
        grid_size = len(z_medians)
        z_grid_dict = {}
        for z_median in z_medians:
            grid1 = (0 + math.ceil(y_size / 2), int(y_max / 2) - 1, 0 + math.ceil(x_size / 2), int(x_max / 2) - 1)
            grid2 = (
            0 + math.ceil(y_size / 2), int(y_max / 2) - 1, int(x_max / 2), (x_max - 1) - (math.floor(y_size / 2)))
            grid3 = (
            int(y_max / 2), (y_max - 1) - (math.floor(y_size / 2)), 0 + math.ceil(x_size / 2), int(x_max / 2) - 1)
            grid4 = (int(y_max / 2), (y_max - 1) - (math.floor(y_size / 2)), int(x_max / 2),
                     (x_max - 1) - (math.floor(y_size / 2)))
            z_grid_dict[z_median] = (grid1, grid2, grid3, grid4)
        return z_medians, z_grid_dict, grid_size

    def chunks(self, l, n):
        return [l[i:i + n] for i in range(0, len(l), n)]

    def getCoordinates(self, slices_dict, z, img_array):
        coordinates = slices_dict[z]
        filtered_coordinates = [(y, x) for (y, x) in coordinates if self.inBoundary((z, y, x), img_array.shape)]
        if len(filtered_coordinates) == 0:
            print("Error: no x,y coordinates found that are in boundary of the image taking the patch size in mind")
        random_coords_index = np.random.choice(len(filtered_coordinates))
        y, x = filtered_coordinates[random_coords_index]
        return (z, y, x)

    def getBackground(self, grid, msk_array, z):
        (y_min, y_max, x_min, x_max) = grid
        y_indices, x_indices = np.where(msk_array[z, :, :] == 3)
        coords = self.getCoords(z, y_indices, x_indices, msk_array)
        i = np.random.randint(len(coords) - 1)
        (y, x) = coords[i]
        return (z, y, x)

    def getCoords(self, z, y_indices, x_indices, msk_array):
        coords = []
        for i, y in enumerate(y_indices):
            x = x_indices[i]
            coord = (y, x)
            if self.inBoundary((z, y, x), msk_array.shape):
                coords.append(coord)
        return coords

    def inBoundary(self, location, img_shape):
        _, y_size, x_size = img_shape
        _, y_patch, x_patch = self.patch_extractor.patch_size

        y_min = math.ceil(0 + (y_patch / 2))
        y_max = math.floor(y_size - (y_patch / 2))

        x_min = math.ceil(0 + (x_patch / 2))
        x_max = math.ceil(x_size - (x_patch / 2))

        _, y, x = location

        if (y <= y_max and y >= y_min) and (x <= x_max and x >= x_min):
            return True
        else:
            return False

    def checkBackground(self, location, lbl_array, msk_array):
        z, y, x = location
        if (lbl_array[z, y, x] == 0) and (msk_array[z, y, x] == 3):
            return True
        else:
            return False

    def get_generator(self, batch_size):
        '''returns a generator that will yield batches infinitely'''
        while True:
            yield self.create_batch(batch_size)


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


def dice_coefficient(y_true, y_pred, smooth=1.):
    # Loss calculation for 3D U-net
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coefficient_loss(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)


def unet(input_shape):
    levels = list()
    inputs = Input(input_shape)
    # Block 1
    layer1 = Conv3D(32 * (2 ** 0), (3, 3, 3), padding='valid', strides=1)(inputs)
    layer1 = Activation('relu')(layer1)
    layer2 = Conv3D(32 * (2 ** 0), (3, 3, 3), padding='valid', strides=1)(layer1)
    layer2 = Activation('relu')(layer2)
    pool = MaxPooling3D(pool_size=(2, 2, 2))(layer2)
    levels.append([layer1, layer2, pool])
    # Block 2
    layer1 = Conv3D(32 * (2 ** 1), (3, 3, 3), padding='valid', strides=1)(pool)
    layer1 = Activation('relu')(layer1)
    layer2 = Conv3D(32 * (2 ** 1), (3, 3, 3), padding='valid', strides=1)(layer1)
    layer2 = Activation('relu')(layer2)
    pool = MaxPooling3D(pool_size=(2, 2, 2))(layer2)
    levels.append([layer1, layer2, pool])
    # Block 3
    layer1 = Conv3D(32 * (2 ** 2), (3, 3, 3), padding='valid', strides=1)(pool)
    layer1 = Activation('relu')(layer1)
    layer2 = Conv3D(32 * (2 ** 2), (3, 3, 3), padding='valid', strides=1)(layer1)
    layer2 = Activation('relu')(layer2)
    pool = MaxPooling3D(pool_size=(2, 2, 2))(layer2)
    levels.append([layer1, layer2, pool])
    # Block 4
    layer1 = Conv3D(32 * (2 ** 3), (3, 3, 3), padding='valid', strides=1)(pool)
    layer1 = Activation('relu')(layer1)
    layer2 = Conv3D(32 * (2 ** 3), (3, 3, 3), padding='valid', strides=1)(layer1)
    layer2 = Activation('relu')(layer2)
    levels.append([layer1, layer2])

    # Block 5
    layer0 = UpSampling3D(size=2)(layer2)
    # layer0 = Conv3D(32*(2**2), (2,2,2))(layer0)
    crop = levels[2][1]
    size = (crop._keras_shape[-3] - layer0._keras_shape[-3],
            crop._keras_shape[-2] - layer0._keras_shape[-2],
            crop._keras_shape[-1] - layer0._keras_shape[-1])
    size = ((int(np.floor(size[0] / 2)), int(np.ceil(size[0] / 2))),
            (int(np.floor(size[1] / 2)), int(np.ceil(size[1] / 2))),
            (int(np.floor(size[2] / 2)), int(np.ceil(size[2] / 2))))
    crop = Cropping3D(cropping=size)(crop)
    concatenate([layer0, crop], axis=1)
    layer1 = Conv3D(32 * (2 ** 2), (3, 3, 3), padding='valid', strides=1)(layer0)
    layer1 = Activation('relu')(layer1)
    layer2 = Conv3D(32 * (2 ** 2), (3, 3, 3), padding='valid', strides=1)(layer1)
    layer2 = Activation('relu')(layer2)
    # Block 6
    layer0 = UpSampling3D(size=2)(layer2)
    # layer0 = Conv3D(32*(2**1), (2,2,2))(layer0)
    crop = levels[1][1]
    size = (crop._keras_shape[-3] - layer0._keras_shape[-3],
            crop._keras_shape[-2] - layer0._keras_shape[-2],
            crop._keras_shape[-1] - layer0._keras_shape[-1])
    size = ((int(np.floor(size[0] / 2)), int(np.ceil(size[0] / 2))),
            (int(np.floor(size[1] / 2)), int(np.ceil(size[1] / 2))),
            (int(np.floor(size[2] / 2)), int(np.ceil(size[2] / 2))))
    crop = Cropping3D(cropping=size)(crop)
    concatenate([layer0, crop], axis=1)
    layer1 = Conv3D(32 * (2 ** 1), (3, 3, 3), padding='valid', strides=1)(layer0)
    layer1 = Activation('relu')(layer1)
    layer2 = Conv3D(32 * (2 ** 1), (3, 3, 3), padding='valid', strides=1)(layer1)
    layer2 = Activation('relu')(layer2)
    # Block 7
    layer0 = UpSampling3D(size=2)(layer2)
    # layer0 = Conv3D(32*(2**0), (2,2,2))(layer0)
    crop = levels[0][1]
    size = (crop._keras_shape[-3] - layer0._keras_shape[-3],
            crop._keras_shape[-2] - layer0._keras_shape[-2],
            crop._keras_shape[-1] - layer0._keras_shape[-1])
    size = ((int(np.floor(size[0] / 2)), int(np.ceil(size[0] / 2))),
            (int(np.floor(size[1] / 2)), int(np.ceil(size[1] / 2))),
            (int(np.floor(size[2] / 2)), int(np.ceil(size[2] / 2))))
    crop = Cropping3D(cropping=size)(crop)
    concatenate([layer0, crop], axis=1)
    layer1 = Conv3D(32 * (2 ** 0), (3, 3, 3), padding='valid', strides=1)(layer0)
    layer1 = Activation('relu')(layer1)
    layer2 = Conv3D(32 * (2 ** 0), (3, 3, 3), padding='valid', strides=1)(layer1)
    layer2 = Activation('relu')(layer2)

    final = Conv3D(3, (1, 1, 1))(layer2)
    final = Activation('softmax')(final)
    model = Model(inputs=inputs, outputs=final)

    model.compile(optimizer=Adam(lr=0.00001), loss=dice_coefficient_loss, metrics=[dice_coefficient])
    model.summary()
    return model


