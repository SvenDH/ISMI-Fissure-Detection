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
    
    def __init__(self,patch_extractor,dataset,patch_indices,batch_division):
        self.patch_extractor = patch_extractor
        self.patch_size = self.patch_extractor.patch_size

        self.dataset = dataset
        
        self.img_list = dataset['image'].values
        self.lbl_list = dataset['fissuremask'].values
        self.msk_list = dataset['lungmask'].values
        
        self.a_indices = dataset.index[dataset['label'] == "a"].tolist()
        self.b_indices = dataset.index[dataset['label'] == "b"].tolist()
        self.c_indices = dataset.index[dataset['label'] == "c"].tolist()
        
        self.img_indices = self.dataset.index.values.tolist()
        
        self.fc_indices = patch_indices[0]
        self.fi_indices = patch_indices[1]
        
        self.batch_division = batch_division
        
        self.examined_images = []
        
    def create_batch(self, batch_size):
        
        if len(self.examined_images) == len(self.img_indices):
            self.examined_images = []
            
        img_index = self.pickImage()
        
        x_data, y_data, fissure_data = self.initializeOutputArrays(batch_size)
        
        fc_slices_dict = self.fc_indices[img_index]
        fi_slices_dict = self.fi_indices[img_index]
        
        img_array, lbl_array, msk_array = self.img2array(img_index)

        # get the minima for z, y, x to check if patches are in boundary
        minima = self.getMinima(self.patch_extractor.patch_size,img_array.shape)

        # get list of potential fissure complete and incomplete coords and the number of patches to generate
        # for fissure complete, incomplete and background
        fc_coords, fi_coords, fc_nr, fi_nr, b_nr = self.checkEmpty(batch_size,minima,fc_slices_dict,fi_slices_dict)

        # get list of potential background coords
        b_coords = self.getBackground(msk_array,minima)
        
        for i in range(batch_size):
            if i < fc_nr:
                (z,y,x) = random.choice(fc_coords)
                x_data[i] = self.patch_extractor.get_patch(img_array,(z,y,x),False)
                y_data[i,0,0,0,2] = 1
                fissure_data[i] = self.patch_extractor.get_patch(lbl_array,(z,y,x),True)
            elif ((i >= fc_nr) and (i < (fc_nr + fi_nr))):
                (z,y,x) = random.choice(fi_coords)
                x_data[i] = self.patch_extractor.get_patch(img_array,(z,y,x),False)
                y_data[i,0,0,0,1] = 1
                fissure_data[i] = self.patch_extractor.get_patch(lbl_array,(z,y,x),True)
            else:
                (z,y,x) = random.choice(b_coords)
                x_data[i] = self.patch_extractor.get_patch(img_array,(z,y,x),False)
                y_data[i,0,0,0,0] = 1
                fissure_data[i] = self.patch_extractor.get_patch(lbl_array,(z,y,x),True)
        
        return x_data, fissure_data
        
    def pickImage(self):
        img_index = self.img_indices[len(self.examined_images)]
        self.examined_images.append(img_index)
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
        img_path = self.dataset.iloc[self.img_indices.index(img_index)]['image']
        img_array = readImg(img_path)
        
        # compute numpy array from fissure mask
        lbl_path = self.dataset.iloc[self.img_indices.index(img_index)]['fissuremask']
        lbl_array = readImg(lbl_path)
        
        # compute numpy array from lung mask
        msk_path = self.dataset.iloc[self.img_indices.index(img_index)]['lungmask']
        msk_array = readImg(msk_path)
        return img_array, lbl_array, msk_array
    
    def getMinima(self,patch_size,img_shape):
        z_size, y_size, x_size = self.patch_extractor.patch_size
        z_max, y_max, x_max = img_shape
        z_minimum = round(0+(z_size/2))
        z_maximum = round(z_max-(z_size/2))
        y_minimum = round(0+(y_size/2))
        y_maximum = round(y_max-(y_size/2))
        x_minimum = round(0+(x_size/2))
        x_maximum = round(x_max-(x_size/2))
        minima = ((z_minimum,z_maximum), (y_minimum,y_maximum), (x_minimum, x_maximum))
        return minima
    
    def filterCoords(self,minima,slices_dict):
        (z_minimum,z_maximum), (y_minimum,y_maximum), (x_minimum,x_maximum) = minima
        slices = slices_dict.keys()
        filtered_slices = [z for z in slices if z > z_minimum and z < z_maximum]
        filtered_coordinates = []
        for z in filtered_slices:
            coords = slices_dict[z]
            for (y,x) in coords:
                if (y > y_minimum) and (y < y_maximum):
                    if (x > x_minimum) and (x < x_maximum):
                        filtered_coordinates.append((z,y,x))
        return filtered_coordinates
    
    def getBackground(self,msk_array,minima):
        (z_minimum,z_maximum), (y_minimum,y_maximum), (x_minimum,x_maximum) = minima
        filtered_coordinates = []
        for z in range(z_minimum,z_maximum):
            y_indices, x_indices = np.where(msk_array[z,y_minimum:y_maximum,x_minimum:x_maximum] == 3)
            for i, y in enumerate(y_indices):
                x = x_indices[i]
                filtered_coordinates.append((z,y+y_minimum,x+x_minimum))
        if len(filtered_coordinates) == 0:
            print("Error: could not find background patch for slice %s, with y_minimum set to %s, y_maximum set to %s, x_minimum set to %s and x_maximum set to %s"%(z,y_minimum,y_maximum,x_minimum,x_maximum))
        return filtered_coordinates
    
    def checkEmpty(self,batch_size,minima,fc_slices_dict,fi_slices_dict):
        fc_coords = self.filterCoords(minima,fc_slices_dict)
        fi_coords = []
        
        (fc_nr,fi_nr) = self.batch_division
        b_nr = batch_size-(fc_nr+fi_nr)
        
        # if no 'fissure complete' coords are found inside boundaries
        if len(fc_coords) == 0:
            # if there are at least some 'fissure incomplete' slices
            if not len(list(fi_slices_dict.keys())) == 0:
                # make sure to skip fissure complete generation
                # by setting number of fissure complete patches to zero
                fi_nr = fc_nr + fi_nr
                fc_nr = 0
                fi_coords = self.filterCoords(minima,fi_slices_dict)
                # if there are no 'fissure incomplete' coords inside boundaries
                # make sure to skip fissure incomplete generation
                if len(fi_coords) == 0:
                    fi_nr = 0
            # else skip both fissure complete and incomplete generation
            else:
                fc_nr = fi_nr = 0
        else:
            # if there are no 'fissure incomplete' slices
            if len(list(fi_slices_dict.keys())) == 0:
                #skip fissure incomplete generation
                # by setting number of fissure incomplete patches to zero
                fc_nr = fc_nr + fi_nr
                fi_nr = 0
            # if there are slices with fissure incomplete parts
            else:
                # find coords that are in boundary
                fi_coords = self.filterCoords(minima,fi_slices_dict)
                # if none are found, skip fissure incomplete patch generation
                if len(fi_coords) == 0:
                    fc_nr = fc_nr + fi_nr
                    fi_nr = 0
                    
        return fc_coords, fi_coords, fc_nr, fi_nr, b_nr
            
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


