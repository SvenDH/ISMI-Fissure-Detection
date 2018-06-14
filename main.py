import os, random
import ntpath
import SimpleITK
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from utils import PatchExtractor, BatchCreator #custom file for utilities
import callbacks #custom file for callbacks
import pickle
import time

from keras import backend as K
from keras.engine import Input, Model
from keras.layers import Conv3D, MaxPooling3D, Activation, Deconvolution3D, Cropping3D, UpSampling3D, BatchNormalization
from keras.optimizers import Adam


from keras.layers.merge import concatenate
from sklearn.model_selection import StratifiedShuffleSplit

K.set_image_data_format('channels_last')

# Loading data from pickle:
data = pd.read_pickle("NewDataLoading/train-data-filelist.pkl")

splitter = StratifiedShuffleSplit(1, test_size=0.1)

for train_index, test_index in splitter.split(data, data['label'].values):
    train_set = data.loc[train_index]
    validation_set = data.loc[test_index]
    
patch_indices = pickle.load(open("patch_indices.p","rb"))
x,y = patch_indices

patch_size = (132,132,116) # smallest possible patch size is (108,108,108)
output_shape = (44,44,28) # smallest possible output shape is (20,20,20)
patch_extractor = PatchExtractor(patch_size, output_shape)
batch_size = 16 # 16 is max due to gpu memory errors
batch_division = (np.ceil(batch_size/2),np.ceil(batch_size/2))

batch_creator = BatchCreator(patch_extractor, train_set, patch_indices, batch_division)
x,y = batch_creator.create_batch(batch_size) # batch testing
print('x: {}'.format(x.shape))
print('y: {}'.format(y.shape))
train_generator = batch_creator.get_generator(batch_size)
batch_creator = BatchCreator(patch_extractor, validation_set, patch_indices, batch_division)
validation_generator = batch_creator.get_generator(batch_size)

# Loss calculation for 3D U-net
def dice_coefficient(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coefficient_loss(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)


def create_network(input_shape, features=32):
    
    # Downward path
    
    levels = list()
    inputs = Input(input_shape)
    
    # Block 1
    layer1 = Conv3D(features*(2**0), (3,3,3), padding='valid', strides=1)(inputs)
    layer1 = BatchNormalization()(layer1)
    layer1 = Activation('relu')(layer1)
    layer2 = Conv3D(features*(2**0), (3,3,3), padding='valid', strides=1)(layer1)
    layer2 = BatchNormalization()(layer2)
    layer2 = Activation('relu')(layer2)
    pool = MaxPooling3D(pool_size=(2,2,2))(layer2)
    levels.append([layer1, layer2, pool])
    
    # Block 2
    layer1 = Conv3D(features*(2**1), (3,3,3), padding='valid', strides=1)(pool)
    layer1 = BatchNormalization()(layer1)
    layer1 = Activation('relu')(layer1)
    layer2 = Conv3D(features*(2**1), (3,3,3), padding='valid', strides=1)(layer1)
    layer2 = BatchNormalization()(layer2)
    layer2 = Activation('relu')(layer2)
    pool = MaxPooling3D(pool_size=(2,2,2))(layer2)
    levels.append([layer1, layer2, pool])
    
    # Block 3
    layer1 = Conv3D(features*(2**2), (3,3,3), padding='valid', strides=1)(pool)
    layer1 = BatchNormalization()(layer1)
    layer1 = Activation('relu')(layer1)
    layer2 = Conv3D(features*(2**2), (3,3,3), padding='valid', strides=1)(layer1)
    layer2 = BatchNormalization()(layer2)
    layer2 = Activation('relu')(layer2)
    pool = MaxPooling3D(pool_size=(2,2,2))(layer2)
    levels.append([layer1, layer2, pool])
    
    # Block 4
    layer1 = Conv3D(features*(2**3), (3,3,3), padding='valid', strides=1)(pool)
    layer1 = BatchNormalization()(layer1)
    layer1 = Activation('relu')(layer1)
    layer2 = Conv3D(features*(2**3), (3,3,3), padding='valid', strides=1)(layer1)
    layer2 = BatchNormalization()(layer2)
    layer2 = Activation('relu')(layer2)
    levels.append([layer1, layer2])

    # Block 5
    layer0 = UpSampling3D(size=2)(layer2)
    #layer0 = Conv3D(32*(2**2), (2,2,2))(layer0)
    crop = levels[2][1]
    size = (crop._keras_shape[-4] - layer0._keras_shape[-4],
            crop._keras_shape[-3] - layer0._keras_shape[-3], 
            crop._keras_shape[-2] - layer0._keras_shape[-2])
    size = ((int(np.floor(size[0]/2)),int(np.ceil(size[0]/2))),
            (int(np.floor(size[1]/2)),int(np.ceil(size[1]/2))),
            (int(np.floor(size[2]/2)),int(np.ceil(size[2]/2))))
    crop = Cropping3D(cropping=size)(crop)
    concatenate([layer0, crop],axis=4)
    layer1 = Conv3D(features*(2**2), (3,3,3), padding='valid', strides=1)(layer0)
    layer1 = BatchNormalization()(layer1)
    layer1 = Activation('relu')(layer1)
    layer2 = Conv3D(features*(2**2), (3,3,3), padding='valid', strides=1)(layer1)
    layer2 = BatchNormalization()(layer2)
    layer2 = Activation('relu')(layer2)
    
    # Upward path
    
    # Block 6
    layer0 = UpSampling3D(size=2)(layer2)
    #layer0 = Conv3D(32*(2**1), (2,2,2))(layer0)
    crop = levels[1][1]
    size = (crop._keras_shape[-4] - layer0._keras_shape[-4],
            crop._keras_shape[-3] - layer0._keras_shape[-3], 
            crop._keras_shape[-2] - layer0._keras_shape[-2])
    size = ((int(np.floor(size[0]/2)),int(np.ceil(size[0]/2))),
            (int(np.floor(size[1]/2)),int(np.ceil(size[1]/2))),
            (int(np.floor(size[2]/2)),int(np.ceil(size[2]/2))))
    crop = Cropping3D(cropping=size)(crop)
    concatenate([layer0, crop],axis=4)
    layer1 = Conv3D(features*(2**1), (3,3,3), padding='valid', strides=1)(layer0)
    layer1 = BatchNormalization()(layer1)
    layer1 = Activation('relu')(layer1)
    layer2 = Conv3D(features*(2**1), (3,3,3), padding='valid', strides=1)(layer1)
    layer2 = BatchNormalization()(layer2)
    layer2 = Activation('relu')(layer2)
    
    # Block 7
    layer0 = UpSampling3D(size=2)(layer2)
    #layer0 = Conv3D(32*(2**0), (2,2,2))(layer0)
    crop = levels[0][1]
    size = (crop._keras_shape[-4] - layer0._keras_shape[-4],
            crop._keras_shape[-3] - layer0._keras_shape[-3], 
            crop._keras_shape[-2] - layer0._keras_shape[-2])
    size = ((int(np.floor(size[0]/2)),int(np.ceil(size[0]/2))),
            (int(np.floor(size[1]/2)),int(np.ceil(size[1]/2))),
            (int(np.floor(size[2]/2)),int(np.ceil(size[2]/2))))
    crop = Cropping3D(cropping=size)(crop)
    concatenate([layer0, crop],axis=4)
    layer1 = Conv3D(features*(2**0), (3,3,3), padding='valid', strides=1)(layer0)
    layer1 = BatchNormalization()(layer1)
    layer1 = Activation('relu')(layer1)
    layer2 = Conv3D(features*(2**0), (3,3,3), padding='valid', strides=1)(layer1)
    layer2 = BatchNormalization()(layer2)
    layer2 = Activation('relu')(layer2)

    final = Conv3D(1, (1, 1, 1))(layer2) # 1 output channel due to segmentation image being grayscale like input image patch
    final = Activation('softmax')(final)
    model = Model(inputs=inputs, outputs=final)

    model.compile(optimizer=Adam(lr=0.00001), loss=dice_coefficient_loss, metrics=[dice_coefficient])
    model.summary()
    return model

if __name__ == "__main__":
    model = create_network(input_shape=[patch_size[0],patch_size[1],patch_size[2],1],features=8) 
    # nr of feature maps are bottleneck for gpu memory; 8 seems to be max; down from 5 mil parameters to less than 400,000

    #logger = Logger(data, patch_size, stride=88) #on training data instead of validation data
    timeNow = time.strftime("%e%m-%H%M%S")
    #tensorboard = TensorBoard(log_dir='./logs/'+timeNow, batch_size=batch_size, histogram_freq=1, embeddings_freq=0, write_images=True)
    modelcheck = callbacks.ModelCheckpoint("weights-"+str(timeNow)+".hdf5", monitor='val_loss', verbose=0, save_best_only=True, 
                                 save_weights_only=False, mode='auto', period=1)
    slacklogger = callbacks.SlackLogger()

    model.fit_generator(generator=train_generator,
                        validation_data=validation_generator,
                        steps_per_epoch=90,
                        epochs=10,
                        validation_steps=50,
                        callbacks=[modelcheck, slacklogger])

