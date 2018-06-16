import numpy as np

from keras import backend as K
from keras.engine import Input, Model
from keras.layers import Conv3D, MaxPooling3D, Activation, Deconvolution3D, Cropping3D, UpSampling3D, BatchNormalization
from keras.optimizers import Adam
from keras.layers.merge import concatenate

from utils import dice_coefficient, dice_coefficient_loss

def unet(input_shape, features=32):
    # Downward path

    levels = list()
    inputs = Input(input_shape)
    # Block 1
    layer1 = Conv3D(features * (2 ** 0), (3, 3, 3), padding='valid', strides=1)(inputs)
    layer1 = BatchNormalization()(layer1)
    layer1 = Activation('relu')(layer1)
    layer2 = Conv3D(features * (2 ** 0), (3, 3, 3), padding='valid', strides=1)(layer1)
    layer2 = BatchNormalization()(layer2)
    layer2 = Activation('relu')(layer2)
    pool = MaxPooling3D(pool_size=(2, 2, 2))(layer2)
    levels.append([layer1, layer2, pool])
    # Block 2
    layer1 = Conv3D(features * (2 ** 1), (3, 3, 3), padding='valid', strides=1)(pool)
    layer1 = BatchNormalization()(layer1)
    layer1 = Activation('relu')(layer1)
    layer2 = Conv3D(features * (2 ** 1), (3, 3, 3), padding='valid', strides=1)(layer1)
    layer2 = BatchNormalization()(layer2)
    layer2 = Activation('relu')(layer2)
    pool = MaxPooling3D(pool_size=(2, 2, 2))(layer2)
    levels.append([layer1, layer2, pool])
    # Block 3
    layer1 = Conv3D(features * (2 ** 2), (3, 3, 3), padding='valid', strides=1)(pool)
    layer1 = BatchNormalization()(layer1)
    layer1 = Activation('relu')(layer1)
    layer2 = Conv3D(features * (2 ** 2), (3, 3, 3), padding='valid', strides=1)(layer1)
    layer2 = BatchNormalization()(layer2)
    layer2 = Activation('relu')(layer2)
    pool = MaxPooling3D(pool_size=(2, 2, 2))(layer2)
    levels.append([layer1, layer2, pool])
    # Block 4
    layer1 = Conv3D(features * (2 ** 3), (3, 3, 3), padding='valid', strides=1)(pool)
    layer1 = BatchNormalization()(layer1)
    layer1 = Activation('relu')(layer1)
    layer2 = Conv3D(features * (2 ** 3), (3, 3, 3), padding='valid', strides=1)(layer1)
    layer2 = BatchNormalization()(layer2)
    layer2 = Activation('relu')(layer2)
    levels.append([layer1, layer2])

    # Block 5
    layer0 = UpSampling3D(size=2)(layer2)
    # layer0 = Conv3D(32*(2**2), (2,2,2))(layer0)
    crop = levels[2][1]
    size = (crop._keras_shape[-4] - layer0._keras_shape[-4],
            crop._keras_shape[-3] - layer0._keras_shape[-3],
            crop._keras_shape[-2] - layer0._keras_shape[-2])
    size = ((int(np.floor(size[0] / 2)), int(np.ceil(size[0] / 2))),
            (int(np.floor(size[1] / 2)), int(np.ceil(size[1] / 2))),
            (int(np.floor(size[2] / 2)), int(np.ceil(size[2] / 2))))
    crop = Cropping3D(cropping=size)(crop)
    concatenate([layer0, crop], axis=4)
    layer1 = Conv3D(features * (2 ** 2), (3, 3, 3), padding='valid', strides=1)(layer0)
    layer1 = BatchNormalization()(layer1)
    layer1 = Activation('relu')(layer1)
    layer2 = Conv3D(features * (2 ** 2), (3, 3, 3), padding='valid', strides=1)(layer1)
    layer2 = BatchNormalization()(layer2)
    layer2 = Activation('relu')(layer2)

    # Upward path

    # Block 6
    layer0 = UpSampling3D(size=2)(layer2)
    # layer0 = Conv3D(32*(2**1), (2,2,2))(layer0)
    crop = levels[1][1]
    size = (crop._keras_shape[-4] - layer0._keras_shape[-4],
            crop._keras_shape[-3] - layer0._keras_shape[-3],
            crop._keras_shape[-2] - layer0._keras_shape[-2])
    size = ((int(np.floor(size[0] / 2)), int(np.ceil(size[0] / 2))),
            (int(np.floor(size[1] / 2)), int(np.ceil(size[1] / 2))),
            (int(np.floor(size[2] / 2)), int(np.ceil(size[2] / 2))))
    crop = Cropping3D(cropping=size)(crop)
    concatenate([layer0, crop], axis=4)
    layer1 = Conv3D(features * (2 ** 1), (3, 3, 3), padding='valid', strides=1)(layer0)
    layer1 = BatchNormalization()(layer1)
    layer1 = Activation('relu')(layer1)
    layer2 = Conv3D(features * (2 ** 1), (3, 3, 3), padding='valid', strides=1)(layer1)
    layer2 = BatchNormalization()(layer2)
    layer2 = Activation('relu')(layer2)
    # Block 7
    layer0 = UpSampling3D(size=2)(layer2)
    # layer0 = Conv3D(32*(2**0), (2,2,2))(layer0)
    crop = levels[0][1]
    size = (crop._keras_shape[-4] - layer0._keras_shape[-4],
            crop._keras_shape[-3] - layer0._keras_shape[-3],
            crop._keras_shape[-2] - layer0._keras_shape[-2])
    size = ((int(np.floor(size[0] / 2)), int(np.ceil(size[0] / 2))),
            (int(np.floor(size[1] / 2)), int(np.ceil(size[1] / 2))),
            (int(np.floor(size[2] / 2)), int(np.ceil(size[2] / 2))))
    crop = Cropping3D(cropping=size)(crop)
    concatenate([layer0, crop], axis=4)
    layer1 = Conv3D(features * (2 ** 0), (3, 3, 3), padding='valid', strides=1)(layer0)
    layer1 = BatchNormalization()(layer1)
    layer1 = Activation('relu')(layer1)
    layer2 = Conv3D(features * (2 ** 0), (3, 3, 3), padding='valid', strides=1)(layer1)
    layer2 = BatchNormalization()(layer2)
    layer2 = Activation('relu')(layer2)

    final = Conv3D(1, (1, 1, 1))(
        layer2)  # 1 output channel due to segmentation image being grayscale like input image patch
    final = Activation('softmax')(final)
    model = Model(inputs=inputs, outputs=final)

    model.compile(optimizer=Adam(lr=0.0001), loss=dice_coefficient_loss, metrics=[dice_coefficient])
    model.summary()
    return model