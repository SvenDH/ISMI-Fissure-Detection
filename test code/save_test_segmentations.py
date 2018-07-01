import os, random
import ntpath
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from utils import shift_and_stitch, get_output_size, read_img, completeness, save_submission
import callbacks
from models import unet
import pickle
import time

from keras import backend as K

data = pd.read_pickle("test-data-filelist.pkl")
patch_size = (132,132,116)

output_size = get_output_size(patch_size)
model = unet(input_shape=[*patch_size,1])

model.load_weights('weights-2406-231334.hdf5')
model.summary()

mask_preds = []
for path in data['image'].values:
    image = read_img(path)
    output = shift_and_stitch(model, image, patch_size, output_size, output_size)
    mask_preds.append(output)
    
np.save('test_segmentations_3',mask_preds[:64])
np.save('test_segmentations_4',mask_preds[64:])