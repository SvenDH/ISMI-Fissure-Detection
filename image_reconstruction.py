import numpy as np
from sklearn.feature_extraction.image import reconstruct_from_patches_2d

import os, random
import ntpath
import SimpleITK
from matplotlib import pyplot as plt
import pickle

test_segmentations = np.load("test_patch_segmentations.npy")

'''
So I assume there are patches with x and y set to 44 and 28 slices with values for 1 of each 5 labels.
The first Thing I need to do is create a list of patches with labels set to the max of the 5 values.
'''

patches = []

x_size = test_segmentations.shape[1]
y_size = test_segmentations.shape[2]
z_size = test_segmentations.shape[3]
for i in range(test_segmentations.shape[0]):
    patch = np.zeros((x_size, y_size, z_size))
    for z in range(z_size):
        for y in range(y_size):
            for x in range(x_size):
                patch[x,y,z] = np.where(test_segmentations[i,x,y,z,:] == np.max(test_segmentations[i,x,y,z,:]))[0][0]
    patches.append(patch)

'''
Next I define a function to get a slice as I am working with 'reconstruct_from_patches_2d'
and this will need to have a numpy array filled with patches from just one slice as this slice is 2d. 
'''

def get_slice(patches, slice_nr):
    slice_patches = []
    for patch in patches:
        slice_patches.append(patch[:,:,slice_nr])
    return np.array(slice_patches)

'''
As I am workign with this example input 'test_segmentation_patches' which has only 4 patches,I assume that these 4 patches are from 1 image
and that the patch step size was set to half the patch size of X and Y.
So 22. This means that the output image should have a shape of (66,66). If this is not the case or if more batches create one image, the following code needs to be updated.
But for reconstruct_from_patches_2d to work, it needs a numpy array filled with patches, preferably in order (but it keeps in mind overlap) and the original image size or target size.
'''

def combine_patches(patches, nr_slices, img_size):
    
    output_img = []
    
    for i in range(nr_slices):
        patches_2d = get_slice(patches,i)
        reconstructed_slice = reconstruct_from_patches_2d(patches_2d,img_size)
        output_img.append(reconstructed_slice)
    
    return np.array(output_img)

output_img = combine_patches(patches, 28, (66,66))
