from sklearn.feature_extraction.image import reconstruct_from_patches_2d

def combine_patches(img_patches, target_shape, patch_size, step_size):
    z_step, y_step, x_step = step_size
    z_patch, y_patch, x_patch = patch_size

    output = []
    
    for slice_index, slice_patches in enumerate(img_patches):
        # alle possible z values from 0 to the z shape of the patch
        for z in range(z_patch):
            # initialize empty patch list to get all the patches belonging to 1 slice
            patches = []
            # if not last slice
            if not slice_index == (len(img_patches)-1):
                # keep in mind overlap, so if not last slice only work till z_step
                if not z > z_step:
                    # for all 3d patches 
                    for patch in slice_patches:
                        # get only the current z slice
                        patches.append(patch[z,:,:])
            # if it is the last slice, then do not bother any overlap
            else:
                for patch in slice_patches:
                    patches.append(patch[z,:,:])
            # using all patches of 1 slice reconstruct 2d image
            reconstructed_slice = reconstruct_from_patches_2d(patches,img_size)
            # append reconstructed slice to output list
            output.append(reconstructed_slice)

    return output

def pad_img(img, target_shape):
    padded_img = np.zeros(target_shape)

    padded_img[:img.shape[0],:img.shape[1],:img.shape[2]] = img

    return padded_img

def obtain_test_patches(model, img, patch_size, step_size, output_shape):
    z_step, y_step, x_step = step_size

    z_patch, y_patch, x_patch

    n_slices, height, width = img.shape

    # because the step size might cause a patch to go over boundry image
    # define target_shape for padding of image
    target_shape = (n_slices+(n_slices%z_step),height+(height%y_step),width+(width%x_step))

    # pad the image on one side with zeros to ensure that patches are extracted from entire image
    padded_img = pad_img(img,target_shape)
    
    test_segmentation_patches = []

    # go over slices but skip slices with z_step
    for z in range(0,target_shape[0],z_step):
        # collect all patches per z_step in one list
        # this should make it easier to reconstruct
        # 2d image compared to appending all patches
        # to one list
        slice_patches = []
        for y in range(0,target_shape[1],y_step):
            for x in range(0, target_shape[2], x_step):
                patch = padded_img[z:z_patch,y:y_patch,x:x_patch]
                segmentation = model.predict(patch, batch_size=1)
                slice_patches.append(segmentation)
        test_segmentation_patches.append(slice_patches)

    return test_segmentation_patches
