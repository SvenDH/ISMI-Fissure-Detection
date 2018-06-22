def shift_and_stitch(model, X, patch_size, output_shape, stride, n_classes):
    pad_z, pad_y, pad_x = [p//2 for p in patch_size]
    stride_z, stride_y, stride_x = stride
    n_batches, n_slices, height, width, n_channels = X.shape

    target_shape = n_batches, n_slices + 2*pad_z, height + 2*pad_y, width + 2*pad_x, n_channels
    Y = np.zeros(output_shape)
    for s in range(stride_z):
        for r in range(stride_y):
            for c in range(stride_x):
                X_in = np.zeros(target_shape)
                z_start, y_start, x_start = pad_z-s, pad_y-r, pad_x-c
                X_in[:, z_start:z_start+n_slices, y_start:y_start+height, x_start:x_start+width] = X
                Y_out = model.predict(X_in, batch_size=1)
                _, z, y, x, _ = Y_out.shape
                Y[:, s:z*stride_z:stride_z, r:y*stride_y:stride_y, c:x*stride_x:stride_x,:] = Y_out
    return Y[:,:n_slices,:height,:width,1]


output = shift_and_stitch(model, imgs, patch_size, output_shape, (stride,stride,stride), 3)
