class BatchCreator:
    
    def __init__(self,patch_extractor,dataset,patch_indices,batch_division):
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
            
        img_index = self.pickImage()
        
        x_data, y_data, fissure_data = self.initializeOutputArrays(batch_size)
        
        fc_slices_dict = self.fc_indices[img_index]
        fi_slices_dict = self.fi_indices[img_index]
        
        img_array, lbl_array, msk_array = self.img2array(img_index)

        # get the minima for z, y, x to check if patches are in boundary
        minima = self.getMinima(self.patch_extractor.patch_size,img_array.shape)

        # get list of potential fissure complete and incomplete coords and the number of patches to generate
        # for fissure complete, incomplete and background
        fc_coords, fi_coords, fc_nr, fi_nr, b_nr = self.checkEmpty(minima,fc_slices_dict,fi_slices_dict)

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
        img_path = self.img_list[img_index]
        img_array = readImg(img_path)
        
        # compute numpy array from fissure mask
        lbl_path = self.lbl_list[img_index]
        lbl_array = readImg(lbl_path)
        
        # compute numpy array from lung mask
        msk_path = self.msk_list[img_index]
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
    
    def checkEmpty(self,minima,fc_slices_dict,fi_slices_dict):
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
