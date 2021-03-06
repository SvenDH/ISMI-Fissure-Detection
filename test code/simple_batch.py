class SimpleBatchCreator:
    
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
        
    def create_batch(self,batch_size):
        
        if len(self.examined_images) == len(self.img_indices):
            self.examined_images = []
            
        img_index = self.pickImage()
        
        x_data, y_data, fissure_data = self.initializeOutputArrays(batch_size)
        
        fc_slices_dict = self.fc_indices[img_index]
        fi_slices_dict = self.fi_indices[img_index]
        
        img_array, lbl_array, msk_array = self.img2array(img_index)
        
        minima = self.getMinima(self.patch_extractor.patch_size,img_array.shape)
        
        (fc_nr,fi_nr) = self.batch_division
        b_nr = batch_size-(fc_nr+fi_nr)
        
        if len(list(fi_slices_dict.keys())) == 0:
            fc_nr = fc_nr + fi_nr
            fi_nr = 0
            
        for i in range(batch_size):
            if i < fc_nr:
                (z,y,x) = self.pickCoordinate(minima)
                print(minima)
                print((z,y,x))
                print("\n")
                x_data[i] = self.patch_extractor.get_patch(img_array,(z,y,x),False)
                y_data[i,0,0,0,2] = 1
                fissure_data[i] = self.patch_extractor.get_patch(lbl_array,(z,y,x),True)
            elif ((i >= fc_nr) and (i < (fc_nr + fi_nr))):
                (z,y,x) = self.pickCoordinate(minima)
                print(minima)
                print((z,y,x))
                print("\n")
                x_data[i] = self.patch_extractor.get_patch(img_array,(z,y,x),False)
                y_data[i,0,0,0,1] = 1
                fissure_data[i] = self.patch_extractor.get_patch(lbl_array,(z,y,x),True)
            else:
                (z,y,x) = self.pickCoordinate(minima)
                print(minima)
                print((z,y,x))
                print("\n")
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
    
    def pickCoordinate(self, minima):
        (z_minimum,z_maximum), (y_minimum,y_maximum), (x_minimum,x_maximum) = minima
        z = random.choice(range(z_minimum,z_maximum))
        y = random.choice(range(y_minimum,y_maximum))
        x = random.choice(range(x_minimum,x_maximum))
        return (z,y,x)
        
        
    def get_generator(self, batch_size):
        '''returns a generator that will yield batches infinitely'''
        while True:
            yield self.create_batch(batch_size) 
