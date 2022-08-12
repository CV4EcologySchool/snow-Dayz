'''
    PyTorch dataset class for COCO-CT-formatted datasets. Note that you could
    use the official PyTorch MS-COCO wrappers:
    https://pytorch.org/vision/master/generated/torchvision.datasets.CocoDetection.html

    We just hack our way through the COCO JSON files here for demonstration
    purposes.

    See also the MS-COCO format on the official Web page:
    https://cocodataset.org/#format-data

    2022 Benjamin Kellenberger
'''

import os
import json
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image
import pandas as pd
import glob
from sequenceGenerator import sequenceGenerator
import random
from PIL import Image, ImageFile
import ipdb
import IPython

ImageFile.LOAD_TRUNCATED_IMAGES = True


class CTDataset(Dataset):

    # label class name to index ordinal mapping. This stays constant for each CTDataset instance.
    LABEL_CLASSES = {
        'None': 0,
        'Rain': 1,
        'Snow': 2,
        'Fog': 3,
        'Other': 4
    }
    LABEL_CLASSES_BINARY = {
        'None': 0,
        'Rain': 1,
        'Snow': 1,
        'Fog': 1,
        'Other': 1
    }

    def __init__(self, labels, cfg, split='train', folder='train'):
        '''
            Constructor. Here, we collect and index the dataset inputs and
            labels.
        '''
        self.data_root = cfg['data_root']
        self.split = split
        self.folder = folder
        self.sequenceType = cfg['sequenceType']
        self.transform = Compose([              # Transforms. Here's where we could add data augmentation (see Björn's lecture on August 11).
            Resize((cfg['image_size'])),        # For now, we just resize the images to the same dimensions...
            ToTensor()                          # ...and convert them to torch.Tensor.
        ])
        
        # index data into list
        self.data = []

        # load annotation file
        self.annoPath = os.path.join(
            self.data_root, labels) ############# should i set this as an input?? 

        meta = pd.read_csv(self.annoPath)
        meta = meta[meta['Weather'] != 'Fog'] ### could also drop 'other'
        meta = meta[meta['Weather'] != 'Other']
        meta = meta.drop_duplicates().reset_index() ## maybe I should keep the original indices??
        


        #images = meta['File']
        #labels = meta['Weather']

        ## add a check to make sure it exists in the folder of interest
        list_of_images = glob.glob(os.path.join(self.data_root,'train_resized/*')) ####UPDATED
        list_of_images = pd.Series(list_of_images)
        list_of_images = pd.DataFrame(list_of_images.str.split('/', expand=True)[5])

        if self.sequenceType == 'None':
            for file, weather in zip(meta['File'], meta['Weather']):
                #if random.uniform(0.0, 1.0) <= 0.99:
                    #continue
                    #(random.uniform(0.0, 1.0) <= 0.005) and
                if (sum(list_of_images == file) > 0): ## make sure there is the file in the train folder
                    imgFileName = file
                    if cfg['num_classes'] == 2:
                        self.data.append([imgFileName, self.LABEL_CLASSES_BINARY[weather]])
                    else: self.data.append([imgFileName, self.LABEL_CLASSES[weather]]) ## why label index and not label?

######################### sequences #################
        if self.sequenceType != 'None':
            for file, weather in zip(meta['File'], meta['Weather']):
                ## (random.uniform(0.0, 1.0) <= 0.001) and 
                if sum(list_of_images == file) > 0: ## make sure there is the file in the image (train) folder
                    imgFileName = file
                    before, file, after = sequenceGenerator(meta, file, sequenceType = self.sequenceType)
                    imgFileName = file 
                    if cfg['num_classes'] == 2:
                        imgFileName = file
                        self.data.append([[before, imgFileName, after], self.LABEL_CLASSES_BINARY[weather]])
                    else: self.data.append([[before, imgFileName, after], self.LABEL_CLASSES[weather]]) ## why label index and not label?


    def __len__(self):
        '''
            Returns the length of the dataset.
        '''
        return len(self.data)

    def __shape__(self):
        return (self.data)

    def __sequenceType__(self):
        return (self.sequenceType)

    def __getitem__(self, idx):
        '''
            Returns a single data point at given idx.
            Here's where we actually load the image.
        '''
        image_name, label = self.data[idx]              # see line 57 above where we added these two items to the self.data list

        # load image
        if self.sequenceType == 'None':
        #try:
            image_path = os.path.join(self.data_root, self.folder, image_name) ## should specify train folder and get image name 
            img = Image.open(image_path).convert('RGB')     # the ".convert" makes sure we always get three bands in Red, Green, Blue order
            #except: pass
            # transform: see lines 31ff above where we define our transformations
            img_tensor = self.transform(img)
            #print(img_tensor.shape)
        
    ######################################## sequences ##########################
        ##import IPython ## for testing : may need to install: pip install IPython
        ##IPython.embed() ## for testing
        #ipdb.set_trace()
       
        #IPython.embed()
        if self.sequenceType != 'None':
            before, image_name, after = image_name

            image_path1 = os.path.join(self.data_root, self.folder, before) ## should specify train folder and get image name 
            print(image_path1)
            print(self.data_root)
            print(self.folder)
            image_path2 = os.path.join(self.data_root, self.folder, image_name)
            image_path3 = os.path.join(self.data_root, self.folder, after) ####

            img1 = Image.open(image_path1).convert('L')     # the ".convert" makes sure we always get three bands in Red, Green, Blue order
            img2 = Image.open(image_path2).convert('L')
            img3 = Image.open(image_path3).convert('L')
            (print(img3.size))
            #except: pass

            # transform: see lines 31ff above where we define our transformations
            img_tensor1 = self.transform(img1)
            img_tensor2 = self.transform(img2)
            img_tensor3 = self.transform(img3)

            img_tensor = torch.cat([img_tensor1, img_tensor2,img_tensor3], dim = 0) ### 

############################################################################# kadjfldsf

        return img_tensor, label




  
