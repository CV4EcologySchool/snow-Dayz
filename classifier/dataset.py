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
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image
import pandas as pd


class CTDataset(Dataset):

    def __init__(self, labels, cfg, folder, split='train'):
        '''
            Constructor. Here, we collect and index the dataset inputs and
            labels.
        '''
        self.data_root = cfg['data_root']
        self.split = split
        self.folder = folder
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
        images_covered = set()      # all those images for which we have already assigned a label
        meta = meta[meta['Weather'] != 'Fog']
        meta = meta.drop_duplicates().reset_index() ## maybe I should keep the original indices??
        
        #images = meta['File']
        #labels = meta['Weather']
        for file, weather in zip(meta['File'], meta['Weather']):
            imgFileName = file
            labelIndex = meta[meta['Weather'] == weather].index ## do we need this?
            label = meta[meta['Weather'] == weather]
            imgID = labelIndex ## they are the same thing in my dataset because I didn't generate a imgID
            self.data.append([imgFileName, weather]) ## why label index and not label?
            ##images_covered.add(imgID) ## this is kind of irrelevant for my data

        #self.data.append([imgFileName, labelIndex])
        #images_covered.add(imgID)       # make sure image is only added once to dataset
     


        # meta = json.load(open(annoPath, 'r'))

        #images = dict([[idx, file] for idx, file in enumerate(meta['File'])]) ## do enumerate or do index
        #labels = dict([[idx, file] for idx, file in enumerate(meta['Weather'])])

        ######### image sort?? 

        # images = dict([[i['id'], i['file_name']] for i in meta['images']])          # image id to filename lookup
        # labels = dict([[c['id'], idx] for idx, c in enumerate(meta['categories'])]) # custom labelclass indices that start at zero
        
        # # since we're doing classification, we're just taking the first annotation per image and drop the rest

        #for anno in meta['Weather']:
         #     if (anno != "Fog"):
        
        #       imgID = anno['image_id'] ## i don't have image ID
        #       if imgID in images_covered:
        #           continue
            
            #     # append image-label tuple to data
            #     imgFileName = images[imgID]
            #     label = anno['category_id']
            #     labelIndex = labels[label]

    

    def __len__(self):
        '''
            Returns the length of the dataset.
        '''
        return len(self.data)

    def __shape__(self):
        return (self.data)
    
    def __getitem__(self, idx):
        '''
            Returns a single data point at given idx.
            Here's where we actually load the image.
        '''
        image_name, label = self.data[idx]              # see line 57 above where we added these two items to the self.data list

        # load image
        image_path = os.path.join(self.data_root, self.folder, image_name) ## should specify train folder and get image name 
        img = Image.open(image_path).convert('RGB')     # the ".convert" makes sure we always get three bands in Red, Green, Blue order

        # transform: see lines 31ff above where we define our transformations
        img_tensor = self.transform(img)

        return img_tensor, label