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
from torchvision.transforms import RandomVerticalFlip, RandomVerticalFlip, RandomGrayscale, ColorJitter #RandomResizedCrop #RandomErasing, RandomCrop
from PIL import Image
import pandas as pd
import glob
import random
from PIL import Image, ImageFile
import ipdb
import IPython
import torchvision.transforms as transforms

class RandomApplyTransform:
    def __init__(self, transform, p=0.5):
        self.transform = transform
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return self.transform(img)
        return img


ImageFile.LOAD_TRUNCATED_IMAGES = True


def train_test_split(cfg, images_path, labels):
    df_data = pd.read_csv(labels)
    valid_cameras = ['175', '54', '484', '1376', '486', '1175', '3036', '1746', '970', '1142', '1185', '688', 
                     '2027', '638', '870', '317', '1184', '953', 
                     '2029', '518', '1495', '850', '1613', '842', '1263', '656', '1150', '1192', '1121', '1438']


    training_samples = df_data[~df_data['cameraID'].isin(valid_cameras)] 
    valid_samples = df_data[df_data['cameraID'].isin(valid_cameras)] 

    ##### only images that exist
    all_images = glob.glob(images_path + ('/*'))
    filenames = [item.split('/')[-1] for item in all_images]
    valid_samples = valid_samples[valid_samples['filename'].isin(filenames)]
    training_samples = training_samples[training_samples['filename'].isin(filenames)]
    
    if not os.path.exists(f"{cfg['output_path']}/{cfg['exp_name']}"):
            os.makedirs(f"{cfg['output_path']}/{cfg['exp_name']}", exist_ok=True)
    training_samples.to_csv(f"{cfg['output_path']}/{cfg['exp_name']}/training_samples.csv")
    valid_samples.to_csv(f"{cfg['output_path']}/{cfg['exp_name']}/valid_samples.csv")

    return training_samples, valid_samples


class CTDataset(Dataset):

    # label class name to index ordinal mapping. This stays constant for each CTDataset instance.
    LABEL_CLASSES = {
        0:0, 
        1:1, 
        2:2,
        3:3
    }
    LABEL_CLASSES_BINARY = {
        0:0, 
        1:1, 
        2:1,
        3:1
    }

    def __init__(self, cfg, dataframe, labels): #folder)
        '''
            Constructor. Here, we collect and index the dataset inputs and
            labels.
        '''
        self.data_root = cfg['data_root']
        self.transform = Compose([              # Transforms. Here's where we could add data augmentation (see Björn's lecture on August 11).
            Resize((cfg['image_size'])),        # For now, we just resize the images to the same dimensions...
            RandomVerticalFlip(p=0.3),
            RandomVerticalFlip(p=0.3),
            #RandomGrayscale(p=0.5),
            # RandomApplyTransform(transforms.RandomResizedCrop(224, scale = (0.08, 1.0)), p=0.3),
            #RandomApplyTransform(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), p=0.3),
            ToTensor()                          # ...and convert them to torch.Tensor.
        ])
        
        # index data into list
        self.data = []
        self.labels = []

        # meta = pd.read_csv(self.annoPath)
        meta = dataframe
        meta = meta[meta['filename'] != '2015_04_05_09_00_00.jpg']
        meta = meta[(meta['classification'] != 2) & (meta['classification'] != 3)]
        IPython.embed()
        meta = meta[meta['cameraID'] > 0]
        meta = meta.drop_duplicates(subset=['filename']).reset_index() ## maybe I should keep the original indices??

        ## add a check to make sure it exists in the folder of interest
        list_of_images = glob.glob(os.path.join(self.data_root)+'/*') 
        list_of_images = [file.split('/')[-1] for file in list_of_images]
    
        #######maybe instead walk through list_of_images
        for file, weather in zip(meta['filename'], meta['label']):
            if (random.uniform(0.0, 1.0) <= 0.1):
                continue
                # (random.uniform(0.0, 1.0) <= 0.005) ands
            if file in list_of_images: 
                imgFileName = file ## make sure there is the image file in the train folder
                if cfg['num_classes'] == 2: 
                    self.data.append([imgFileName, self.LABEL_CLASSES_BINARY[weather]])
                    self.labels.append(self.LABEL_CLASSES_BINARY[weather])
                elif cfg['num_classes'] != 2: self.data.append([imgFileName, self.LABEL_CLASSES[weather]]) ## why label index and not label?

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
        image_path = os.path.join(self.data_root, image_name) ## should specify train folder and get image name 
        img = Image.open(image_path).convert('RGB')     # the ".convert" makes sure we always get three bands in Red, Green, Blue order

        ## add in temp information to the array 


        # transform: see lines 31ff above where we define our transformations
        img_tensor = self.transform(img)
        #print(img_tensor.shape)

        return img_tensor, label
    