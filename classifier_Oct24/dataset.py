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
from PIL import ExifTags

class RandomApplyTransform:
    def __init__(self, transform, p=0.5):
        self.transform = transform
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return self.transform(img)
        return img


ImageFile.LOAD_TRUNCATED_IMAGES = True


def train_test_split(cfg, images_path, labels): # val_labels):
    #IPython.embed()
    df_data = pd.read_csv(labels)
    df_data = df_data[df_data['filename'] != '2015_04_05_09_00_00.jpg']
    #df_data = df_data[(df_data['label'] != 2) & (df_data['label'] != 3)]
    df_data['cameraID'] = df_data['cameraID'].astype(str)
    #df_data = df_data[(df_data['cameraID'] != '842')]

    #####
    cameras = ["639", "1480", "1620", "641", "1761", "1571", "1570", "1760", "953", "1180", "1803",
    "3034", "1802", "3036", "1788", "1557", "870", "1725", "1409", "513", "1825", "244",
    "1135", "1754", "1635", "1634", "1361", "1410", "673", "517", "845", "258", "54",
    "1767", "1487", "1585", "1551", "656", "1550", "1446", "1746", "1789", "1726", "1444",
    "1445", "1417", "469", "625", "194", "631", "1655", "1565", "954", "1175", "938",
    "1424", "598", "1148", "1647", "1374", "1156", "865", "1733", "193", "1193", "501",
    "600", "1425", "1747", "486", "535", "640", "1107", "1196", "1376", "3043", "1181",
    "506", "1355", "636", "484", "1438", "43", "175", "1149", "638", "662", "1368",
    "944", "554", "309", "945", "1403", "1529", "842", "249", "1669", "1262", "929",
    "828", "231", "1382", "521", "1641", "1613", "1354", "1691", "1466", "653", "1397",
    "457", "580", "897", "871", "127", "1170", "1627", "1186", "555", "1250", "1381",
    "240", "1121", "1423", "704", "1162", "611", "330", "1452", "1252", "1140", "894",
    "1598", "1592", "292", "192", "701", "494", "895", "1197", "1120", "470", "1593",
    "1591", "317", "692", "1951", "456", "1508", "675", "1117", "651", "1248", "468",
    "460", "1150", "839", "621", "1501", "827", "2027", "835", "1194", "1774", "843",
    "447", "1662", "710", "1263", "1775", "606", "241", "297", "246", "970", "1168",
    "25", "270", "117", "618", "1267", "1166", "2029", "980", "664", "859", "1185",
    "1141", "693", "831", "928", "903", "507", "925", "699", "1190", "626", "824",
    "518", "688", "896", "1119", "585", "1528", "1395", "566", "1486", "328", "700",
    "1739", "415", "526", "965", "829", "1494", "1493", "1495", "41", "459", "1683",
    "1612", "702", "1431", "1139", "1147", "529", "341", "1345", "1719", "709", "728",
    "1152", "1599", "1712", "850", "1705", "1144", "694", "916", "3033", "1718", "851",
    "869", "1626", "1142", "706", "1184", "868", "1192", "257", "979", "747", "696"]
    
    # df_data['year'] = [i.split(':')[0] for i in df_data['datetime']]
    # df_data['month'] = [int(i.split(':')[1]) if ":" in i else i for i in df_data['datetime']]
    # valid_samples = df_data[(df_data['year'] == '2019') & (df_data['cameraID'].astype(str).isin(cameras)) & (df_data['month'].isin([10,11,12,
    #                                                                                                                         1,2,3,4]))]
    # Split into train, val, and test sets ensuring no overlap
    #train_cameras = random.sample(cameras, 50)
    #remaining_cameras = list(set(cameras) - set(train_cameras))  # Remaining cameras after train selection
    val_cameras = random.sample(cameras, 50)
    remaining_cameras = list(set(cameras) - set(val_cameras))  # Remaining cameras after val selection
    test_cameras = random.sample(remaining_cameras, 150)
        
    training_samples = df_data[~df_data['cameraID'].astype(str).isin(test_cameras) & ~df_data['cameraID'].astype(str).isin(val_cameras)]
    valid_samples = df_data[df_data['cameraID'].astype(str).isin(val_cameras)]
    test_samples = df_data[df_data['cameraID'].astype(str).isin(test_cameras)] 

    print('train', len(pd.unique(training_samples['cameraID'])))
    print('val',len(pd.unique(valid_samples['cameraID'])))
    print('test', len(pd.unique(test_samples['cameraID'])))
    print(len((training_samples['cameraID'])))
    print(len((valid_samples['cameraID'])))
    print(len((test_samples['cameraID'])))

    ##### only images that exist
    all_images = glob.glob(images_path + ('/*'))
    filenames = [item.split('/')[-1] for item in all_images]
    valid_samples = valid_samples[valid_samples['filename'].isin(filenames)]
    print('valid',valid_samples['label'].value_counts())
    training_samples = training_samples[training_samples['filename'].isin(filenames)]
    print('train',training_samples['label'].value_counts())
    
    if not os.path.exists(f"{cfg['output_path']}/{cfg['exp_name']}"):
            os.makedirs(f"{cfg['output_path']}/{cfg['exp_name']}", exist_ok=True)
    training_samples.to_csv(f"{cfg['output_path']}/{cfg['exp_name']}/training_samples.csv")
    valid_samples.to_csv(f"{cfg['output_path']}/{cfg['exp_name']}/valid_samples.csv")
    test_samples.to_csv(f"{cfg['output_path']}/{cfg['exp_name']}/test_samples.csv")

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
        2:0,
        3:0
    }

    def __init__(self, cfg, dataframe, labels): #folder)
        '''
            Constructor. Here, we collect and index the dataset inputs and
            labels.
        '''
        self.data_root = cfg['data_root']
        self.transform = Compose([              # Transforms. Here's where we could add data augmentation (see Björn's lecture on August 11).
            Resize((cfg['image_size'])),        # For now, we just resize the images to the same dimensions...
            # RandomVerticalFlip(p=0.3),
            # RandomVerticalFlip(p=0.3),
            #RandomGrayscale(p=0.5),
            # RandomApplyTransform(transforms.RandomResizedCrop(224, scale = (0.08, 1.0)), p=0.3),
            RandomApplyTransform(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), p=0.3),
            ToTensor()                          # ...and convert them to torch.Tensor.
        ])
        
        # index data into list
        self.data = []
        self.labels = []

        # meta = pd.read_csv(self.annoPath)
        meta = dataframe
        #IPython.embed()
        #meta = meta[meta['cameraID'] > 0]
        meta = meta.drop_duplicates(subset=['filename']).reset_index() ## maybe I should keep the original indices??

        ## add a check to make sure it exists in the folder of interest
        list_of_images = glob.glob(os.path.join(self.data_root)+'/*') 
        list_of_images = [file.split('/')[-1] for file in list_of_images]
    
        #######maybe instead walk through list_of_images
        for file, weather in zip(meta['filename'], meta['label']):
            # if (random.uniform(0.0, 1.0) <= 0.005): continue
            #     # (random.uniform(0.0, 1.0) <= 0.005) ands
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
        #print(image_path)
        #IPython.embed()
        ## add in temp information to the array 
        #exif_info = img._getexif()

        # transform: see lines 31ff above where we define our transformations
        img_tensor = self.transform(img)
        #print(img_tensor.shape)

        return img_tensor, label
    