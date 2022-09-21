'''
To do:
put in keypoint columns in the config file and update here

Updates: 
- switched train/test from random to 16 cameras : 4 cameras (OOD testing)
- specified columns in keypoints because we have extra columns in our df
- Specifed the __getitem__ function to look in nested folders of cameraIDs
    rather than training and testing
- specified the cameras for validation, 2 from each side, split from in and out of canopy
- hardcoded the training files path

'''

import torch
import cv2
import pandas as pd
import numpy as np
import config
import utils
from torch.utils.data import Dataset, DataLoader
import IPython
import matplotlib.pyplot as plt
import glob
import torch
import torchvision.transforms as T
from PIL import Image
from PIL import Image, ImageFile
import albumentations as A
from torchvision.transforms import Compose, Resize, ToTensor

##### re-write this for out of domain testing
def train_test_split(csv_path, path) : # split):
    #IPython.embed()
    df_data = pd.read_csv(csv_path)
    #len_data = len(df_data)
    # calculate the validation data sample length
    ## validation cameras: 
    val_cameras = ['E9E', 'W2B', 'E6B', 'W8A','CHE8', 'CHE9', 'CHE10'] ## check, out of domanin testing

    #valid_split = int(len_data * split)
    # calculate the training data samples length
    #train_split = int(len_data - valid_split)

    #training_samples = df_data.iloc[:train_split][:]
    #valid_samples = df_data.iloc[-valid_split:][:]

    valid_samples = df_data[df_data['Camera'].isin(val_cameras)]  
    training_samples = df_data[~df_data['Camera'].isin(val_cameras)]

    ##### only images that exist
    #IPython.embed()
    all_images = glob.glob(path + ('/**/*.JPG'))
    filenames = [item.split('/')[-1] for item in all_images]
    valid_samples = valid_samples[valid_samples['filename'].isin(filenames)].reset_index()
    training_samples = training_samples[training_samples['filename'].isin(filenames)].reset_index()

    return training_samples, valid_samples


class snowPoleDataset(Dataset):
    def __init__(self, samples, path): # split='train'):
        self.data = samples
        self.path = path
        self.resize = 224
        #self.split = split
        # self.transform = Compose([              # Transforms. Here's where we could add data augmentation (see Björn's lecture on August 11).
        #     Resize(([224, 224])), 
        #            # For now, we just resize the images to the same dimensions...
        #     ToTensor()                          # ...and convert them to torch.Tensor.
        # ])
        self.transform = A.Compose([
                #A.RandomCrop(width=100, height=100, p=0.5),
                #A.Rotate(p=0.5),
                #A.HorizontalFlip(p=0.5),
            A.CropAndPad(px=75, p =1.0), ## final model is 50 pixels
            A.ShiftScaleRotate(p=0.5),
            #A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.8),
            A.OneOf([
                A.Affine(translate_px = (-5, 5),p=0.5), ### will throw off algorithm 
                A.Affine(scale = (0.5, 1.0), p =0.5),
                A.Affine(translate_percent = (-0.15,0.15), p =0.5)], p =0.5),
            A.Resize(224, 224),
            ], 
            keypoint_params=A.KeypointParams(format='xy'))

    def __len__(self):
        return len(self.data)

    def __filename__(self, index):
        #print('test')
        filename = self.data.iloc[index]['filename']
        return filename
    
    def __getitem__(self, index):
        cameraID = self.data.iloc[index]['filename'].split('_')[0] ## need this to get the right folder
        filename = self.data.iloc[index]['filename']
        #IPython.embed()
        
        image = cv2.imread(f"{self.path}/{cameraID}/{self.data.iloc[index]['filename']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w, channel = image.shape
        
        # resize the image into `resize` defined above
        image = cv2.resize(image, (self.resize, self.resize))
        #IPython.embed()
        # again reshape to add grayscale channel format
        image = image / 255.0
        # transpose for getting the channel size to index 0
        #image = np.transpose(image, (2, 0, 1))

        ### PIL library
        #img = Image.open(f"{self.path}/{cameraID}/{self.data.iloc[index]['filename']}").convert('RGB')  
        #orig_h, orig_w = img.size

        #img_tensor = self.transform(img)  
        # if config.COLOR_JITTER == True:# and (split == 'train'): 
        #     jitter = T.ColorJitter(brightness=.8, hue=.5)
        #     img_tensor = jitter(img_tensor)

        # if config.RANDOM_ROTATION == True:# and (split == 'train'): 
        #     rotator = T.RandomRotation(degrees=(0, 180))
        #     img_tensor = rotator(img_tensor)

        # if config.GAUSSIAN == True:
        #     blur = T.GaussianBlur(kernel_size=(51, 91), sigma=(3,7))
        #     img_tensor = blur(img_tensor)

        # get the keypoints
        keypoints = self.data.iloc[index][1:][['x1','y1','x2','y2']]  #[3:7]  ### change to x1 y1 x2 y2
        keypoints = np.array(keypoints, dtype='float32')
        # reshape the keypoints
        keypoints = keypoints.reshape(-1, 2)
        # rescale keypoints according to image resize
        #IPython.embed()
        keypoints = keypoints * [self.resize / orig_w, self.resize / orig_h]

        #=if config.RANDOM_ROTATION == True: 
         #   image = T.ColorJitter(brightness=.5, hue=.3)
        #utils.vis_keypoints(image, keypoints)
        transformed = self.transform(image=image, keypoints=keypoints)
        img_transformed = transformed['image']
        keypoints = transformed['keypoints']
        #IPython.embed()
        #utils.vis_keypoints(transformed['image'], transformed['keypoints'])
        image = np.transpose(img_transformed, (2, 0, 1))
        #IPython.embed()
        if len(keypoints) != 2:
            IPython.embed()
            utils.vis_keypoints(transformed['image'], transformed['keypoints'])

        return {
            'image': torch.tensor(image, dtype=torch.float),
            'keypoints': torch.tensor(keypoints, dtype=torch.float),
            'filename': filename
        }

# get the training and validation data samples
training_samples, valid_samples = train_test_split(f"{config.ROOT_PATH}/snowPoles_labels.csv", f"{config.ROOT_PATH}") #config.TEST_SPLIT)

# initialize the dataset - `snowPoleDataset()`
train_data = snowPoleDataset(training_samples, 
                                 f"{config.ROOT_PATH}")  ## we want all folders
#IPython.embed()
valid_data = snowPoleDataset(valid_samples, 
                                 f"{config.ROOT_PATH}")
# prepare data loaders
train_loader = DataLoader(train_data, 
                          batch_size=config.BATCH_SIZE, 
                          shuffle=True, num_workers = 0)
valid_loader = DataLoader(valid_data, 
                          batch_size=config.BATCH_SIZE, 
                          shuffle=False, num_workers = 0) 
print(f"Training sample instances: {len(train_data)}")
print(f"Validation sample instances: {len(valid_data)}")

# whether to show dataset keypoint plots
if config.SHOW_DATASET_PLOT:
    utils.dataset_keypoints_plot(valid_data)


