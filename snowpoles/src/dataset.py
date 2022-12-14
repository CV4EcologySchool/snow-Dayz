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
- data aug docs: https://albumentations.ai/docs/getting_started/keypoints_augmentation/ 
- data aug docs cont. : https://albumentations.ai/docs/api_reference/augmentations/transforms/

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
import albumentations as A ### better for keypoint augmentations
from torchvision.transforms import Compose, Resize, ToTensor
from sklearn.model_selection import train_test_split

##### re-write this for out of domain testing
def train_test_split(csv_path, path, split, domain):
    #IPython.embed()
    df_data = pd.read_csv(csv_path)

    ########## EXP #1: CAN MODEL DETECT SNOW  
    if domain == True: 
        print('testing IN DOMAIN')
        len_data = len(df_data)
        # calculate the validation data sample length
        ## validation cameras: 
        #valid_split = int(len_data * split)
        # calculate the training data samples length
        #train_split = int(len_data - valid_split)
        #training_samples = df_data.iloc[:train_split][:]
        #alid_samples = df_data.iloc[-valid_split:][:]
        #training_samples = df_data.sample(n = int(len_data * split))
        #training_samples, valid_samples = strain_test_split(df_data, test_size=split, random)

        training_samples = df_data.sample(frac=0.9, random_state=100) ## same shuffle everytime
        valid_samples = df_data[~df_data.index.isin(training_samples.index)]

    else:
        print('testing OUT OF DOMAIN')
        ######### EXP #2: OUT OF DOMAIN TESTING ############
        val_cameras = ['E9E', 'W2E', 'CHE8', 'CHE9', 'TWISP-U-01'] 
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

    def __init__(self, samples, path, domain): # split='train'):
        self.data = samples
        self.path = path
        self.resize = 224
        # self.transform = Compose([              # Transforms. Here's where we could add data augmentation (see Bj??rn's lecture on August 11).
        #     Resize(([224, 224])), 
        #            # For now, we just resize the images to the same dimensions...
        #     ToTensor()                          # ...and convert them to torch.Tensor.
        # ])
        if domain == True: 
            self.transform = A.Compose([
                A.Resize(224, 224),
                ], keypoint_params=A.KeypointParams(format='xy'))
        else: 
            self.transform = A.Compose([
                A.ToFloat(max_value=1.0),
                A.CropAndPad(px=75, p =1.0), ## final model is 50 pixels
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.2, rotate_limit=20, p=0.5),
                #A.OneOf([
                 #   A.Affine(translate_px = (-3, 3),p=0.5), ### will throw off algorithm 
                  #  A.Affine(scale = (0.5, 1.0), p =0.5),
                   # A.Affine(translate_percent = (-0.15,0.15), p =0.5)], p =0.5),
                A.OneOf([
                    A.RandomBrightnessContrast(p=0.5),
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.5),
                    A.ToGray(p=0.5)], p = 0.5),
                #A.FromFloat(max_value=1.0),
                A.Resize(224, 224),
                ], keypoint_params=A.KeypointParams(format='xy'))

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
        #IPython.embed()
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
training_samples, valid_samples = train_test_split(f"{config.ROOT_PATH}/snowPoles_labels_clean.csv", f"{config.ROOT_PATH}", config.TEST_SPLIT, config.DOMAIN)

# initialize the dataset - `snowPoleDataset()`
train_data = snowPoleDataset(training_samples, 
                                 f"{config.ROOT_PATH}", config.DOMAIN)  ## we want all folders
#IPython.embed()
valid_data = snowPoleDataset(valid_samples, 
                                 f"{config.ROOT_PATH}", domain = True) # we always want the transform to be the normal transform
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
    utils.dataset_keypoints_plot(train_data)
    utils.dataset_keypoints_plot(valid_data)




