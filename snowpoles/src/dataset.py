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


##### re-write this for out of domain testing
def train_test_split(csv_path, split):
    df_data = pd.read_csv(csv_path)
    len_data = len(df_data)
    # calculate the validation data sample length
    ## validation cameras: 
    val_cameras = ['E9E', 'W2B', 'E6B', 'W8A'] ## check

    #valid_split = int(len_data * split)
    # calculate the training data samples length
    #train_split = int(len_data - valid_split)

    #training_samples = df_data.iloc[:train_split][:]
    #valid_samples = df_data.iloc[-valid_split:][:]
    valid_samples = df_data[df_data['CameraID'].isin(val_cameras)]
    training_samples = df_data[~df_data['CameraID'].isin(val_cameras)]

    return training_samples, valid_samples


class snowPoleDataset(Dataset):
    def __init__(self, samples, path):
        self.data = samples
        self.path = path
        self.resize = 224
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        cameraID = self.data.iloc[index]['filename'].split('_')[0]
        image = cv2.imread(f"{self.path}/{cameraID}/{self.data.iloc[index]['filename']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w, channel = image.shape
        # resize the image into `resize` defined above
        image = cv2.resize(image, (self.resize, self.resize))
        # again reshape to add grayscale channel format
        image = image / 255.0
        # transpose for getting the channel size to index 0
        image = np.transpose(image, (2, 0, 1))
        # get the keypoints
        keypoints = self.data.iloc[index][1:][3:7]  ### change to x1 y1 x2 y2

        keypoints = np.array(keypoints, dtype='float32')
        # reshape the keypoints
        keypoints = keypoints.reshape(-1, 2)
        # rescale keypoints according to image resize
        keypoints = keypoints * [self.resize / orig_w, self.resize / orig_h]
        return {
            'image': torch.tensor(image, dtype=torch.float),
            'keypoints': torch.tensor(keypoints, dtype=torch.float),
        }

# get the training and validation data samples
training_samples, valid_samples = train_test_split(f"{config.ROOT_PATH}/snowPoles_labels.csv",
                                                   config.TEST_SPLIT)

# initialize the dataset - `snowPoleDataset()`
train_data = snowPoleDataset(training_samples, 
                                 f"{config.ROOT_PATH}/**/*.JPG")  ## we want all folders
valid_data = snowPoleDataset(valid_samples, 
                                 f"{config.ROOT_PATH}/**/*.JPG")
# prepare data loaders
train_loader = DataLoader(train_data, 
                          batch_size=config.BATCH_SIZE, 
                          shuffle=True)
valid_loader = DataLoader(valid_data, 
                          batch_size=config.BATCH_SIZE, 
                          shuffle=False)
print(f"Training sample instances: {len(train_data)}")
print(f"Validation sample instances: {len(valid_data)}")

# whether to show dataset keypoint plots
if config.SHOW_DATASET_PLOT:
    utils.dataset_keypoints_plot(valid_data)


