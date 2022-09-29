## for each image resize and scale down points 
# Resave the labels into 
# snowPoles_labels_RESIZED.csv     
# 
#    

import glob
import cv2
import numpy as np
import pandas as pd
import os
import tqdm
import IPython
from PIL import Image

#snowpolefiles = glob.glob('/Volumes/CatBreen/CV4ecology/SNEX20_TLI_test/**/*') #glob.glob('/datadrive/vmData/SNEX20_TLI/**/*')
#snowpolefiles = glob.glob('/datadrive/vmData/SNEX20_TLI/**/*')
#snowpolefiles = glob.glob('/Volumes/CatBreen/Chelewah_Timelapse_Photos/samples/**/*')
snowpolefiles = glob.glob('/Volumes/CatBreen/Okanagan_Timelapse_Photos/**/*')

snowpoleList = [item.split('/')[-1] for item in snowpolefiles]
labels = pd.read_csv('/Users/catherinebreen/Documents/MATLAB/Okanagan/okanagan_poles.csv')
#labels = pd.read_csv('/datadrive/vmData/SNEX20_TLI/snowPoles_labels_clean.csv')
snowpoleImages =  labels[labels['filename'].isin(snowpoleList)].reset_index() ## only error code = 0 

newPath = '/Volumes/CatBreen/CV4ecology/SNEX20_TLI_resized/' #'/datadrive/vmData/SNEX20_TLI_resized/'
#newPath = '/Volumes/CatBreen/Chelewah_Timelapse_Photos/Resized/'

if not os.path.exists(newPath):
    os.makedirs(newPath)

Camera = []
files =[]
x1s=[]
y1s=[]
x2s=[]
y2s=[]

for file in tqdm.tqdm(snowpoleImages['filename']): 
    #IPython.embed()
    CameraID = file.split('_')[0] ## need this
    image = cv2.imread(f"/Volumes/CatBreen/Okanagan_Timelapse_Photos/{CameraID}/{file}")
    new_filename = newPath + CameraID + ('/')
    if not os.path.exists(new_filename):
        os.makedirs(new_filename)
    new_filename = newPath + CameraID + ('/') + file
    orig_h, orig_w, channel = image.shape
    newsize = (448, 448)
    im1 = cv2.resize(image, newsize)
    im1 = cv2.imwrite(new_filename, im1)

    keypoints = snowpoleImages.loc[snowpoleImages['filename'] == file][['x1','y1','x2','y2']]

    #keypoints = self.data.iloc[index][1:][['x1','y1','x2','y2']]  #[3:7]  ### change to x1 y1 x2 y2
    keypoints = np.array(keypoints, dtype='float32')
    # reshape the keypoints
    keypoints = keypoints.reshape(-1, 2)
    # rescale keypoints according to image resize
    keypoints = keypoints * [448 / orig_w, 448 / orig_h]
    ## flatten
    keypoints = keypoints.flatten()
    Camera.append(CameraID)
    files.append(file)
    x1s.append(keypoints[0])
    y1s.append(keypoints[1])
    x2s.append(keypoints[2])
    y2s.append(keypoints[3])
    #keypoints = keypoints.view(keypoints.size(0), -1)

labels_resized = pd.DataFrame({'Camera':Camera, 'filename':files, 'x1':x1s, 'y1':y1s, 'x2':x2s, 'y2':y2s})
#IPython.embed()IPython.embed()
labels_resized.to_csv('//Volumes/CatBreen/Okanagan_Timelapse_Photos/snowPoles_labels_clean.csv')
IPython.embed()


# /datadrive/vmData/SNEX20_TLI_resized/snowPoles_labels_clean.csv /datadrive/vmData/SNEX20_TLI_resized_clean/snowPoles_labels_clean.csv 