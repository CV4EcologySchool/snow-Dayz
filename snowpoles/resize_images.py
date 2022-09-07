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

snowpolefiles = glob.glob('/Volumes/CatBreen/CV4ecology/SNEX20_TLI_test/**/*') #glob.glob('/datadrive/vmData/SNEX20_TLI/**/*')
snowpoleList = [item.split('/')[-1] for item in snowpolefiles]
labels = pd.read_csv('/Volumes/CatBreen/CV4ecology/SNEX20_TLI_test/snowPoles_labels.csv')
snowpoleImages =  labels[labels['filename'].isin(snowpoleList)].reset_index()

newPath = '/Volumes/CatBreen/CV4ecology/SNEX20_TLI_resized/' #'/datadrive/vmData/SNEX20_TLI_resized/'

if not os.path.exists(newPath):
    os.makedirs(newPath)

files =[]
x1s=[]
y1s=[]
x2s=[]
y2s=[]

for file in tqdm.tqdm(snowpoleImages['filename']): 
    #IPython.embed()
    CameraID = file.split('_')[0] ## need this
    image = cv2.imread(f"/Volumes/CatBreen/CV4ecology/SNEX20_TLI_test/{CameraID}/{file}")
    new_filename = newPath + file
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
    files.append(file)
    x1s.append(keypoints[0])
    y1s.append(keypoints[1])
    x2s.append(keypoints[2])
    y2s.append(keypoints[3])
    #keypoints = keypoints.view(keypoints.size(0), -1)

labels_resized = pd.DataFrame({'filename':files, 'x1':x1s, 'y1':y1s, 'x2':x2s, 'y2':y2s})
IPython.embed()
labels_resized.to_csv('/Volumes/CatBreen/CV4ecology/SNEX20_TLI_resized/snowPoles_labels.csv')
