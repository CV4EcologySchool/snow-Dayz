'''
Catherine Breen
Sunday, February 5, 2023

In this script we will convert all the annotations from a .csv with x and y labels, to a binary mask using the 
top and bottom of the poles as our bounding box for the 1-value pixels (white) and 0-value pixels

'''
# First we will import all the required packages 
import glob
import pandas as pd 
#from PIL import Image
import cv2
import numpy as np
import IPython
from tqdm import tqdm

# List of all the images
images = glob.glob('/Users/catherinebreen/Documents/Chapter 1/WRRsubmission/data/448res/SNEX20_TLI_resized_clean/**/*') 
labels = pd.read_csv('/Users/catherinebreen/Documents/Chapter 1/WRRsubmission/data/448res/SNEX20_TLI_resized_clean/snowPoles_resized_labels_clean.csv') ##*** DOUBLE CHEKC****###

images_that_didnt_work =[]
imagesthatworked = []
top_row_img_cutoff = []
bot_row_img_cutoff= []

for image in tqdm(images): 
    img_name = image.split('/')[-1] ## the last part of the filename
    first_letter = img_name[0]
    ## look up image in the csv:
    try: 
        top_x = float(labels.loc[labels['filename'] == img_name]['x1']) #x1
        top_y = float(labels.loc[labels['filename'] == img_name]['y1']) #y1
        bottom_x = float(labels.loc[labels['filename'] == img_name]['x2']) #x2
        bottom_y = float(labels.loc[labels['filename'] == img_name]['y2']) #y2
        imagesthatworked.append(img_name)
    ## create bounding boxes using the coordinates, although we will add a small buffer in the x direction only 
        top_left = [top_x - 2,top_y]
        top_right = [top_x + 2, top_y]
        bottom_left = [bottom_x + 2, bottom_y]
        bottom_right = [bottom_x - 2, bottom_y]

        # create empty image of the same dimensions as original image:
        img = cv2.imread(image)
        height, width, channels = img.shape
        zeros = np.zeros((width,height)) ## check to make sure this looks right on actual images

        #now set the bounding box within the empty image to 255 (can be converted to 1 later)
        # Define an array of endpoints of triangle
        points = np.array([top_left, top_right, bottom_left, bottom_right])
    
        # Use fillPoly() function and give input as
        # image, end points,color of polygon
        # Here color of polygon will blue
        masked = cv2.fillPoly(zeros, pts=np.int32([points]), color=(255, 0, 0)) ##np.int32([points])

        ## cropping based on camera type: 
        camDict = {'Reconyx':['C','T'],'Wingscape':['E','W']}
        if first_letter in camDict['Reconyx']:
            bottom_5 = round(height*(90/100)) ## 2%? 
            top_5 =  round(height*(100/100)) ## 2%

        if first_letter in camDict['Wingscape']:

        bottom_5 = round(height*(90/100)) ## 2%? 
        top_5 =  round(height*(100/100)) ## 2%

        ## cut off top 10% ##
        masked = masked[0:bottom_5,:] ## crop bottom 10% and top 10% (could make this 5%, and it would 24:424)
        img = img[0:bottom_5,:] ## pictures going upside down

        ## check the number of instances where it "cuts off" the image 
        if np.sum(masked[0:1,:])>0: ## first row
            #print('top row cutoff')
            top_row_img_cutoff.append(img_name)
        if np.sum(masked[-1,:])>0:  #429:430
            #print('bottom row cutoff')
            bot_row_img_cutoff.append(img_name)

        #cv2.imshow('masked iamge', masked)
        # save
        #IPython.embed()
        #cv2.imwrite(f'/Volumes/CatBreen/CV4ecology/segmented_images_resized_cropped2_10/mask_{img_name}', masked) #, binary * 255)
        #cv2.imwrite(f'/Volumes/CatBreen/CV4ecology/input_images_resized_cropped2_10/{img_name}', img) #, binary * 255)
        #cv2.imwrite('/Users/catherinebreen/Documents/test')
        #IPython.embed()

    except Exception:
        print(Exception)
        print('ERROR')
        images_that_didnt_work.append(img_name)
        #IPython.embed()

IPython.embed()