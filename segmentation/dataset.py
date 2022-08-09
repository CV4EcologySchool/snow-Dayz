import os
import json
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image
import pandas as pd
import glob
import display ## custom
import load_image ## custom
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from IPython.display import clear_output
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


##inputImage_path = "drive/My Drive/ImageSegmentation/InputImage/"
##Mask_Path = "drive/My Drive/ImageSegmentation/TrueMask/*"

class CTDataset(Dataset):

    def __init__(self, inputImage_path, Mask_path, cfg, folder):
        inputImages = []
        inputImagesNames = []
        for file in glob.glob(inputImage_path + "*"):
            if os.path.splitext(file)[1].lower() in ('.jpg', '.jpeg'):
                filename = file.split('/')[-1]
                image = cv2.imread(file)
                ## add a cropping step 
                image = image[30:-30,:] ## crop top and bottom

                ## optional ##
                saturation = avg_saturation(image)
                if saturation > 0.02:
                    inputImages.append(image)
                    inputImagesNames.append(filename)

        maskedImages = []
        maskedImagesNames = []
        for file in glob.glob("drive/My Drive/ImageSegmentation/TrueMask/*"):
            if os.path.splitext(file)[1].lower() in ('.jpg', '.jpeg'):
                image = cv2.imread(file)
                image = image[30:-30,:]
                maskedImages.append(image)
                filename = file.split('_',1)[1]
                name = (filename.split('.')[0])
                ending = (filename.split('.')[1]).upper()
                filename = name + '.' + ending
                maskedImagesNames.append(filename)

        ## glob doesn't always work in the same order so we want to be sure that everything is in the same order, we will sort by name (ascending) and then make sure everything lines up 
        def sortMasks(inputImages, inputImagesNames, maskedImages, maskedImagesNames):

            df1 = pd.DataFrame(inputImagesNames)
            df2 = pd.DataFrame(maskedImagesNames)

            sortedImages = []#[None] * len(df1) 
            sortedImagesNames = []#[None] * len(df1)
            sortedMasks = []#[None] * len(df1)
            sortedMasksNames = []#[None] * len(df1)
            for i in range(0, len(df1)):
                if sum(df2[0] == df1[0][i]) > 0:
                sortedImages.append(inputImages[i])
                sortedImagesNames.append(inputImagesNames[i])    
                index = (df2[df2[0] == df1[0][i]]).index[0]
                mask = maskedImages[index]
                name = maskedImagesNames[index]
                sortedMasks.append(mask) 
                sortedMasksNames.append(name) 
                # index = (df2[df2[0] == df1[0][i]]).index[0]
                #mask = maskedImages[index]
                # name = maskedImagesNames[index]
                #sortedMasks[i]= mask
                #sortedMasksNames[i] = name

            return sortedImages, sortedImagesNames, sortedMasks, sortedMasksNames
    
        self.inputImages, self.inputImagesNames, self.maskedImages, self.maskedImagesNames = sortMasks(inputImages, inputImagesNames, maskedImages, maskedImagesNames)

        def Preprocessing(self.inputImages, self.maskedImages):
            x_data = []
            #y_data1 = []
            #y_data2 = []
            y_data = []
            for i in range(0, len(self.inputImages)):
                x, y = load_image(self.inputImages[i], self.maskedImages[i])
                y = y[:,:,0]//255
                y1 = tf.cast(y, dtype = tf.uint16)
                y2 = tf.bitwise.invert(y1)
                y2= y2[:,:]//65535 ## I don't know why it's doing that but it works if you divide it by 65535. We may need to convert back to float 32
                #(print(x.shape))
                t1 = y1 ### snow
                t2 = y2 ### no snow
                stack = tf.stack([y2, y1], 2) ## channel 0 = no snow; channel 2 = snow
                x_data.append(x)
                #y_data1.append(y1)
                #y_data2.append(y2)
                y_data.append(stack)
            return x_data, y_data

        self.x_data, self.y_data = Preprocessing(self.inputImages, self.maskedImages)

    def __len__(self):
        return print(len())

    def __display__(self, idx):
        return display([self.inputImages[idx], maskedImages[idx]])

    def __visualize_MaskPixel_distribution__(self.y_data):
        self.pixelHist = []
        for i in range(len(y_data)):
            image = y_data[i][:,:,1]
            height, width = image.shape
            image = image[10:(height-10),:]

            #bottom_half = round(height/2)
            #plt.imshow(image[(round(height/2)):height,:]) just bottom half 
                # errors include (trees, animal in image, etc)
            #cropped = image[(round(height/2)):(height-10),:]
            #width, height = cropped.shape
            white_pixel_fraction = (np.sum(image))/(width *height)
            self.pixelHist.append(white_pixel_fraction)

        plt.hist(self.pixelHist, bins = 20)
        plt.xlabel('Snow Pixel Fractions')
        plt.ylabel('Number of images')
        plt.title('Snow Fraction Distribution from True Masks')
        plt.grid(False)#b=None)
        plt.tight_layout()
        plt.tick_params(axis='x', which='both', bottom=True, top=False)
        plt.tick_params(axis='y', which='both', right=False, left=True)
        plt.show()

        return self.pixelHist, plt.show() ## will this work?? 
       