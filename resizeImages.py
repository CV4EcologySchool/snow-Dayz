
# Importing Image class from PIL module
from PIL import Image
import glob
import os 
import tqdm
import pandas as pd

#ImageFile.LOAD_TRUNCATED_IMAGES = True

#files = glob.glob('/datadrive/vmData/weather/train/*')
#newPath = '/datadrive/vmData/weather/train_resized/'

#files = glob.glob('/datadrive/vmData/weather/test/*')
#newPath = '/datadrive/vmData/weather/test_resized/'
 
snoqfiles = glob.glob('/datadrive/vmData/snoq/**/*')
newPath = '/datadrive/vmData/weather/snoq_resized/'

#if not os.path.exists(newPath):
#    os.makedirs(newPath)

#files = glob.glob('/Volumes/CatBreen/CV4ecology/scandcam/Snow/*')

for file in tqdm.tqdm(snoqfiles): 
    #print(file.split('/')[-1])
    try:
        #test = file[32:len(file)] #.spilt('/') ### last item file.split('/')
            # print(test)
        # Opens a image in RGB mode
        filename = file.split('/')[-1]
        new_filename = newPath + filename
        if not os.path.exists(new_filename):
            im = Image.open(str(file))
            newsize = (500, 500)
            im1 = im.resize(newsize)
            #im1.show()
            # Shows the image in image viewer
            im1 = im1.save(new_filename)
    except Exception as e:
        print(f"{e} {file}")
        pass
        
#################################################################################
olympexfiles = glob.glob('/datadrive/vmData/olympex/**/*')
OLYnewPath = '/datadrive/vmData/weather/olympex_resized/'

#if not os.path.exists(newPath):
#    os.makedirs(newPath)

#files = glob.glob('/Volumes/CatBreen/CV4ecology/scandcam/Snow/*')

for file in tqdm.tqdm(olympexfiles): 
    #print(file.split('/')[-1])
    try:
        #test = file[32:len(file)] #.spilt('/') ### last item file.split('/')
            # print(test)
        # Opens a image in RGB mode
        filename = file.split('/')[-1]
        new_filename = OLYnewPath + filename
        if not os.path.exists(new_filename):
            im = Image.open(str(file))
            newsize = (500, 500)
            im1 = im.resize(newsize)
            #im1.show()
            # Shows the image in image viewer
            im1 = im1.save(new_filename)
    except Exception as e:
        print(f"{e} {file}")
        pass
        
