
# Importing Image class from PIL module
from PIL import Image
import glob
import os 
import tqdm
import pandas as pd

#files = glob.glob('/datadrive/vmData/weather/train/*')
#newPath = '/datadrive/vmData/weather/train_resized/'

files = glob.glob('/datadrive/vmData/weather/test/*')
newPath = '/datadrive/vmData/weather/test_resized/'
 


#if not os.path.exists(newPath):
#    os.makedirs(newPath)

#files = glob.glob('/Volumes/CatBreen/CV4ecology/scandcam/Snow/*')

for file in tqdm.tqdm(files): 
    try:
        test = file[32:len(file)] #.spilt('/') ### last item
        # print(test)
    # Opens a image in RGB mode
        filename = test
        new_filename = newPath + filename
        if not os.path.exists(new_filename):
            im = Image.open(str(file))
            newsize = (500, 500)
            im1 = im.resize(newsize)
            #im1.show()
            # Shows the image in image viewer
            im1 = im1.save(new_filename)
    except Exception: 
        pass
