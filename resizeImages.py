
# Importing Image class from PIL module
from PIL import Image
import glob
import os 

files = glob.glob('/datadrive/data/weather/train/*.JPG')
newPath = '/datadrive/data/weather/train_resized/'
 
if not os.path.exists(newPath):
    os.makedirs(newPath)

for file in files: 
    filename = file.spilt('/')[-1] ### last item
 # Opens a image in RGB mode
    im = Image.open(str(file))
    newsize = (300, 300)
    im1 = im.resize(newsize)
    # Shows the image in image viewer
    im1 = im1.save(newPath + filename)
