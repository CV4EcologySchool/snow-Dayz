
# Importing Image class from PIL module
from PIL import Image
import glob
files = glob.glob('/datadrive/data/weather/**/*.JPG')

 
for file in files: 
 # Opens a image in RGB mode
    im = Image.open(str(file))
    
    # Size of the image in pixels (size of original image)
    # (This is not mandatory)
    width, height = im.size

    newsize = (300, 300)
    im1 = im1.resize(newsize)
    # Shows the image in image viewer

