'''
sample code to rename images to make sure that camera ID is in the front of the filename
Catherine Breen
'''


import os
import glob

folder = glob.glob('/Volumes/CatBreen/CV4ecology/SNEX20_TLI/TWISP-U-01/*.JPG')

for file in folder:
    filename = file.split('/')[-1]
    filename = filename.split('IMG_')[1]
    new_filename = 'TWISP-U-01_' + filename
    src = file.split('IMG')[0]
    new_path = src+new_filename
    print(new_path)
    os.rename(file, new_path)