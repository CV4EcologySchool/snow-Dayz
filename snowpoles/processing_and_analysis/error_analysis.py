import IPython
import matplotlib.pyplot as plt
import PIL
from PIL import Image
from PIL import ExifTags
import glob
import exiftool ## more comprehensive than PIL
import tqdm
import pandas as pd
import numpy as np
import shutil 

results = pd.read_csv('/Volumes/CatBreen/aurora_outputs/snow_poles_outputs_resized_LRe4_BS64_E100_clean_SNEX_IN/eval_orig/results.csv')
all_images = glob.glob('/Users/catherinebreen/Documents/Chapter1/WRRsubmission/data/448res/SNEX20_TLI_resized_clean/**/*.JPG')
filenames = [img.split('/')[-1] for img in all_images]

results = results.sort_values(by='difference', ascending=True)

labels = []
img_paths = []
not_found = []

# def on_key(event):
#     if event.key == 'enter':  # Capture 'Enter' key to store input
#         # try: 
#         entered_value = input("Enter a value: ")
#         labels.append(entered_value)
#         plt.close()
#         print('LABAELS list length', len(labels))
#         # except:
#         #     IPython.embed()

for i, file in tqdm.tqdm(enumerate(results['filename'])):
    if file in filenames:
        k = filenames.index(file)
        img = all_images[k]
        ## copy to this path
        dest = '/Users/catherinebreen/Dropbox/Chapter1/WRRsubmission/data/results_wa_images_for_error_analysis'
        shutil.copy2(img, dest)
        img_paths.append(file)
        
df = pd.DataFrame({'image_path':img_paths})
df.to_csv('/Users/catherinebreen/Dropbox/Chapter1/WRRsubmission/data/results_wa_images_for_error_analysis/model_error_codes.csv')    

IPython.embed()
# df = pd.DataFrame({'image_path':img_paths, 'wav_path':wav_paths, 'label':labels})
# df.to_csv('model_error_codes.csv'