#!/usr/bin/env python
# coding: utf-8

# **Catherine Breen \
# Assignment 0**
# #### **note**: I plan to add a little bit more code over the weekend/before Monday, but I wanted to submit what I had so far!

# # Data Visualization

# I will be building a model to detect weather from camera traps. Currently, the only way to truly "know" what is rain or snow from a camera trap is to use a fancy device called a disdrometer. It measures the speed and size of precipitation droplets to differentiate between rain and snow. However, this device is costly and hard to install in remote terrain. Cameras on the other hand can take pictures daily, hourly and detect visual differences between rain and snow. For example, there are some differences: snow tends to look big, round and white, and rain tends to look more "streak-y." Other work has differentiated the two using histograms, lighting, etc. We hope to build a model that can accurately differentiate between the two so that future work could install cameras instead of disdrometers and pick up rain and snow for wildlife and other ecological studies.
# 
# For the class, I will be combining two datasets. One dataset is from a wildlife camera network in Norway (from herein called the Norway dataset). This dataset is your normal wildlife camera dataset. The other dataset is a dataset from Washington that had a disdrometer next to it. Hypothetically these image lables are less "noisy" than the Norway dataset since sometimes we didn't totally know in the Norway dataset what was rain and what was snow. However, in the Washington dataset, there was a daylight savings shift and a couple places where the disdrometer and camera were off by a day so there is some noise in that too. 

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
print('packages loaded')


# In[ ]:


data = pd.read_csv('/datadrive/data/all_labels_QC.csv')
print(data.head())


# In[ ]:


label_list = ['None', 'Snow', 'Rain', 'Fog', 'Other']
print(label_list)


# In[ ]:


images = data['File']
annotations = data['Weather']
categories = pd.unique(data['Weather'])
#train_categories = set([ann['category_id'] for ann in annotations])
locations = list(pd.unique(data['location']))
days = list(pd.unique(data['Date']))
im_to_cat = {data['File'][i]: data['Weather'][i] for i in range(0,len(data['File']))}


# In[ ]:


print('High-level statistics:\n')

print('Images: '+str(len(images)))
print('Annotations: '+str(len(annotations)))
print('Categories: ' + str(len(categories)))
print('Weather Images: ' + str(len([data['Weather'] for weather in annotations if (weather == 'Snow') or (weather == 'Rain') or (weather == 'Fog')])))
print('Weather Labels: ' + str(pd.unique(data['Weather'])))
print('Empty Images: '+ str(len([data['Weather'] for weather in annotations if (weather == 'None')])))
print('Locations: ' + str(len(locations)))
print('Days: ' + str(len(pd.unique(data['Date']))))


# In[ ]:


plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams['figure.dpi'] = 100
images_per_category = pd.DataFrame(annotations.value_counts()).reset_index().rename(columns={'index':'labels', 'Weather':'Weather'})
#for im in images:
 # images_per_category[im_to_cat[im['file']]].append(im['file'])
ind = range(len(images_per_category['labels']))
plt.bar(ind, images_per_category['Weather'],edgecolor = 'b', log=True)
plt.xlabel('Category')
plt.xticks(ind, labels = ['None', 'Snow', 'Fog', 'Rain', 'Other'])
plt.ylabel('Number of images')
plt.title('Images per category')
plt.grid(b=None)
plt.tight_layout()
plt.tick_params(axis='x', which='both', bottom=True, top=False)
plt.tick_params(axis='y', which='both', right=False, left=True)
plt.show()


# In[ ]:





# In[ ]:


plt.rcParams['figure.figsize'] = [12, 6]
locs = [locations[1], locations[2], locations[3]]

images_per_category_per_loc = {}
for loc in locations:
    images_per_location = data[data['location'] == loc]
    none, Snow, Rain, Fog = sum(images_per_location['Weather'] == 'None'),         sum(images_per_location['Weather'] == 'Snow'), sum(images_per_location['Weather'] == 'Rain'),         sum(images_per_location['Weather'] == 'Fog')
    df = pd.DataFrame({'labels': ['None', 'Snow', 'Rain', 'Fog'], 'Weather':[none, Snow, Rain, Fog]})
    images_per_category_per_loc[loc] = df

ind = range(0,4)

for idx, loc in enumerate(locs):
  plt.subplot(3,1,idx+1)
  plt.bar(ind, images_per_category_per_loc[loc]['Weather'], log=True)
  plt.xlabel('Category')
  plt.ylabel('Number of images')
  plt.xticks(ind, labels = ['None', 'Snow', 'Fog', 'Rain'])
  plt.title('Location: '+str(loc))
  plt.grid(b=None)
  plt.tight_layout()
  plt.tick_params(axis='x', which='both', bottom=True, top=False)
  plt.tick_params(axis='y', which='both', right=False, left=True)
plt.show()


# # Prompt 2: load one of your images or videos and visualize it
# 
# I have an azure account, so I will run this again on my azure account and update the paths so that everything is ready by August. 

# In[ ]:


import os
import glob #for loading images from a directory
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import random
import numpy as np
import pandas as pd
from PIL import Image
from urllib.request import urlopen
from PIL.ExifTags import TAGS 
from scipy import ndimage
print()

