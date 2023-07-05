#from tokenize import PlainToken
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
from PIL import Image
import IPython
from scipy.spatial import distance

data = pd.read_csv('/Users/catherinebreen/Documents/Chapter1/BenStakes/ok_top_bottom_points.csv') ## manual labels of interest, make sure they are in the right resolution!
snwfreetbl = pd.read_csv('/Users/catherinebreen/Documents/Chapter1/WRRsubmission/snowfree_table.csv') # updated, make sure these tables have the meta information for the sites of interest
dates = pd.read_csv('/Users/catherinebreen/Documents/Chapter1/WRRsubmission/labeledImgs_datetime_info.csv') # updated, make sure these tables have the meta information for the sites of interest
dates.rename(columns = {'filenames' : 'filename'}, inplace = True)

data = data.merge(dates, on='filename', how='left') ## make sure that the datetime format is right

che_ok_files = []
che_ok_dates = []
che_ok_cam = []
che_ok_length_cm = []
che_ok_sd = []

for filename in data['filename']:
    camera = filename.split('_')[0]
    date = data.loc[data['filename'] == filename, 'datetimes'].iloc[0]
    conversion = snwfreetbl.loc[snwfreetbl['camera'] == camera, 'conversion'].iloc[0]
    snwfreestake = snwfreetbl.loc[snwfreetbl['camera'] == camera, 'snow_free_cm'].iloc[0]
    #IPython.embed()
    x1 = data.loc[data['filename'] == filename, 'x1'].iloc[0]
    y1 = data.loc[data['filename'] == filename, 'y1'].iloc[0]
    x2 = data.loc[data['filename'] == filename, 'x2'].iloc[0]
    y2 = data.loc[data['filename'] == filename, 'y2'].iloc[0]
    pixelLengths = distance.euclidean([x1,y1],[x2,y2])
    cmLengths = pixelLengths * float(conversion)
    snowdepth = snwfreestake - cmLengths

    che_ok_files.append(filename), che_ok_dates.append(date)
    che_ok_cam.append(camera), che_ok_length_cm.append(cmLengths), che_ok_sd.append(snowdepth)


    df = pd.DataFrame({'camera': che_ok_cam, 'filename': che_ok_files, 'dates':che_ok_dates, 'cmLengths': che_ok_length_cm,'snowDepth':che_ok_sd})


IPython.embed()