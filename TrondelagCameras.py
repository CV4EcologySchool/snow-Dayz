## bash script to upload Trondelag cameras 
import os
import glob
import pandas as pd

folder2018 = '/Volumes/CBreen/2018_VILTKAMERA_BACKUP_IS PUT IN TO IMPORT AND RUN TROUGH THE PROGRAM/**/*'
folder2019 = '/Volumes/CBreen/2019_VILTKAMERA_BACKUP_IS PUT IN TO IMPORT AND RUN TROUGH THE PROGRAM/**/*'
folder2020 = '/Volumes/CBreen/2020_VILTKAMERA_BACKUP_IS PUT IN TO IMPORT AN RUN TROUGH THE PROGRAM/**/*'

camerasTrondelag = pd.read_csv('/Users/cmbreen/Documents/Chapter 1/CV4EcologyData/AllScandCamcameras_CopyFeatures1_TableToExcel.csv', encoding_errors='ignore')
cameraList = camerasTrondelag['LokalitetID'].values.tolist()
cameraList

images2018 = glob.glob(folder2018)
images2019 = glob.glob(folder2019)
images2020 = glob.glob(folder2020)

all_images = images2018 + (images2019) + (images2020) ## 3660371

#rsync -rP /Volumes/CatBreen/CV4ecology/wyn1wyn2.csv dendrite:/datadrive/vmData/weather
idx = 0
cameraSorted = []
for file in all_images: 
    try: 
        filename = file.split('/')[-1]
        cameraID = filename.split('_')[0]
        ending = filename.split('.')[-1]
        if (int(cameraID) in cameraList):
            command = "rsync -rP '{file}' dendrite:/datadrive/vmData/weather/trondelag_allYears".format(file = file)
            print(command)
            idx+=1
            cameraSorted.append(cameraID)
            os.system(command)
    except Exception as e:
        #print(f"{e} {filename}")
        pass
    ##print(filename)
            
                
file = '/Volumes/CBreen/2020_VILTKAMERA_BACKUP_IS PUT IN TO IMPORT AN RUN TROUGH THE PROGRAM/3066/3066_20200201 (14).JPG'

os.system("some_command < input_file | another_command > output_file") 