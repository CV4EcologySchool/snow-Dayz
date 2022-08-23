## bash script to upload Trondelag cameras 
import os
import glob

folder2018 = '/Volumes/CBreen/2018_VILTKAMERA_BACKUP_IS PUT IN TO IMPORT AND RUN TROUGH THE PROGRAM/**/*'
folder2019 = '/Volumes/CBreen/2019_VILTKAMERA_BACKUP_IS PUT IN TO IMPORT AND RUN TROUGH THE PROGRAM/**/*'
folder2020 = '/Volumes/CBreen/2020_VILTKAMERA_BACKUP_IS PUT IN TO IMPORT AN RUN TROUGH THE PROGRAM/**/*'

camerasTrondelag = pd.read_csv('/Users/cmbreen/Documents/Chapter 1/CV4EcologyData/AllScandCamcameras_CopyFeatures1_TableToExcel.csv', encoding_errors='ignore')
cameras = camerasTrondelag['LokalitetID'].values.tolist()
cameras

images2018 = glob.glob(folder2018)
images2019 = glob.glob(folder2019)
images2020 = glob.glob(folder2020)

rsync -rP /Volumes/CatBreen/CV4ecology/wyn1wyn2.csv dendrite:/datadrive/vmData/weather


os.system("some_command < input_file | another_command > output_file") 