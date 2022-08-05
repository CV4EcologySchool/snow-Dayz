#### Catherine Breen

print('start')


import shutil
import pandas as pd
import glob 
import os 

##### sort labels into train and test
labels = pd.read_csv('/datadrive/data/all_labels_QC.csv')

####### sort images into train and test folders #####

norway = ('/datadrive/data/scandcam/**/*')
olympex = ('/datadrive/data/olympex/**/*')
snoq = ('/datadrive/data/snoq/**/*')

trainDest = ('/datadrive/data/weather/train')
testDest = ('/datadrive/data/weather/test')

if not os.path.exists(trainDest):
    os.makedirs(trainDest)

if not os.path.exists(testDest):
    os.makedirs(testDest)

print('Path Statistics:\n')

print('Found {} images in scandcam folder'.format(len(glob.glob(norway))))
print('Found {} images in olympex folder'.format(len(glob.glob(olympex))))
print('Found {} images in snoq folder'.format(len(glob.glob(snoq))))

## load all the data 
def olympex_snoq_train_data(path):
    for file in glob.glob(path):
        print(file)
        filename = os.path.join(path,file)
        if os.path.splitext(file)[1].lower() in ('.jpg', '.jpeg'):
            name = file.split('/')[-1]
            shutil.copy2(filename, trainDest+str('/')+name)
    

### get the image ID from the path. We will then use the filename 
### to look up the timestamp from the label dataframe 

### We look up the date it was taken looking up the index of the
### filename

TestDate1 = pd.Timestamp('2018-09-30 00:00:00')
TestDate2 = pd.Timestamp('2019-04-01 00:00:00')

def norway_train_test(path):
    for file in glob.glob(path):
        if os.path.splitext(file)[1].lower() in ('.jpg', '.jpeg'):
            
            filename = file.split('/')[-1]  
            fileIndex = labels[labels['File'] == filename].index
            date = (labels['Date'][fileIndex].values.tolist())[0] ## gets the date and turns it into a string
            
            ## convert date to proper timestamp
            date = pd.to_datetime(date)

##### 2018-2019 year of interest [October 1 2018 - April 2019] 
### ecological justification: rain vs. snow is the year we have full dataset (because of covid). I had to leave in March
            if (date >= TestDate1) & (date <= TestDate2):
                    shutil.copy2(filename, testDest+str('/')+filename)

##### 2017-2018 train data, 2019-2020 (lots of rain), and leave out 2018-2019 (sufficient)          
            else: shutil.copy2(filename, trainDest+str('/')+filename)



olympex_snoq_train_data(olympex)
olympex_snoq_train_data(snoq)
norway_train_test(norway)

##use glob to get the label and file name (and merge with the )

## train folder and test folder statistics

train = ('/datadrive/data/weather/train/*')
test = ('/datadrive/data/weather/test/*')

print('Found {} images in train folder'.format(len(glob.glob(train))))
print('Found {} images in test folder'.format(len(glob.glob(test))))

print('done')