#### Catherine Breen
### can run as python datasort.py in snow-Dayz folder on VM 
### or run as ipython and copy and paste 


print('start')


import shutil
import pandas as pd
import glob 
import os 

##### sort labels into train and test
labels = pd.read_csv('/datadrive/data/all_labels_QC.csv')
labels = labels[['File', 'Weather','Date','Time','SnowCover','Temperature','location']]

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

indices_train = []
## sort all the data to the train folder and add to DF
def olympex_snoq_train_data(path):
    for file in glob.glob(path):
        print(file)
        filename = os.path.join(path,file)
        if os.path.splitext(file)[1].lower() in ('.jpg', '.jpeg'):
            name = file.split('/')[-1]
            ## subset df
            if len(labels[labels['File'] == name]) > 0:
                fileIndex = labels[labels['File'] == name].index.values.tolist()[0]
                indices_train.append(fileIndex)
                shutil.copy2(filename, trainDest+str('/')+name)


## shoud be the same as the number 

TestDate1 = pd.Timestamp('2018-09-30 00:00:00')
TestDate2 = pd.Timestamp('2019-04-01 00:00:00')

def norway_train_test(path):
    for file in glob.glob(path):
        if os.path.splitext(file)[1].lower() in ('.jpg', '.jpeg'):
            filename = os.path.join(path,file)
            name = file.split('/')[-1]  
            fileIndex = labels[labels['File'] == name].index
            date = (labels['Date'][fileIndex].values.tolist())[0] ## gets the date and turns it into a string
            
            ## convert date to proper timestamp
            date = pd.to_datetime(date)

##### 2018-2019 year of interest [October 1 2018 - April 2019] 
### ecological justification: rain vs. snow is the year we have full dataset (because of covid). I had to leave in March
            if (date >= TestDate1) & (date <= TestDate2):
                    print(filename)
                    if len(labels[labels['File'] == name]) > 0:
                        fileIndex = labels[labels['File'] == name].index.values.tolist()[0]
                        weather = labels['Weather'][fileIndex]
                        date = labels['Date'][fileIndex]
                        time = labels['Time'][fileIndex]
                        snowcover = labels['SnowCover'][fileIndex]
                        temperature = labels ['Temperature'][fileIndex]
                        location = labels['location'][fileIndex]
                        trainfiles.append(name), trainweathers.append(weather), traindate.append(date)
                        traintime.append(time), trainsnowcover.append(snowcover), traintemperature.append(temperature)
                        trainlocation.append(location)
                        shutil.copy2(filename, testDest+str('/')+name)

##### 2017-2018 train data, 2019-2020 (lots of rain), and leave out 2018-2019 (sufficient)          
            else: 
                if len(labels[labels['File'] == name]) > 0:
                    fileIndex = labels[labels['File'] == name].index.values.tolist()[0]
                    weather = labels['Weather'][fileIndex]
                    date = labels['Date'][fileIndex]
                    time = labels['Time'][fileIndex]
                    snowcover = labels['SnowCover'][fileIndex]
                    temperature = labels ['Temperature'][fileIndex]
                    location = labels['location'][fileIndex]
                    testfiles.append(name), testweathers.append(weather), testdate.append(date)
                    testtime.append(time), testsnowcover.append(snowcover), testtemperature.append(temperature), testlocation.append(location)
                    shutil.copy2(filename, trainDest+str('/')+name)


indices_train = []
indices_test = []
olympex_snoq_train_data(olympex)
olympex_snoq_train_data(snoq)
norway_train_test(norway)

##use glob to get the label and file name (and merge with the )

## train folder and test folder statistics

train = ('/datadrive/data/weather/train/*')
test = ('/datadrive/data/weather/test/')

print('Found {} images in train folder'.format(len(glob.glob(train))))
print('Found {} images in test folder'.format(len(glob.glob(test))))

print('done sorting')


### separate labels to train.csv and test.csv 
labels.head()
## now turn into dataframe 

trainLabels = pd.DataFrame({'File':trainfiles, 'Weather':trainweathers,'Date':traindate,'Time':traindate,'SnowCover':trainsnowcover,'Temperature':traintemperature,'location':trainlocation})
testLabels = pd.DataFrame({'File':testfiles, 'Weather':testweathers,'Date':testdate,'Time':testdate,'SnowCover':testsnowcover,'Temperature':testtemperature,'location':testlocation})

print('Found {} images in train labels'.format(len(trainLabels['File'])))
print('Found {} images in test labels'.format(len(testLabels['File'])))
print('Found {} images in all labels'.format(len(labels['File'])))


### need this to be in root '/datadrive/data/weather' because that is going to the root for this model(the weather model)

trainLabels.to_csv('/datadrive/data/weather/trainLabels.csv')
testLabels.to_csv('/datadrive/data/weather/testLabels.csv')
             
#########
### 

## do a check that length of 

trainDest = ('/datadrive/data/weather/train/*')
testDest = ('/datadrive/data/weather/test/*')
trainLabels = pd.read_csv('/datadrive/data/weather/trainLabels.csv')
testLabels = pd.read_csv('/datadrive/data/weather/testLabels.csv')

print('Found {} images in train folder'.format(len(glob.glob(trainDest))))
print('Found {} images in test folder'.format(len(glob.glob(testDest))))

print('Found {} images in train labels'.format(len(trainLabels['File'])))
print('Found {} images in test labels'.format(len(testLabels['File'])))


FilesThatExist = []
trainIndices = []
FilesThatDontExist = []
for file in glob.glob(trainDest):
        #print(file)
        #filename = os.path.join(trainDest,file)
        #if os.path.splitext(file)[1].lower() in ('.jpg', '.jpeg'):
        name = file.split('/')[-1]
       #print(name)
        if len(labels[labels['File'] == name]) >0:
            FileIndex = labels[labels['File']==name].index.tolist()[0]
            trainIndices.append(FileIndex)
        else: FilesThatDontExist.append(file)



testIndices = []
for file in glob.glob(testDest):
        #print(file)
        #filename = os.path.join(trainDest,file)
        #if os.path.splitext(file)[1].lower() in ('.jpg', '.jpeg'):
        name = file.split('/')[-1]
       #print(name)
        if len(labels[labels['File'] == name]) >0:
            FileIndex = labels[labels['File']==name].index.tolist()[0]
            testIndices.append(FileIndex)


testLabels = labels.loc[testIndices].reset_index()
testLabels.to_csv('/datadrive/data/weather/testLabels.csv')
testLabels = pd.read_csv('/datadrive/data/weather/testLabels.csv')
test.head()

trainLabels = labels.loc[trainIndices].reset_index()
trainLabels.to_csv('/datadrive/data/weather/trainLabels.csv')
trainLabels = pd.read_csv('/datadrive/data/weather/trainLabels.csv')

for file in FilesThatDontExist:
    if os.path.isfile(file):
        os.remove(file)
    else: print("Error: %s file not found" % file)



########################################
## val script

TestDate1 = pd.Timestamp('2018-01-01 00:00:00')
TestDate2 = pd.Timestamp('2019-04-01 00:00:00')

origin_folder = ('/datadrive/vmData/weather/train_resized/*')
valDest = ('/datadrive/vmData/weather/val_resized/')
labels = pd.read_csv('/datadrive/vmData/weather/trainLabels.csv')
labels = labels[['File', 'Weather','Date','Time','SnowCover','Temperature','location']]

valfiles = []
valweathers = []
valdate = []
valtime = []
valsnowcover = []
valtemperature = []
vallocation = []

def train_sort(path):
    for file in glob.glob(path):
        if os.path.splitext(file)[1].lower() in ('.jpg', '.jpeg'):
            filename = os.path.join(path,file)
            name = file.split('/')[-1]  
            fileIndex = labels[labels['File'] == name].index
            location = (labels['Date'][fileIndex].values.tolist())[0] ## gets location and turns it into a string
            date = (labels['Date'][fileIndex].values.tolist())[0] ## gets the date and turns it into a string
            ## convert date to proper timestamp
            date = pd.to_datetime(date)
##### 2018-2019 year of interest [Jan 1 2017 - April 2018] 
### ecological justification: rain vs. snow is the year we have full dataset (because of covid). I had to leave in March
            if (date >= TestDate1) & (date <= TestDate2):
                if len(labels[labels['File'] == name]) > 0:
                    fileIndex = labels[labels['File'] == name].index.values.tolist()[0]
                    weather = labels['Weather'][fileIndex]
                    date = labels['Date'][fileIndex]
                    time = labels['Time'][fileIndex]
                    snowcover = labels['SnowCover'][fileIndex]
                    temperature = labels ['Temperature'][fileIndex]
                    location = labels['location'][fileIndex]
                    valfiles.append(name), valweathers.append(weather), valdate.append(date)
                    valtime.append(time), valsnowcover.append(snowcover), valtemperature.append(temperature), vallocation.append(location)
                    print(filename)
                    shutil.move(filename, valDest+str('/')+name)

train_sort(origin_folder)
valLabels = pd.DataFrame({'File':valfiles, 'Weather':valweathers,'Date':valdate,'Time':valdate,'SnowCover':valsnowcover,'Temperature':valtemperature,'location':vallocation})
valLabels.to_csv('/datadrive/vmData/weather/valLabels.csv')