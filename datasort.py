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

#trainLabels = pd.DataFrame(columns = ['File', 'Weather','Date','Time','SnowCover','Temperature','location'])
trainfiles = []
trainweathers = []
traindate = []
traintime = []
trainsnowcover = []
traintemperature = []
trainlocation = []
#testLabels = pd.DataFrame(columns = ['File', 'Weather','Date','Time','SnowCover','Temperature','location'])
testfiles = []
testweathers = []
testdate = []
testtime = []
testsnowcover = []
testtemperature = []
testlocation = []


## sort all the data to the train folder and add to DF
def olympex_snoq_train_data(path):
    for file in glob.glob(path):
        print(file)
        filename = os.path.join(path,file)
        if os.path.splitext(file)[1].lower() in ('.jpg', '.jpeg'):
            name = file.split('/')[-1]
            #shutil.copy2(filename, trainDest+str('/')+name)

            ## subset df
            if len(labels[labels['File'] == name]) > 0:
                fileIndex = labels[labels['File'] == name].index.values.tolist()[0]
                weather = labels['Weather'][fileIndex]
                date = labels['Date'][fileIndex]
                time = labels['Time'][fileIndex]
                snowcover = labels['SnowCover'][fileIndex]
                temperature = labels ['Temperature'][fileIndex]
                location = labels['location'][fileIndex]
                #row = {'File': name, 'Date':date, 'Time':time, 'SnowCover':snowcover, 'Temperature':temperature, 'location':location}
                #print(row)
                #trainLabels.concat(row, ignore_index = True)
                trainfiles.append(name), trainweathers.append(weather), traindate.append(date)
                traintime.append(time), trainsnowcover.append(snowcover), traintemperature.append(temperature)
                trainlocation.append(location)

### get the image ID from the path. We will then use the filename 
### to look up the timestamp from the label dataframe 

### We look up the date it was taken looking up the index of the
### filename

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
                    #shutil.copy2(filename, testDest+str('/')+name)
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

##### 2017-2018 train data, 2019-2020 (lots of rain), and leave out 2018-2019 (sufficient)          
            else: 
                #shutil.copy2(filename, trainDest+str('/')+name)
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


olympex_snoq_train_data(olympex)
olympex_snoq_train_data(snoq)
norway_train_test(norway)

##use glob to get the label and file name (and merge with the )

## train folder and test folder statistics

train = ('/datadrive/data/weather/train/*')
test = ('/datadrive/data/weather/test/*')

print('Found {} images in train folder'.format(len(glob.glob(train))))
print('Found {} images in test folder'.format(len(glob.glob(test))))

print('done sorting')


### separate labels to train.csv and test.csv 
labels.head()
## now turn into dataframe 

trainLabels = pd.DataFrame({'File':trainfiles, 'Weather':trainweathers,'Date':traindate,'Time':traindate,'SnowCover':trainsnowcover,'Temperature':traintemperature,'location':trainlocation})
testLabels = pd.DataFrame({'File':testfiles, 'Weather':testweathers,'Date':testdate,'Time':testdate,'SnowCover':testsnowcover,'Temperature':testtemperature,'location':testlocation})

trainLabels.to_csv('/datadrive/data/trainLabels.csv')
testLabels.to_csv('/datadrive/data/testLabels.csv')
             