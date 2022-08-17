#datasort.py


import shutil
import pandas as pd
import glob 
import os 

def main():

    TestDate1 = pd.Timestamp('2018-01-01 00:00:00')
    TestDate2 = pd.Timestamp('2019-04-01 00:00:00')

    origin_folder = ('/datadrive/vmData/weather/train_resized/*')
    valDest = ('/datadrive/vmData/weather/val_resized')
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
    print("length of valLabels")
    print(len(valLabels))
    valLabels.to_csv('/datadrive/vmData/weather/valLabels.csv')



if __name__ == '__main__':
    # This block only gets executed if you call the "train.py" script directly
    # (i.e., "python ct_classifier/train.py").
    main()
