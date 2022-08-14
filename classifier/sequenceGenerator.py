'''
There are 4 different type of sequences to try 
3-6hr: takes the first image between 0 and 6 hours
6-12hr: takes the first image between 6 and 12 hours
12-24hr: takes the first image between 12 and 24 hours
sliding: sliding window of the image right before and right after (time could possibly vary)


'sliding' has the caveat that the first and last image of the camera will 
not have a before and after image, respectively. In these cases, 
the before or the after image is replaced with the middle image,
which is also the image of interest. 


'''

import pandas as pd
import numpy as np

def sequenceGenerator(meta, file, sequenceType):
    
    ## subset dataframe for filename to get the location and other metadata
    fileIndex = meta[meta['File'] == file].index 
    date =  (meta['Date'][fileIndex].values.tolist())[0]
    location = (meta['location'][fileIndex].values.tolist())[0]

    ## subset dataframe of location
    cameraIDsubset = meta[meta['location'] == location]
    timestamp = pd.to_datetime(date)
    
    ## turn date column into datetime column to be able to find time deltas
    ### look at all the days that you have from that camera
    ### look at all the images that you have from that file 
    times = pd.to_datetime(cameraIDsubset['Date']) 
    files = cameraIDsubset['File']
    
    if sequenceType != 'sliding':
        before = []
        after = []
        rightBefore =[]
        rightAfter = []

        ##########We want a range of times for different experiments to test for accuracy
        startTimeInSeconds = { '3-6hr' : 10800 , '6-12hr' : 21600, '12-24hr' : 43200, }
        endTimeInSeconds = { '3-6hr' : 21600 , '6-12hr' : 43200, '12-24hr' : 86400 }

        ## find all the images that are close to it in time
        for (file, t) in zip(files, times):
            difference = timestamp - t
            diff = difference.total_seconds()  #days
            ######## first choice are images from ranges in dictionary
            #### before
            if (abs(diff) >= startTimeInSeconds[sequenceType]) and (abs(diff) <= endTimeInSeconds[sequenceType]): 
                if diff > 0: before.append(file)
                else: after.append(file)
            ### backups right before and right after
            elif (abs(diff) <= 43200): 
                if diff > 0: rightBefore.append(file)
                else: rightAfter.append(file) 

        #dictionary.update({filename: sequence})
        ## first choice list 
        after = sorted(after) 
        before = sorted(before)
        
        ## backup list 
        rightAfter = sorted(rightAfter) 
        rightBefore = sorted(rightBefore)

        if (len(before) > 0): finalBefore = before[0]
        elif (len(rightBefore)>0): finalBefore = rightBefore[0]
        else: finalBefore = file

        if (len(after) > 0): finalAfter = after[0]
        elif len(rightAfter)>0: finalAfter = rightAfter[0]
        else: finalAfter = file


    if sequenceType == 'sliding':
        #print(file)
        cameraIDsubset['Date'] = pd.to_datetime(cameraIDsubset['Date']) 
        cameraIDsubset = cameraIDsubset.sort_values(by='Date',ascending=True)
        cameraIDsubset = cameraIDsubset.drop(['level_0'], axis=1)
        cameraIDsubset = pd.DataFrame(cameraIDsubset).reset_index()
        #print(cameraIDsubset)
        ### find the file 
        slidingIndex = cameraIDsubset[cameraIDsubset['File'] == file].index.tolist()[0]  ### gets the value from the index
   
        if slidingIndex != 0:
            slidingBefore = cameraIDsubset['File'][slidingIndex-1] #.values.tolist() ## image right before, so use -1
        else: slidingBefore = file ## just use the same image 2x
        if slidingIndex != len(cameraIDsubset)-1: ## because len will be 226 but index goes up to 225
            #print(slidingIndex)
            #print(cameraIDsubset['File'])
            #print(cameraIDsubset['File'][1])
            #print('filename',cameraIDsubset['File'][slidingIndex+1])
            slidingAfter = cameraIDsubset['File'][slidingIndex+1] #.values.tolist()
        else: slidingAfter = file ## just use the same image 2x
        finalBefore = slidingBefore
        #print(finalBefore)
        finalAfter = slidingAfter
        #print(finalAfter)

    
    #print({filename:[finalBefore,finalAfter]})
    return (finalBefore,file,finalAfter)


