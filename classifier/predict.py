### might be able to just call model ???
'''
Evaluation metrics

From Sara:

What is my model doing? 
1. Write a prediction function! 
2. Make sure to include confidence scores
3. Logits?? (the output of the final layer before taking a softmax)
4. make a confusion matrix!! 
5. plot recision recall for each class
6. Find the most confusing images 
    - how do I do this without an index?
Catherine Breen
2022

'''

import yaml
import torch
import scipy
import numpy as np
import argparse
import os
import glob
import matplotlib.pyplot as plt
from model import CustomResNet50
from train import load_model 
import pandas as pd
import random
import IPython
import tqdm
from PIL import Image, ImageFile
import torch
from torchvision import transforms

## documentation for saving and loading models https://pytorch.org/tutorials/beginner/saving_loading_models.html

def load_model(num_of_classes, exp_name, epoch=None): ## what does epoch=None do in function? 
    '''
        Creates a model instance and loads the latest model state weights.
    '''
    # this is an empty model that we will load our model into
    #print(exp_name)
    model_instance = CustomResNet50(num_of_classes)         # create an object instance of our CustomResNet18 class
    # load all model states
   
    model_states = glob.glob(exp_name+'/*.pt')
 
    ## if there is more than one model state, take the most recent one
    if len(model_states) > 0:
    #     # at least one save state found; get latest
    #     IPython.embed()
    #     model_epochs = [int(m.replace((exp_name),'').replace('.pt','')) for m in model_states]
    #     ##### if statement that if you set the epoch in your function to evaluate from there
    #     ##### otherwise start at the most recent epoch
    #     if epoch:
    #         eval_epoch = str(epoch) ### if you set the epoch in the function, if it's none it will take the max
    #     else:
    #         eval_epoch = str(max(model_epochs))
        
        # load state dict and apply weights to model
        #print(f'Evaluating from epoch {eval_epoch}')
        state = torch.load(open('/Users/catherinebreen/Documents/Chapter 1/weather_model/exp_resnet50_2classes_None/119.pt', 'rb'), map_location='cpu')  ### what is this doing? 
        model_instance.load_state_dict(state['model'])
        model_instance.eval()
        ### how do I get to a model?? 

    else:
        # no save state found; start anew
        print('No model found')

    return model_instance

    
########## extracting true_labels and predicted_labels for later use in the accuracy metrics
def predict(num_of_classes, files, model):
    with torch.no_grad(): # no gradients needed for prediction
        filenames = []
        predicted_labels = [] ## labels as 0, 1 .. (classes)
        confidences0list = [] ## soft max of probabilities 
        confidences1list = []
        confidences2list = []
 
        print(len(files))

        for file in tqdm.tqdm(files):
            filename = file.split('/')[-1]
            filenames.append(filename)

            img = Image.open(file).convert('RGB')     # the ".convert" makes sure we always get three bands in Red, Green, Blue order
            #except: pass
            # transform: see lines 31ff above where we define our transformations
            convert_tensor = transforms.ToTensor()
            img_tensor = convert_tensor(img)

            data1 = img_tensor.unsqueeze(0) ## add dimension at the beginning because fake batch is 1
            prediction = model(data1) ## the full probabilty

            confidence = torch.nn.Softmax(dim=1)(prediction).detach().numpy() ## had to add .detach()

            #predicted_labels.extend(predict_label)
            #print(predict_label)
            #IPython.embed()
            if num_of_classes == 2:
                confidence1 = confidence[:,1]
                confidences1list.extend(confidence1)
                if confidence1 > 0.3: predict_label = 1
                else: predict_label = 0 
                predicted_labels.append(predict_label)

            if num_of_classes == 3:
                confidence0 = confidence[:,0]
                confidence1 = confidence[:,1]
                confidence2 = confidence[:,2]

                if confidence1 > 0.3: predict_label = 1
                else: predict_label = 0 
                predicted_labels.append(predict_label)

                confidences0list.extend(confidence0)
                confidences1list.extend(confidence1)
                confidences2list.extend(confidence2)
        
        if num_of_classes == 2: return filenames, predicted_labels, confidences1list
        else: return filenames, predicted_labels, confidences0list, confidences1list, confidences2list
   

def main():
    # Argument parser for command-line arguments:
    # python code/train.py --output model_runs
    parser = argparse.ArgumentParser(description='predict on deep learning model.')
    parser.add_argument('--exp_name', required=True, help='Path to experiment folder', default = "experiment_name")
    parser.add_argument('--images_folder', help='Path to test_folder', default='test_resized')
    args = parser.parse_args()

    # setup dataloader validation
    files = glob.glob(f"{args.images_folder}/*")
    print(len(files))

    model = load_model(2, args.exp_name)

    filenames, predicted_labels, confidences = predict(2, files, model)  
    #IPython.embed()
    results = pd.DataFrame({'filename':filenames, 'predicted_labels': predicted_labels, 'confidences': confidences})

    results.to_csv(f'{args.exp_name}/results_predictions_2018.csv')

if __name__ == '__main__':
    main()


'''
example parser argument: 

#2019
python predict.py --exp_name '/Users/catherinebreen/Documents/Chapter 1/weather_model/exp_resnet50_2classes_None' --images_folder '/Users/catherinebreen/Documents/Chapter 3/corresponding wetransfer images 2019'

#2018
python predict.py --exp_name '/Users/catherinebreen/Documents/Chapter 1/weather_model/exp_resnet50_2classes_None' --images_folder '/Users/catherinebreen/Documents/Chapter 3/corresponding wetransfer images 2018'

'''