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

    model_instance = CustomResNet50(num_of_classes)         # create an object instance of our CustomResNet18 class
    # load all model states
    
    state = torch.load(open(f'/Users/catherinebreen/Dropbox/Chapter1/weather_model/exp_resnet50_2classes_None/119.pt', 'rb'), map_location='cpu')  ### what is this doing? 
    model_instance.load_state_dict(state['model'])
    model_instance.eval()

    return model_instance

# def predict(num_of_classes, files, model):
#     with torch.no_grad(): # no gradients needed for prediction
#         filenames = []
#         predicted_labels = [] ## labels as 0, 1 .. (classes)
#         confidences1list = []
 
#         print(len(files))

#         for file in tqdm.tqdm(files):
#             cameraID = file.split('/')[-2]
#             # if np.int(cameraID) < 6: 
#             filename = cameraID + '_' + file.split('/')[-1]
#             filenames.append(filename)

#             img = Image.open(file).convert('RGB')     # the ".convert" makes sure we always get three bands in Red, Green, Blue order
#             #except: pass
#             # transform: see lines 31ff above where we define our transformations
#             convert_tensor = transforms.ToTensor()
#             img_tensor = convert_tensor(img)

#             data1 = img_tensor.unsqueeze(0) ## add dimension at the beginning because fake batch is 1
#             prediction = model(data1) ## the full probabilty

#             confidence = torch.nn.Softmax(dim=1)(prediction).detach().numpy() ## had to add .detach()

#             confidence1 = confidence[:,1]
#             confidences1list.extend(confidence1)
#             if confidence1 > 0.5: 
#                 predict_label = 1
#                 #save_image(filename, data1)
#             else: predict_label = 0 
#             predicted_labels.append(predict_label)
#             # else: pass

#         return filenames, predicted_labels, confidences1list
   
def predict(num_of_classes, files, model, device='cpu', batch_size=2):
    with torch.no_grad():
        filenames = []
        predicted_labels = []
        confidences1list = []
        convert_tensor = transforms.ToTensor()
        for i in tqdm.tqdm(range(0, len(files), batch_size)):
            batch_files = files[i:i+batch_size]
            images = []
            batch_filenames = []
            for file in batch_files:
                cameraID = file.split('/')[-2]
                filename = cameraID + '_' + file.split('/')[-1]
                batch_filenames.append(filename)
                img = Image.open(file).convert('RGB')
                img_tensor = convert_tensor(img).unsqueeze(0).to(device)  # Move image tensor to GPU
                images.append(img_tensor)

            batch_data = torch.cat(images)
            predictions = model(batch_data)
            confidences = torch.nn.Softmax(dim=1)(predictions).cpu().detach().numpy()  # Bring back to CPU if needed
            confidence1 = confidences[:, 1]
            confidences1list.extend(confidence1)
            predict_labels = (confidence1 > 0.5).astype(int)
            predicted_labels.extend(predict_labels)
            filenames.extend(batch_filenames)


def main():
    # Argument parser for command-line arguments:
    # python code/train.py --output model_runs
    parser = argparse.ArgumentParser(description='predict on deep learning model.')
    parser.add_argument('--exp_name', required=True, help='Path to experiment folder', default = "experiment_name")
    parser.add_argument('--images_folder', help='Path to test_folder', default='test_resized')
    args = parser.parse_args()

    # setup dataloader validation
    files = glob.glob(f"{args.images_folder}/*.JPG")
    print(len(files))

    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    print(device)
    model = load_model(2, args.exp_name)

    filenames, predicted_labels, confidences = predict(2, files, model)  
    #IPython.embed()
    results = pd.DataFrame({'filename':filenames, 'predicted_labels': predicted_labels, 'confidences': confidences})
    #IPython.embed()
    results.to_csv(f'/Users/catherinebreen/Documents/TEST/scandcam2018_exp_resnet50_2classes_None.csv')

if __name__ == '__main__':
    main()


'''
example parser argument: 


#scandcam images
python predict.py --exp_name '/Users/catherinebreen/Documents/Chapter1/weather_model/exp_resnet50_2classes_None' --images_folder '/Volumes/CBreen/2018_VILTKAMERA_BACKUP_IS PUT IN TO IMPORT AND RUN TROUGH THE PROGRAM/**'


'''