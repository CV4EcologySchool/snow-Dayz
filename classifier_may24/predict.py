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

def save_image(filename, image) : 
    """
    This function plots the regressed (predicted) keypoints and the actual 
    keypoints after each validation epoch for one image in the batch.
    'eval' is the method to check the model, whether is the valid data (eval) or test data (test)
    """
    # detach the image, keypoints, and output tensors from GPU to CPU
    save_path = '/Volumes/CatBreen/CV4ecology/WEATHER/Chewelah_weather_only'
    #### make the path if it doesn't exist
    if not os.path.exists(save_path):  
        os.makedirs(save_path, exist_ok=True)
    image = image.detach().cpu()
    image = image.squeeze(0) ## drop the dimension because no longer need it for model 
    img = np.array(image, dtype='float32')
    # If img is in the range [0, 1], scale it to [0, 255]
    img = np.clip(img, 0, 1) * 255.0
    img = img.astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))
    img = Image.fromarray(img)
    resized_image = img.resize((448, 448))
    resized_image.save(f'{save_path}/{filename}')
    print('filesaved')

def load_model(num_of_classes, exp_name, epoch=None): ## what does epoch=None do in function? 
    '''
        Creates a model instance and loads the latest model state weights.
    '''

    model_instance = CustomResNet50(num_of_classes)         # create an object instance of our CustomResNet18 class
    # load all model states
   
    model_states = glob.glob(exp_name) #+'/*.pt')
 
    ## if there is more than one model state, take the most recent one
    if len(model_states) > 0:
        state = torch.load(open('/Users/catherinebreen/Dropbox/Chapter4/WEATHER_MODEL/classifier_results/baseline/model_states/29.pt', 'rb'), map_location='cpu')  ### what is this doing? 
        model_instance.load_state_dict(state['model'])
        model_instance.eval()

    else:
        # no save state found; start anew
        print('No model found')

    return model_instance

def predict(num_of_classes, files, model):
    with torch.no_grad(): # no gradients needed for prediction
        filenames = []
        predicted_labels = [] ## labels as 0, 1 .. (classes)
        confidences1list = []
 
        print(len(files))

        for file in tqdm.tqdm(files):
            cameraID = file.split('/')[-2]
            filename = cameraID + '_' + file.split('/')[-1]
            filenames.append(filename)

            img = Image.open(file).convert('RGB')     # the ".convert" makes sure we always get three bands in Red, Green, Blue order
            #except: pass
            # transform: see lines 31ff above where we define our transformations
            convert_tensor = transforms.ToTensor()
            img_tensor = convert_tensor(img)

            data1 = img_tensor.unsqueeze(0) ## add dimension at the beginning because fake batch is 1
            prediction = model(data1) ## the full probabilty

            confidence = torch.nn.Softmax(dim=1)(prediction).detach().numpy() ## had to add .detach()

            confidence1 = confidence[:,1]
            confidences1list.extend(confidence1)
            if confidence1 > 0.8: 
                predict_label = 1
                save_image(filename, data1)
            else: predict_label = 0 
            predicted_labels.append(predict_label)

        return filenames, predicted_labels, confidences1list
   
   
   

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
    IPython.embed()
    results.to_csv(f'result_predictions.csv')

if __name__ == '__main__':
    main()


'''
example parser argument: 


python classifier_may24/predict.py --exp_name '/Users/catherinebreen/Dropbox/Chapter4/WEATHER_MODEL/classifier_results/baseline/model_states/29.pt' --images_folder '/Volumes/CatBreen/Chelewah_Timelapse_Photos/**'




#2019
python predict.py --exp_name '/Users/catherinebreen/Documents/Chapter 1/weather_model/exp_resnet50_2classes_None' --images_folder '/Users/catherinebreen/Documents/Chapter 3/corresponding wetransfer images 2019'

#2018
python predict.py --exp_name '/Users/catherinebreen/Documents/Chapter 1/weather_model/exp_resnet50_2classes_None' --images_folder '/Users/catherinebreen/Documents/Chapter 3/corresponding wetransfer images 2018'

## 2023 field data: 
python predict.py --exp_name '/Users/catherinebreen/Documents/Chapter1/weather_model/exp_resnet50_2classes_None/119.pt' --images_folder '/Users/catherinebreen/Documents/Chapter3/fieldwork/TRO_S_K0060'
python predict.py --exp_name '/Users/catherinebreen/Documents/Chapter1/weather_model/exp_resnet50_2classes_None/119.pt' --images_folder '/Users/catherinebreen/Documents/Chapter3/fieldwork/TRO_N_K0158'
python predict.py --exp_name '/Users/catherinebreen/Documents/Chapter1/weather_model/exp_resnet50_2classes_None/119.pt' --images_folder '/Users/catherinebreen/Documents/Chapter3/fieldwork/OSL_S_K0076'
python predict.py --exp_name '/Users/catherinebreen/Documents/Chapter1/weather_model/exp_resnet50_2classes_None/119.pt' --images_folder '/Users/catherinebreen/Documents/Chapter3/fieldwork/OSL_N_K0381'
python predict.py --exp_name '/Users/catherinebreen/Documents/Chapter1/weather_model/exp_resnet50_2classes_None/119.pt' --images_folder '/Users/catherinebreen/Documents/Chapter3/fieldwork/IN_S_K0077'
python predict.py --exp_name '/Users/catherinebreen/Documents/Chapter1/weather_model/exp_resnet50_2classes_None/119.pt' --images_folder '/Users/catherinebreen/Documents/Chapter3/fieldwork/IN_N_K0380'

#snowpole data
python predict.py --exp_name '/Users/catherinebreen/Documents/Chapter1/weather_model/exp_resnet50_2classes_None/119.pt' --images_folder '/Volumes/CatBreen/Okanagan_Timelapse_Photos/**'
python predict.py --exp_name '/Users/catherinebreen/Documents/Chapter1/weather_model/exp_resnet50_2classes_None/119.pt' --images_folder '/Volumes/CatBreen/Chelewah_Timelapse_Photos/**'
python predict.py --exp_name '/Users/catherinebreen/Documents/Chapter1/weather_model/exp_resnet50_2classes_None/119.pt' --images_folder '/Volumes/CatBreen/LabeledData_Wynoochee/all_Cameras/**'

#scandcam images


'''