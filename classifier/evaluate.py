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
    - 
Catherine Breen
2022

'''

import yaml
import torch
import scipy
import numpy as np
import argparse
import os
from glob import glob
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, PrecisionRecallDisplay
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt
from model import CustomResNet50
from train import create_dataloader, load_model 
import pandas as pd
import random

## We need the cfg (model configuration), model 
### could make this a class rather than a function
### could make it a class of predictions ### 

def load_model(cfg, exp_name, epoch=None): ## what does epoch=None do in function? 
    '''
        Creates a model instance and loads the latest model state weights.
    '''
    # this is an empty model that we will load our model into
    print(exp_name)
    model_instance = CustomResNet50(cfg['num_classes'])         # create an object instance of our CustomResNet18 class

    # load all model states
    model_states = glob('experiments/'+exp_name+'/model_states/*.pt')

    ## if there is more than one model state, take the most recent one
    if len(model_states) > 0:
        # at least one save state found; get latest

        model_epochs = [int(m.replace('experiments/'+ (exp_name)+'/model_states/','').replace('.pt','')) for m in model_states]
        print(model_epochs) ## for debugging
        ##### if statement that if you set the epoch in your function to evaluate from there
        ##### otherwise start at the most recent epoch
        if epoch:
            eval_epoch = epoch ### if you set the epoch in the function, if it's none it will take the max
        else:
            eval_epoch = max(model_epochs)

        
        # load state dict and apply weights to model
        print(f'Evaluating from epoch {eval_epoch}')
        state = torch.load(open(f'experiments/{exp_name}/model_states/{eval_epoch}.pt', 'rb'), map_location='cpu')  ### what is this doing? 
        model = model_instance.load_state_dict(state['model'])
        print(model)
        ### how do I get to a model?? 

    else:
        # no save state found; start anew
        print('No model found')

    return model_instance, epoch

def predict(cfg, dataLoader, model):
    with torch.no_grad(): # no gradients needed for prediction
        filenIndex = []
        predictions = []
        predict_labels = [] 
        labels = []
        confidences = []
        ##### may need to adjust this in the dataloader for the sequence:
        for idx, (data, label) in enumerate(dataLoader): 
            if random.uniform(0.0, 1.0) <= 0.1:
                continue
            print(idx)
            prediction = model(data) ## the full probabilty
            print(prediction.shape) ## it is going to be [batch size #num_classes]
            predict_label = torch.argmax(prediction, dim=1) ## the label
            print(predict_label)
            confidence = torch.nn.Softmax(prediction)
            #print(confidence)

############ does this need to be before or after the torch.no_grad()
        predictions.append(prediction)
        predict_labels.append(int(predict_label))
        labels.append(int(label))
        print(labels)
        confidences.append(int(confidence))

    results = pd.DataFrame({"predict_label":predict_labels, "confidence":confidences})

    return predictions, predict_labels, labels, confidence


def save_confusion_matrix(y_true, y_pred, exp_name, epoch, split='train'):
    # make figures folder if not there
    os.makedirs({args.exp_dir}/{args.exp_name}+'/figs', exist_ok=True)

    confmatrix = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confmatrix)
    disp.plot()
    plt.savefig(f'/experiments/'+(exp_name)+'/figs/confusion_matrix_epoch'+(epoch)+'_'+ str(split) +'.png', facecolor="white")
    return confmatrix

## we will calculate overall precision, recall, and F1 score
#def save_accuracy_metrics(y_true, y_pred, args, epoch, split):

    # make a csv of accuracy metrics 

def main():
    # Argument parser for command-line arguments:
    # python code/train.py --output model_runs
    parser = argparse.ArgumentParser(description='Train deep learning model.')
    parser.add_argument('--exp_name', required=True, help='Path to experiment folder', default = "experiment_name")
    parser.add_argument('--split', help='Data split', default ='train')
    parser.add_argument('--config', help='Path to config file', default='configs/exp_resnet50_2classes.yaml')
    args = parser.parse_args()

    # set model directory
    exp_name = args.exp_name

    # reload config, that has to be set in the path
    print(f'Using config "{args.config}"')
    cfg = yaml.safe_load(open(args.config, 'r'))

    # setup dataloader
    dl_val = create_dataloader(cfg, split='train', labels = 'trainLabels.csv', folder = 'train')

    # load model and predict from model
    model, epoch = load_model(cfg, exp_name)
    predictions, predict_labels, labels = predict(cfg, dl_val, model)   
    
    # get accuracy score
    ### this is just a way to get two decimal places 
    acc = accuracy_score(labels, predict_labels)
    print("Accuracy of model is {:0.2f}".format(acc))

    # get precision score
    ### this is just a way to get two decimal places 
    precision = accuracy_score(labels, predict_labels)
    print("Precision of model is {:0.2f}".format(acc))

    # get recall score
    ### this is just a way to get two decimal places 
    recall = accuracy_score(labels, predict_labels)
    print("Recall of model is {:0.2f}".format(acc))

    # get recall score
    ### this is just a way to get two decimal places 
    F1score = accuracy_score(labels, predict_labels)
    print("Recall of model is {:0.2f}".format(acc))

    # confusion matrix
    cm = save_confusion_matrix(labels, predict_labels, exp_name, epoch, args.split)

    # precision recall curve

    # save list of predictions


if __name__ == '__main__':
    main()