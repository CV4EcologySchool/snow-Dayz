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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, PrecisionRecallDisplay
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

########## extracting true_labels and predicted_labels for later use in the accuracy metrics
def predict(cfg, dataLoader, model):
    with torch.no_grad(): # no gradients needed for prediction
        predictions = [] ## predictions as tensor probabilites
        true_labels = [] ## labels as 0, 1 .. (classes)
        predicted_labels = [] ## labels as 0, 1 .. (classes)
        confidences = [] ## soft max of probabilities 
        ##### may need to adjust this in the dataloader for the sequence:
        ### this will evaluate on each batch of data (usually 64)
        print('dataLoader')
        print(dataLoader)
        print(len(dataLoader)) ## number of total divisions n/batchsize
        for idx, (data, label) in enumerate(dataLoader): 
            if random.uniform(0.0, 1.0) <= 0.01:
                true_label = label
                prediction = model(data) ## the full probabilty
                print(prediction.shape) ## it is going to be [batch size #num_classes]
                predict_label = torch.argmax(prediction, dim=1) ## the label
                print(predict_label)
                confidence = torch.nn.Softmax(prediction)
            #print(confidence)

############ does this need to be before or after the torch.no_grad()
                true_label = true_label.tolist()
                true_labels.append(int(true_label))
                predictions.append(prediction)
                predicted_labels.append(int(predict_label))
                true_labels.append(int(label))
                print(true_labels)
                confidences.append(int(confidence))

    #### this should be full dataset as a dataframe
    results = pd.DataFrame({"true_labels": true_labels, "predict_label":predicted_labels, "confidence":confidences})

    return true_labels, predictions, predicted_labels, confidence, results

def export_results(results, exp_name):
    if not os.path.exists('experiments/'+(exp_name)+'/figs'):
        os.makedirs('experiments/'+(exp_name)+'/figs', exist_ok=True)

    results.to_csv('experiments/'+(exp_name)+'/figs/'+'results.csv')

def save_confusion_matrix(y_true, y_pred, exp_name, epoch, split='train'):
    # make figures folder if not there

    #### make the path if it doesn't exist
    if not os.path.exists('experiments/'+(exp_name)+'/figs'):
        os.makedirs('experiments/'+(exp_name)+'/figs', exist_ok=True)

    confmatrix = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confmatrix)
    disp.plot()
    plt.savefig(f'/experiments/'+(exp_name)+'/figs/confusion_matrix_epoch'+(epoch)+'_'+ str(split) +'.png', facecolor="white")
    return confmatrix

## we will calculate overall precision, recall, and F1 score
#def save_accuracy_metrics(y_true, y_pred, args, epoch, split):

    # make a csv of accuracy metrics 

def save_precision_recall_curve(y_true, y_pred, exp_name, epoch, split='train'):
        #### make the path if it doesn't exist
    if not os.path.exists('experiments/'+(exp_name)+'/figs'):
        os.makedirs('experiments/'+(exp_name)+'/figs', exist_ok=True)
    
    PRcurve = PrecisionRecallDisplay.from_predictions(y_true, y_pred)
    PRcurve.savefig(f'/experiments/'+(exp_name)+'/figs/PRcurve'+(epoch)+'_'+ str(split) +'.png', facecolor="white")


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

    # setup dataloader validation
    dl_val = create_dataloader(cfg, split='train', labels = 'trainLabels.csv', folder = 'train')

    # load model and predict from model
    model, epoch = load_model(cfg, exp_name)
    predictions, predicted_labels, true_labels, results = predict(cfg, dl_val, model)   
    
    # get accuracy score
    ### this is just a way to get two decimal places 
    acc = accuracy_score(true_labels, predicted_labels)
    print("Accuracy of model is {:0.2f}".format(acc))

    ######################### put this all in a function ##############
    # get precision score
    ### this is just a way to get two decimal places 
    precision = precision_score(true_labels, predicted_labels)
    print("Precision of model is {:0.2f}".format(precision))

    # get recall score
    ### this is just a way to get two decimal places 
    recall = recall_score(true_labels, predicted_labels)
    print("Recall of model is {:0.2f}".format(recall))

    # get recall score
    ### this is just a way to get two decimal places 
    F1score = f1_score(true_labels, predicted_labels)
    print("Recall of model is {:0.2f}".format(F1score))
    #####################################################

    # confusion matrix
    confmatrix = save_confusion_matrix(y_true=true_labels, y_pred=predicted_labels, exp_name = exp_name, epoch = epoch, split = 'train')
    print("confusion matrix saved")
    
    PRcurve = save_precision_recall_curve(y_true=true_labels, y_pred=predicted_labels, exp_name = exp_name, epoch = epoch, split = 'train')
    print("precision recall curve saved")

    # save list of predictions
    export_results = export_results(results, exp_name)


if __name__ == '__main__':
    main()