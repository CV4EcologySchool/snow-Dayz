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


## We need the cfg (model configuration), model 
### could make this a class rather than a function
### could make it a class of predictions ### 

def load_model(cfg, exp_name, epoch=None):
    '''
        Creates a model instance and loads the latest model state weights.
    '''
    model_instance = CustomResNet50(cfg['num_classes'])         # create an object instance of our CustomResNet18 class

    # load latest model state
    model_states = glob(exp_name+'/model_states/*.pt')

    if len(model_states) > 0:
        # at least one save state found; get latest
        model_epochs = [int(m.replace({exp_name}+'/model_states/','').replace('.pt','')) for m in model_states]
        if epoch:
            start_epoch = epoch
        else:
            start_epoch = max(model_epochs)

        # load state dict and apply weights to model
        print(f'Evaluating from epoch {start_epoch}')
        state = torch.load(open(f'{exp_name}/model_states/{start_epoch}.pt', 'rb'), map_location='cpu')
        model_instance.load_state_dict(state['model'])

        #import IPython
        #IPython.embed()

    else:
        # no save state found; start anew
        print('No model found')

def predict(cfg, dataLoader, model):
    with torch.no_grad(): # no gradients needed for prediction
        predictions = []
        predict_labels = [] 
        labels = []
        confidences = []
        ##### may need to adjust this in the dataloader for the sequence:
        for idx, (data, label) in enumerate(dataLoader): 
            prediction = model(data) ## the full probabilty
            predict_label = torch.argmax(prediction, dim=1) ## the label
            confidence = torch.nn.Softmax(prediction)

        predictions.append(prediction)
        predict_labels.append(int(predict_label))
        labels.append(int(label))
        confidences.append(int(confidence))

    return predictions, predict_labels, labels, confidence


def save_confusion_matrix(y_true, y_pred, args, epoch, split='train'):
    # make figures folder if not there
    os.makedirs({args.exp_dir}/{args.exp_name}+'/figs', exist_ok=True)

    confmatrix = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confmatrix)
    disp.plot()
    plt.savefig({'experiments/'{args.exp_name}+'/figs/confusion_matrix_epoch'+str(epoch)+'_'+split+'.png', facecolor="white")
    return confmatrix

## we will calculate overall precision, recall, and F1 score
#def save_accuracy_metrics(y_true, y_pred, args, epoch, split):

    # make a csv of accuracy metrics 

def main():
    # Argument parser for command-line arguments:
    # python code/train.py --output model_runs
    parser = argparse.ArgumentParser(description='Train deep learning model.')
    parser.add_argument('--exp_folder', required=True, help='Path to experiment folder')
    parser.add_argument('--split', help='Data split', default ='train')
    args = parser.parse_args()

    # set model directory
    exp_folder = args.exp_folder

    # get config from model directory
    config = glob(exp_folder+'*.yaml')[0]

    # load config
    print(f'Using config "{config}"')
    cfg = yaml.safe_load(open(config, 'r'))

    # setup dataloader
    dl_val = create_dataloader(cfg, split=args.split, batch=1)

    # load model and predict from model
    model, epoch = load_model(cfg)
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
    cm = save_confusion_matrix(labels, predict_labels, outdir, epoch, args.split)

    # precision recall curve

    # save list of predictions


if __name__ == '__main__':
    main()