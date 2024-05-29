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
from glob import glob
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, PrecisionRecallDisplay, precision_recall_curve
import matplotlib.pyplot as plt
from model import CustomResNet50
from train import create_dataloader, load_model 
import pandas as pd
import random
import IPython
from sklearn.metrics import balanced_accuracy_score, classification_report
import tqdm

## documentation for saving and loading models https://pytorch.org/tutorials/beginner/saving_loading_models.html

def eval_predictions(cfg, filename, image, true_label, prediction) : 
    """
    This function plots the regressed (predicted) keypoints and the actual 
    keypoints after each validation epoch for one image in the batch.
    'eval' is the method to check the model, whether is the valid data (eval) or test data (test)
    """
    # detach the image, keypoints, and output tensors from GPU to CPU
    save_path = cfg['output_path'] + '/' + cfg['exp_name'] +'/figs/outputs'
    #### make the path if it doesn't exist
    if not os.path.exists(save_path):  
        os.makedirs(save_path, exist_ok=True)
    image = image.detach().cpu()
    image = image.squeeze(0) ## drop the dimension because no longer need it for model 
    img = np.array(image, dtype='float32')
    img = np.transpose(img, (1, 2, 0))
    plt.imshow(img)
    plt.title(f'pred: {prediction}; true: {true_label}')
    plt.savefig(f'{save_path}/eval_{filename}')
    plt.close()

def load_model(cfg, epoch=None): ## what does epoch=None do in function? 
    '''
        Creates a model instance and loads the latest model state weights.
    '''
    # this is an empty model that we will load our model into
    #print(exp_name)
    model_instance = CustomResNet50(cfg['num_classes'])         # create an object instance of our CustomResNet18 class
    root = cfg['data_root']
    # load all model states
    #exp_name
    model_states = glob(cfg['output_path'] + '/' + cfg['exp_name'] + '/model_states/*.pt')
    ##glob('/datadrive/vmData/weather/experiments/exp_resnet50_2classes_seqSliding/model_states/*')
    #IPython.embed()
    #print(model_states)

    ## if there is more than one model state, take the most recent one
    if len(model_states) > 0:
        # at least one save state found; get latest
        root = cfg['output_path'] + '/' + cfg['exp_name']
        model_epochs = [int(m.replace((root)+'/model_states/','').replace('.pt','')) for m in model_states]
        ##### if statement that if you set the epoch in your function to evaluate from there
        ##### otherwise start at the most recent epoch
        if epoch:
            eval_epoch = str(epoch) ### if you set the epoch in the function, if it's none it will take the max
        else:
            eval_epoch = str(max(model_epochs))
        
        # load state dict and apply weights to model
        print(f'Evaluating from epoch {eval_epoch}')
        state = torch.load(open(f'{root}/model_states/{eval_epoch}.pt', 'rb'), map_location='cpu')  ### what is this doing? 
        model_instance.load_state_dict(state['model'])
        model_instance.eval()
        ### how do I get to a model?? 

    else:
        # no save state found; start anew
        print('No model found')

    return model_instance, epoch

    
########## extracting true_labels and predicted_labels for later use in the accuracy metrics
def predict(cfg, dataLoader, model):
    with torch.no_grad(): # no gradients needed for prediction
        filenames = []
        #predictions = [] ## predictions as tensor probabilites
        true_labels = [] ## labels as 0, 1 .. (classes)
        predicted_labels = [] ## labels as 0, 1 .. (classes)
        confidences0list = [] ## soft max of probabilities 
        confidences1list = []

        print(dataLoader.__len__())
        for idx in tqdm.tqdm(range(0, len(dataLoader.dataset.data))):
            data, label = dataLoader.dataset[idx]
            filename = dataLoader.dataset.data[idx][0]
            filenames.append(filename)

            true_label = label #.numpy()
            true_labels.append(true_label)

            data1 = data.unsqueeze(0) ## add dimension at the beginning because fake batch is 1
            prediction = model(data1) ## the full probabilty
            confidence = torch.nn.Softmax(dim=1)(prediction).detach().numpy() ## had to add .detach()

            confidence1 = confidence[:,1]
            confidences1list.extend(confidence1)
            if confidence1 > 0.5: predict_label = 1
            else: predict_label = 0 
            predicted_labels.append(predict_label)

            ## visualize the predictions on the images ## 
            eval_predictions(cfg, filename, data, true_label, predict_label) ## visualize points

        return filenames, true_labels, predicted_labels, confidences1list
   

def save_confusion_matrix(true_labels, predicted_labels, cfg, epoch='128'):
    # make figures folder if not there

    matrix_path = cfg['output_path'] + '/' + cfg['exp_name'] +'/figs'
    #### make the path if it doesn't exist
    if not os.path.exists(matrix_path):  
        os.makedirs(matrix_path, exist_ok=True)

    confmatrix = confusion_matrix(true_labels, predicted_labels)
    disp = ConfusionMatrixDisplay(confmatrix)
    #confmatrix.save(cfg['data_root'] + '/experiments/'+(args.exp_name)+'/figs/confusion_matrix_epoch'+'_'+ str(split) +'.png', facecolor="white")
    disp.plot()
    plt.savefig(cfg['output_path'] + '/' + cfg['exp_name']+'/figs/confusion_matrix_epoch'+'_TEST'+ str(epoch) +'.png', facecolor="white")
       ## took out epoch)
    return confmatrix

## we will calculate overall precision, recall, and F1 score
#def save_accuracy_metrics(y_true, y_pred, args, epoch, split):

    # make a csv of accuracy metrics 

def save_precision_recall_curve(true_labels, confidences, cfg, epoch='128'):
        #### make the path if it doesn't exist
    if not os.path.exists(cfg['output_path'] + '/' + cfg['exp_name']+'/figs'):
        os.makedirs(cfg['output_path'] + '/' + cfg['exp_name']+'/figs', exist_ok=True)
    

    true_labels_float = [float(x) for x in true_labels]
    PRcurve = PrecisionRecallDisplay.from_predictions(true_labels, confidences)
    PRcurve.plot()
    plt.savefig(cfg['output_path'] + '/' + cfg['exp_name']+'/figs/PRcurveTESTconfidences'+str(epoch) +'.png', facecolor="white")

def binaryMetrics(cfg, dl_val, model, epoch):
    print('generating binary predicted labels')
    filenames, true_labels, predicted_labels, confidences = predict(cfg, dl_val, model)   
    print('done generating predicted labels')
        
        # get accuracy score
        ### this is just a way to get two decimal places 
    acc = accuracy_score(true_labels, predicted_labels)
    print("Accuracy of model is {:0.2f}".format(acc))

        # confusion matrix
    confmatrix = save_confusion_matrix(true_labels, predicted_labels, cfg, epoch = epoch)
    print("confusion matrix saved")

    ######################### put this all in a function ##############
    # get precision score
    precision = precision_score(true_labels, predicted_labels)
    print("Precision of model is {:0.2f}".format(precision))

    recall = recall_score(true_labels, predicted_labels)
    print("Recall of model is {:0.2f}".format(recall))

    # get recall score
    F1score = f1_score(true_labels, predicted_labels)
    print("F1score of model is {:0.2f}".format(F1score))
    ######################################################################

    PRcurve = save_precision_recall_curve(true_labels, confidences, cfg, epoch = epoch)
    print("precision recall curve saved")

    metrics = pd.DataFrame({'precision':precision, 'recall':recall, 'F1score':F1score}, index=[0])
    metrics.to_csv(cfg['output_path'] + '/' + cfg['exp_name']+'/figs/'+'metricsTEST.csv')
    print("metrics csv saved")

    # save list of predictions
    results = pd.DataFrame({'filenames':filenames, 'trueLabels':true_labels, 'predictedLabels':predicted_labels, 'confidences':confidences})
    results.to_csv(cfg['output_path'] + '/' + cfg['exp_name']+'/figs/'+'resultsTEST'+str(epoch)+'.csv')
    print("results csv saved")


def main():
    # Argument parser for command-line arguments:
    # python code/train.py --output model_runs
    parser = argparse.ArgumentParser(description='Train deep learning model.')
    parser.add_argument('--config', help='Path to config file', default='configs/exp_resnet50_2classes.yaml')
    args = parser.parse_args()

    # reload config, that has to be set in the path
    print(f'Using config "{args.config}"')
    cfg = yaml.safe_load(open(args.config, 'r'))

    test_labels = pd.read_csv(cfg['test_labels'])

    # setup dataloader validation
    dl_val = create_dataloader(cfg, dataframe = test_labels, labels = cfg['test_labels']) ## labels is technically unused
    print('batches', dl_val.__len__())
    print('total size', dl_val.dataset.__len__())

    model, epoch = load_model(cfg)

    if cfg['num_classes'] == 2:
        print('calculating binary metrics')
        binaryMetrics(cfg, dl_val, model, epoch)

if __name__ == '__main__':
    main()