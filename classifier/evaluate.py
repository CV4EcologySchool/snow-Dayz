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


## We need the cfg (model configuration), model 
### could make this a class rather than a function
### could make it a class of predictions ### 

## documentation for saving and loading models https://pytorch.org/tutorials/beginner/saving_loading_models.html

def load_model(cfg, exp_dir, exp_name, epoch=None): ## what does epoch=None do in function? 
    '''
        Creates a model instance and loads the latest model state weights.
    '''
    # this is an empty model that we will load our model into
    #print(exp_name)
    model_instance = CustomResNet50(cfg['num_classes'])         # create an object instance of our CustomResNet18 class
    root = cfg['data_root']
    # load all model states
    #exp_name
    model_states = glob(root+'/'+exp_dir+'/'+exp_name+'/model_states/*.pt')
    ##glob('/datadrive/vmData/weather/experiments/exp_resnet50_2classes_seqSliding/model_states/*')
    #IPython.embed()
    #print(model_states)

    ## if there is more than one model state, take the most recent one
    if len(model_states) > 0:
        # at least one save state found; get latest

        model_epochs = [int(m.replace((root)+'/'+exp_dir+'/'+ (exp_name)+'/model_states/','').replace('.pt','')) for m in model_states]
        ##### if statement that if you set the epoch in your function to evaluate from there
        ##### otherwise start at the most recent epoch
        if epoch:
            eval_epoch = str(epoch) ### if you set the epoch in the function, if it's none it will take the max
        else:
            eval_epoch = str(max(model_epochs))
        
        # load state dict and apply weights to model
        print(f'Evaluating from epoch {eval_epoch}')
        state = torch.load(open(f'{root}/{exp_dir}/{exp_name}/model_states/{eval_epoch}.pt', 'rb'), map_location='cpu')  ### what is this doing? 
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
        confidences0 = [] ## soft max of probabilities 
        confidences1 = []
        confidences2 = []
        #confidences3 = []
        #confidences4 = []
        ##### may need to adjust this in the dataloader for the sequence:
        ### this will evaluate on each batch of data (usually 64)
        #IPython.embed()
        #print(len(dataLoader)) ## number of total divisions n/batchsize
        # for idx, (data, label) in enumerate(dataLoader): 

        #     true_label = label.numpy()
        #     true_labels.extend(true_label)

        #     prediction = model(data) ## the full probabilty
        #     predictions.append(prediction)
        #     #print(prediction.shape) ## it is going to be [batch size #num_classes]
            
        #     ## predictions
        #     predict_label = torch.argmax(prediction, dim=1).numpy() ## the label
        #     predicted_labels.extend(predict_label)
        #     #print(predict_label)

        #     confidence = torch.nn.Softmax(dim=1)(prediction).numpy()
        #     confidence = confidence[:,1]
        #     confidences.extend(confidence)

    # true_labels = np.array(true_labels)
    # predicted_labels = np.array(predicted_labels)
    # confidences = np.array(confidences)
        print(dataLoader.__len__())
        for idx in range(0, len(dataLoader.dataset.data)):
            data, label = dataLoader.dataset[idx]
            filename = dataLoader.dataset.data[idx][0]
            filenames.append(filename)

            true_label = label #.numpy()
            true_labels.append(true_label)

            data1 = data.unsqueeze(0) ## add dimension at the beginning because fake batch is 1
            prediction = model(data1) ## the full probabilty
            #predictions.append(prediction)
            #print(prediction.shape) ## it is going to be [batch size #num_classes]
            print(prediction)
            ## predictions
            #IPython.embed()
            predict_label = torch.argmax(prediction, dim=1).numpy() ## the label
            predicted_labels.extend(predict_label)
            #print(predict_label)

            confidence = torch.nn.Softmax(dim=1)(prediction).detach().numpy() ## had to add .detach()
            
            if cfg['num_classes'] == 2:
                confidence1 = confidence[:,1]
                confidences1.extend(confidence1)

            if cfg['num_classes'] == 3:
                confidence0 = confidence[:,0]
                confidence1 = confidence[:,1]
                confidence2 = confidence[:,2]

                confidences0.extend(confidence0)
                confidences1.extend(confidence1)
                confidences2.extend(confidence2)
        
        if cfg['num_classes'] == 2: return filenames, true_labels, predicted_labels, confidences1
        else: return filenames, true_labels, predicted_labels, confidences0, confidences1, confidences2
   

    #print(predicted_labels)
    #print(len(predicted_labels))
    #### this should be full dataset as a dataframe
    #results = pd.DataFrame({"true_labels": true_labels, "predict_label":predicted_labels}) #"confidence":confidence


def save_confusion_matrix(true_labels, predicted_labels, cfg, args, epoch='128', split='train'):
    # make figures folder if not there

    matrix_path = cfg['data_root']+'/' + args.exp_dir +'/'+(args.exp_name)+'/figs'
    #### make the path if it doesn't exist
    if not os.path.exists(matrix_path):  
        os.makedirs(matrix_path, exist_ok=True)

    confmatrix = confusion_matrix(true_labels, predicted_labels)
    disp = ConfusionMatrixDisplay(confmatrix)
    #confmatrix.save(cfg['data_root'] + '/experiments/'+(args.exp_name)+'/figs/confusion_matrix_epoch'+'_'+ str(split) +'.png', facecolor="white")
    disp.plot()
    plt.savefig(cfg['data_root'] + '/' + (args.exp_dir) + '/'+(args.exp_name)+'/figs/confusion_matrix_epoch'+'_TEST'+ str(epoch) +'.png', facecolor="white")
       ## took out epoch)
    return confmatrix

## we will calculate overall precision, recall, and F1 score
#def save_accuracy_metrics(y_true, y_pred, args, epoch, split):

    # make a csv of accuracy metrics 

def save_precision_recall_curve(true_labels, predicted_labels, cfg, args, epoch='128', split='train'):
        #### make the path if it doesn't exist
    if not os.path.exists((args.exp_dir) +'/'+(args.exp_name)+'/figs'):
        os.makedirs(args.exp_dir + '/'+(args.exp_name)+'/figs', exist_ok=True)
    
    PRcurve = PrecisionRecallDisplay.from_predictions(true_labels, predicted_labels)
    PRcurve.plot()
    plt.savefig(cfg['data_root'] + '/' + args.exp_dir + '/'+(args.exp_name)+'/figs/PRcurveTEST'+str(epoch) +'.png', facecolor="white")

def binaryMetrics(cfg, dl_val, model, args, epoch):
    print('generating binary predicted labels')
    filenames, true_labels, predicted_labels, confidences = predict(cfg, dl_val, model)   
    print('done generating predicted labels')
        
        # get accuracy score
        ### this is just a way to get two decimal places 
    acc = accuracy_score(true_labels, predicted_labels)
    print("Accuracy of model is {:0.2f}".format(acc))

        # confusion matrix
    confmatrix = save_confusion_matrix(true_labels, predicted_labels, cfg, args, epoch = epoch, split = 'train')
    print("confusion matrix saved")

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
    print("F1score of model is {:0.2f}".format(F1score))
    ######################################################################

    PRcurve = save_precision_recall_curve(true_labels, predicted_labels, cfg, args, epoch = epoch, split = 'train')
    print("precision recall curve saved")

    metrics = pd.DataFrame({'precision':precision, 'recall':recall, 'F1score':F1score}, index=[0])
    metrics.to_csv(cfg['data_root'] + '/'+ args.exp_dir +'/'+(args.exp_name)+'/figs/'+'metricsTEST.csv')
    print("metrics csv saved")

    # save list of predictions
    results = pd.DataFrame({'filenames':filenames, 'trueLabels':true_labels, 'predictedLabels':predicted_labels, 'confidences':confidences})
    results.to_csv(cfg['data_root'] + '/' + args.exp_dir +'/'+(args.exp_name)+'/figs/'+'resultsTEST.csv')
    print("results csv saved")

def multiClassMetrics(cfg, dl_val, model, args, epoch):
    print('generating multi-class predicted labels')
    filenames, true_labels, predicted_labels, confidences0, confidences1, confidences2 = predict(cfg, dl_val, model)   
    print('done generating predicted labels')
    
    # get accuracy score
    ### this is just a way to get two decimal places 
    acc = accuracy_score(true_labels, predicted_labels)
    print("Accuracy of model is {:0.2f}".format(acc))

    balanced_accuracy =  balanced_accuracy_score(true_labels, predicted_labels)
    report = (classification_report(true_labels, predicted_labels, output_dict=True))
    df = pd.DataFrame(report).transpose()
    df.to_csv(cfg['data_root'] + '/' + args.exp_dir +'/'+(args.exp_name)+'/figs/'+'classification_reportTEST.csv')
    print("classification report saved")


    # confusion matrix
    confmatrix = save_confusion_matrix(true_labels, predicted_labels, cfg, args, epoch = epoch, split = 'train')
    print("confusion matrix saved")

    metrics = pd.DataFrame({'accuracy':acc, 'balanced_acc':balanced_accuracy}, index=[0])
    metrics.to_csv(cfg['data_root'] + '/' + args.exp_dir +'/'+(args.exp_name)+'/figs/'+'metricsTEST.csv')
    print("metrics csv saved")

    # save list of predictions
    results = pd.DataFrame({'filenames':filenames, 'trueLabels':true_labels, 'predictedLabels':predicted_labels, 'confidences0':confidences0, 'confidences1':confidences1, 'confidences2':confidences2})
    results.to_csv(cfg['data_root'] + '/' + args.exp_dir +'/'+(args.exp_name)+'/figs/'+'resultsTEST.csv')
    print("results csv saved")

def main():
    # Argument parser for command-line arguments:
    # python code/train.py --output model_runs
    parser = argparse.ArgumentParser(description='Train deep learning model.')
    parser.add_argument('--exp_dir', required=True, help='Path to experiment directory', default = "experiment_dir")
    parser.add_argument('--exp_name', required=True, help='Path to experiment folder', default = "experiment_name")
    parser.add_argument('--split', help='Data split', default ='train')
    parser.add_argument('--config', help='Path to config file', default='configs/exp_resnet50_2classes.yaml')
    args = parser.parse_args()

    epoch = '128'
    # set model directory
    #exp_name = args.exp_name

    # reload config, that has to be set in the path
    print(f'Using config "{args.config}"')
    cfg = yaml.safe_load(open(args.config, 'r'))

    # setup dataloader validation
    dl_val = create_dataloader(cfg, folder = 'test_resized', labels = 'testLabels.csv')
    print(dl_val.__len__())

##create_dataloader(cfg, split='train', folder = 'train', labels = 'trainLabels.csv'):
    # load model and predict from model
    #IPython.embed()
    #IPython.embed()
    model, epoch = load_model(cfg, args.exp_dir, args.exp_name, epoch=None)

    if cfg['num_classes'] == 2:
        print('calculating binary metrics')
        binaryMetrics(cfg, dl_val, model, args, epoch)

    if cfg['num_classes'] == 3:
        print('calculating multiclass metrics')
        multiClassMetrics(cfg, dl_val, model, args, epoch)


if __name__ == '__main__':
    main()