'''
load model and run on data points 
export the csv of the data points and just use the bottom

example command line to run:

python src/evaluate.py --exp_dir '/Volumes/CatBreen/CV4ecology/snow_poles_outputs2' --exp_name snow_poles_outputs2 --image_path '/Volumes/CatBreen/CV4ecology/SNEX20_TLI_test'

'''

import torch
import numpy as np
import cv2
import albumentations  ## may need to do pip install
import config
from model import snowPoleResNet50
import argparse
import glob
import IPython
import utils
import pandas as pd
from dataset import train_data, train_loader, valid_data, valid_loader
from tqdm import tqdm
from scipy.spatial import distance
import os
import matplotlib.pyplot as plt

def load_model():
    model = snowPoleResNet50(pretrained=False, requires_grad=False).to(config.DEVICE)
    # load the model checkpoint
    checkpoint = torch.load(config.OUTPUT_PATH + '/model_epoch98.pth')
    # load model weights state_dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


'''
We will use part of the valid function to write our predict function. It will be very similiar except
that it will use the last model, and we will just use the dataset, not the dataloader.

It is a little bit easier to flatten this way. 

'''

def eval_keypoints_plot(file, image, outputs, args):
    """
    This function plots the regressed (predicted) keypoints and the actual 
    keypoints after each validation epoch for one image in the batch.
    """
    # detach the image, keypoints, and output tensors from GPU to CPU
    #IPython.embed()
    image = image.detach().cpu()
    image = image.squeeze(0) ## drop the dimension because no longer need it for model 
    outputs = outputs #.detach().cpu().numpy()
    # just get a single datapoint from each batch
    #img = image[0]
    output_keypoint = outputs[0] ## don't know why but it is technically nested
    img = np.array(image, dtype='float32')
    img = np.transpose(img, (1, 2, 0))
    plt.imshow(img)
    
    output_keypoint = output_keypoint.reshape(-1, 2)

    for p in range(output_keypoint.shape[0]):
        if p == 0: 
            plt.plot(output_keypoint[p, 0], output_keypoint[p, 1], 'r.') ## top
        else:
            plt.plot(output_keypoint[p, 0], output_keypoint[p, 1], 'g.') ## bottom
    plt.savefig(f"{args.output_path}/predictions/predict_{file}.png")
    plt.close()



def predict(model, args): ## try this without a dataloader
    #files =  glob.glob(args.image_path + ('/**/*.JPG'))
    #df_data = pd.read_csv(f"{config.ROOT_PATH}/snowPoles_labels.csv")
    #IPython.embed()

    if not os.path.exists(f"{args.output_path}/predictions"):
        os.makedirs(f"{args.output_path}/predictions", exist_ok=True)

    Cameras, filenames = [], []
    x1s_pred, y1s_pred, x2s_pred, y2s_pred = [], [], [], []
    
    snowpolefiles = glob.glob(f"{args.image_path}/**/*")
    #snowpoleList = [item.split('/')[-1] for item in snowpolefiles]

    #num_batches = int(len(data)/dataloader.batch_size)
    with torch.no_grad():
        for i, file in tqdm(enumerate(snowpolefiles)): #, total=num_batches):
            
            image = cv2.imread(file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224,224))
            image = image / 255.0   

        #IPython.embed()
        # again reshape to add grayscale channel format
            image = image / 255.0    
            filename = file.split('/')[-1]
            Camera = filename.split('_')[0]
            
            ## add an empty dimension for sample size
            image = image.unsqueeze(0)
            outputs = model(image)
            #IPython.embed()
            outputs = outputs.detach().cpu().numpy()
            #output_list.append(outputs)
            utils.vis_keypoints(image, outputs,) ## visualize points
            pred_keypoint = np.array(outputs[0], dtype='float32')
            x1_pred, y1_pred, x2_pred, y2_pred = pred_keypoint[0], pred_keypoint[1], pred_keypoint[2], pred_keypoint[3]
            
            Cameras.append(Camera)
            filenames.append(filename)
            x1s_pred.append(x1_pred), y1s_pred.append(y1_pred), x2s_pred.append(x2_pred), y2s_pred.append(y2_pred)

    #IPython.embed()
    results = pd.DataFrame({'Camera':Cameras, 'filename':filenames, \
        'x1_pred': x1s_pred, 'y1s_pred': y1s_pred, 'x2_pred': x2s_pred, 'y2_pred': y2s_pred})

    results.to_csv(f"{config.OUTPUT_PATH}/predictions/results.csv")

    return results

def main():
    # Argument parser for command-line arguments:
    # python code/train.py --output model_runs
    parser = argparse.ArgumentParser(description='Train deep learning model.')
    parser.add_argument('--image_path', required=True, help='Path to image directory', default = "image_path")
    parser.add_argument('--output_path', required=True, help='Path to output folder', default = "output_path")
    args = parser.parse_args()


    #args = parser.parse_args()
    model = load_model()

    ## returns a set of images of outputs
    outputs = predict(model, args)  

    #results = eval(outputs)

if __name__ == '__main__':
    main()



