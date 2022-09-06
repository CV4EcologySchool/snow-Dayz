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

def load_model(args):
    model = snowPoleResNet50(pretrained=False, requires_grad=False).to(config.DEVICE)
    # load the model checkpoint
    checkpoint = torch.load(config.OUTPUT_PATH + '/model.pth')
    # load model weights state_dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


'''
We will use part of the valid function to write our predict function. It will be very similiar except
that it will use the last model, and we will just use the dataset, not the dataloader.

It is a little bit easier to flatten this way. 

'''
def predict(model, data): ## try this without a dataloader
    #files =  glob.glob(args.image_path + ('/**/*.JPG'))
    #df_data = pd.read_csv(f"{config.ROOT_PATH}/snowPoles_labels.csv")
    #IPython.embed()
    output_list = []
    #num_batches = int(len(data)/dataloader.batch_size)
    with torch.no_grad():
        for i, data in tqdm(enumerate(data)): #, total=num_batches):
            image, keypoints = data['image'].to(config.DEVICE), data['keypoints'].to(config.DEVICE)
            filename = data['filename']
            # flatten the keypoints
            keypoints = keypoints.view(keypoints.size(0), -1)
            ## add an empty dimension for sample size
            image = image.unsqueeze(0)
            outputs = model(image)
            output_list.append(outputs)
            utils.eval_keypoints_plot(filename, image, outputs, orig_keypoints=keypoints) ## visualize points

    return output_list

def eval(args):
    files =  glob.glob(args.image_path + ('/**/*.JPG'))
    df_data = pd.read_csv(f"{config.ROOT_PATH}/snowPoles_labels.csv")
    #oks_1 =
    #oks_2 =


def main():
    # Argument parser for command-line arguments:
    # python code/train.py --output model_runs
    parser = argparse.ArgumentParser(description='Train deep learning model.')
    parser.add_argument('--exp_dir', required=True, help='Path to experiment directory', default = "experiment_dir")
    parser.add_argument('--exp_name', required=True, help='Path to experiment folder', default = "experiment_name")
    #parser.add_argument('--image_path', required=True, help='Path to full image folder', default = "images")
    #parser.add_argument('')


    args = parser.parse_args()
    model = load_model(args)

    ## returns a set of images of outputs
    outputs = predict(model, valid_data)  


if __name__ == '__main__':
    main()



