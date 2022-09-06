'''
load model and run on data points 
export the csv of the data points and just use the bottom

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

def load_model(args):
    model = snowPoleResNet50(pretrained=False, requires_grad=False).to(config.DEVICE)
    # load the model checkpoint
    checkpoint = torch.load(config.OUTPUT_PATH + '/model.pth')
    # load model weights state_dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def predict(model, args):
    files = glob.glob(args.image_path)
    x1 = []
    y1 = []
    x2 = []
    y2 = []

    with torch.no_grad():
        for file in args.image_path: ## add args file
            IPython.embed()
            image = cv2.imread(file)
            image = cv2.resize(image, (224, 224)) ### could change
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #orig_h, orig_w, channel = image.shape ## only need this to rescale all the points
            image = image / 255.0
            image = np.transpose(image, (2, 0, 1))
            image = torch.tensor(image, dtype=torch.float)
            image = image.unsqueeze(0).to(config.DEVICE)
            outputs = model(image)

    return outputs



def main():
    # Argument parser for command-line arguments:
    # python code/train.py --output model_runs
    parser = argparse.ArgumentParser(description='Train deep learning model.')
    parser.add_argument('--exp_dir', required=True, help='Path to experiment directory', default = "experiment_dir")
    parser.add_argument('--exp_name', required=True, help='Path to experiment folder', default = "experiment_name")
    parser.add_argument('--image_path', required=True, help='Path to full image folder', default = "images")

    args = parser.parse_args()
    model = load_model(args)

    outputs = predict(model, args)  

   

if __name__ == '__main__':
    main()

