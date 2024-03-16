'''
load model results and identify snow depth
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
from tqdm import tqdm
from scipy.spatial import distance
import os
import matplotlib.pyplot as plt
from scipy.spatial import distance

def main():
    # Argument parser for command-line arguments:
    # python code/train.py --output model_runs
    parser = argparse.ArgumentParser(description='Predict top and bottom coordinates.')
    parser.add_argument('--model_path', required=False, help = 'Path to model', default = 'NULL')
    parser.add_argument('--dir_path', required=False, help='Path to camera image directory', default = '/example_data')
    parser.add_argument('--folder_path', required=False, help='Path to camera image folder', default = "/example_data/cam1")
    parser.add_argument('--output_path', required=True, help='Path to output folder', default = "/example_data")
    args = parser.parse_args()


    #args = parser.parse_args()
    model = load_model(args)

    ## returns a set of images of outputs
    outputs = predict(model, args)  

    #results = eval(outputs)

if __name__ == '__main__':
    main()



