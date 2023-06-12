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
from dataset import valid_data, test_data
from tqdm import tqdm
from scipy.spatial import distance
import os
import matplotlib.pyplot as plt
import math


def load_model():
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


def predict(model, data, eval='eval'): ## try this without a dataloader
    ## eval is the method, whether eval or test
    #files =  glob.glob(args.image_path + ('/**/*.JPG'))
    #df_data = pd.read_csv(f"{config.ROOT_PATH}/snowPoles_labels.csv")
    #IPython.embed()

    if not os.path.exists(f"{config.OUTPUT_PATH}/{eval}"):
        os.makedirs(f"{config.OUTPUT_PATH}/{eval}", exist_ok=True)

    output_list = []
    Cameras, filenames = [], []
    x1s_true, y1s_true, x2s_true, y2s_true = [], [], [], []
    x1s_pred, y1s_pred, x2s_pred, y2s_pred = [], [], [], []
    top_pixel_errors, bottom_pixel_errors, total_length_pixels = [], [], []
    total_length_pixel_actuals = []

    automated_sds, manual_sds, diff_sds = [], [], []

    #num_batches = int(len(data)/dataloader.batch_size)
    with torch.no_grad():
        for i, data in tqdm(enumerate(data)): #, total=num_batches):
            image, keypoints = data['image'].to(config.DEVICE), data['keypoints'].to(config.DEVICE)
            filename = data['filename']
            Camera = filename.split('_')[0]

            # flatten the keypoints
            keypoints = keypoints.detach().cpu().numpy().reshape(-1,2)
            x1_true, y1_true, x2_true, y2_true = keypoints[0,0], keypoints[0,1], keypoints[1,0], keypoints[1,1]
            ## add an empty dimension for sample size
            image = image.unsqueeze(0)
            outputs = model(image)
            #IPython.embed()
            outputs = outputs.detach().cpu().numpy()
            #output_list.append(outputs)
            utils.eval_keypoints_plot(filename, image, outputs, orig_keypoints=keypoints, eval) ## visualize points
            pred_keypoint = np.array(outputs[0], dtype='float32')
            x1_pred, y1_pred, x2_pred, y2_pred = pred_keypoint[0], pred_keypoint[1], pred_keypoint[2], pred_keypoint[3]
            
            Cameras.append(Camera)
            filenames.append(filename)
            x1s_true.append(x1_true), y1s_true.append(y1_true), x2s_true.append(x2_true), y2s_true.append(y2_true)
            x1s_pred.append(x1_pred), y1s_pred.append(y1_pred), x2s_pred.append(x2_pred), y2s_pred.append(y2_pred)

            ## outputs proj and in cm
            outputs_cm = utils.outputs_in_cm(Camera, filename, x1_pred, y1_pred, x2_pred, y2_pred)
            automated_sd = outputs_cm['snow_depth']
            automated_sds.append(automated_sd)

            # ## difference between automated and manual
            manual_snowdepth, difference = utils.diffcm(Camera, filename, automated_sd)
            manual_sds.append(manual_snowdepth), diff_sds.append(difference)

            ## error
            top_pixel_error = distance.euclidean([x1_true,y1_true], [x1_pred,y1_pred])
            bottom_pixel_error = distance.euclidean([x2_true,y2_true], [x2_pred,y2_pred])
            total_length_pixel = distance.euclidean([x1_pred,y1_pred],[x2_pred,y2_pred])
            total_length_pixel_actual = distance.euclidean([x1_true,y1_true],[x2_true,y2_true])
            top_pixel_errors.append(top_pixel_error), bottom_pixel_errors.append(bottom_pixel_error), total_length_pixels.append(total_length_pixel)
            total_length_pixel_actuals.append(total_length_pixel_actual)

    #IPython.embed()
    results = pd.DataFrame({'Camera':Cameras, 'filename':filenames, 'x1_true':x1s_true, 'y1_true':y1s_true, 'x2_true':x2s_true, 'y2_true':y2s_true, \
        'x1_pred': x1s_pred, 'y1s_pred': y1s_pred, 'x2_pred': x2s_pred, 'y2_pred': y2s_pred, 'top_pixel_error': top_pixel_errors, \
            'bottom_pixel_error': bottom_pixel_errors, 'total_length_pixel': total_length_pixels, 'total_length_pixel_actual': total_length_pixel_actuals,
            'automated_depth':automated_sds,'manual_snowdepth':manual_sds,'difference':diff_sds})

    #### overall average
    print('Overall Top Pixel Error \n')
    print(np.mean(top_pixel_errors))
    print('Overall Bottom Pixel Error \n')
    print(np.mean(bottom_pixel_errors))
    print('Overall difference in cm')
    print(np.mean(diff_sds))

    results.to_csv(f"{config.OUTPUT_PATH}/{eval}/results.csv")

    return results

def main():
    # Argument parser for command-line arguments:
    # python code/train.py --output model_runs
    #parser = argparse.ArgumentParser(description='Train deep learning model.')
    #parser.add_argument('--exp_dir', required=True, help='Path to experiment directory', default = "experiment_dir")
    #parser.add_argument('--exp_name', required=True, help='Path to experiment folder', default = "experiment_name")

    #args = parser.parse_args()
    model = load_model()

    ## returns a set of images of outputs
    outputs = predict(model, valid_data, eval='eval')  

    print(f"the results for the CHE and OK datasets...")
    outputs = predict(model, test_data, eval='test')

    #results = eval(outputs)

if __name__ == '__main__':
    main()



