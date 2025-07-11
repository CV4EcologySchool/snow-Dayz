import os
from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from model import get_model
import timm
import glob
import tqdm
import cv2
import numpy as np
import IPython
import pandas as pd

def load_model(model_path):
    # IPython.embed()
    checkpoint = torch.load(
        model_path,
        map_location="cpu"
    )
    model = get_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

def vis_predicted_keypoints(output_path, filename, image, snowdepth, color=(0,255,0), diameter=15):
    plt.imshow(image)
    plt.title(f'{snowdepth}')
    plt.savefig(f"{output_path}/predictions/pred_{filename}.png")
    plt.close()
   

def predict(model, images_path, output_path):

    if not os.path.exists(f"{output_path}/predictions"):
        os.makedirs(f"{output_path}/predictions", exist_ok=True)

    Cameras, filenames = [], []
    snowdepth = []
    
    snowpolefiles = glob.glob(f"{images_path}/*.JPG")

    with torch.no_grad():
        for i, file in tqdm.tqdm(enumerate(snowpolefiles)): #, total=num_batches):
      
            image = cv2.imread(file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, *_ = image.shape
            image = cv2.resize(image, (224,224))
            image = image / 255.0   

           # again reshape to add grayscale channel format
            filename = file.split('/')[-1]
            Camera = filename.split('_')[0]
            
            ## add an empty dimension for sample size
            image = np.transpose(image, (2, 0, 1)) ## to get channels to line up
            image = torch.tensor(image, dtype=torch.float)
            image = image.unsqueeze(0)
            image = image.to('cpu')

            #######
            outputs = model(image)
            outputs = outputs.cpu().numpy()
            snowdepth = np.array(outputs[0], dtype='float32')

            image = image.squeeze()
            image = image.cpu()
            image = np.transpose(image, (1, 2, 0))
     
            ###########
            vis_predicted_keypoints(output_path, filename, image, snowdepth) ## visualize points
 
            Cameras.append(Camera)
            filenames.append(filename)

    #IPython.embed()
    IPython.embed()
    results = pd.DataFrame({'Camera':Cameras, 'filename':filenames, 'snowdepth':snowdepth})

    results.to_csv(f"{output_path}/predictions/{Camera}_results.csv")

    return results

def main():

    model_path = '/Users/catherinebreen/Dropbox/Chapter1/aurora_outputsJul26/ViT_bs64_co-wa-aksplit_l1loss/model.pth'
    model = load_model(model_path)

    images_path = "/Users/catherinebreen/Dropbox/Chapter1/WRRsubmission/data/448res/SNEX20_TLI_resized_clean/CHE6"
    output_path = "/Users/catherinebreen/Dropbox/Chapter1/aurora_outputsJul26/ViT_bs64_co-wa-aksplit_l1loss"

    ## returns a set of images of outputs
    outputs = predict(model, images_path, output_path)  

if __name__ == '__main__':
    main()




