import matplotlib.pyplot as plt
import numpy as np
import config
import IPython
import cv2 

def valid_keypoints_plot(image, outputs, orig_keypoints, epoch):
    """
    This function plots the regressed (predicted) keypoints and the actual 
    keypoints after each validation epoch for one image in the batch.
    """
    # detach the image, keypoints, and output tensors from GPU to CPU
    image = image.detach().cpu()
    outputs = outputs.detach().cpu().numpy()
    orig_keypoints = orig_keypoints.detach().cpu().numpy()
    # just get a single datapoint from each batch
    img = image[0]  ## something snow in it ## halfway throught the dataset
    output_keypoint = outputs[0]
    orig_keypoint = orig_keypoints[0]
    img = np.array(img, dtype='float32')
    img = np.transpose(img, (1, 2, 0))
    plt.imshow(img)
    
    output_keypoint = output_keypoint.reshape(-1, 2)
    orig_keypoint = orig_keypoint.reshape(-1, 2)
    for p in range(output_keypoint.shape[0]):
        if p == 0: 
            plt.plot(output_keypoint[p, 0], output_keypoint[p, 1], 'r.') ## top
            plt.plot(orig_keypoint[p, 0], orig_keypoint[p, 1], 'b.')
        else:
            plt.plot(output_keypoint[p, 0], output_keypoint[p, 1], 'g.') ## bottom
            plt.plot(orig_keypoint[p, 0], orig_keypoint[p, 1], 'b.')
    plt.savefig(f"{config.OUTPUT_PATH}/val_epoch_{epoch}.png")
    plt.close()


def dataset_keypoints_plot(data):
    '''  
    #  This function shows the image faces and keypoint plots that the model
    # will actually see. This is a good way to validate that our dataset is in
    # fact corrent and the faces align wiht the keypoint features. The plot 
    # will be show just before training starts. Press `q` to quit the plot and
    # start training.
    '''
    plt.figure(figsize=(10, 10))
    for i in range(9):
        sample = data[i]
        img = sample['image']
        img = np.array(img, dtype='float32') #/255
        #IPython.embed()
        img = np.transpose(img, (1, 2, 0))
        plt.subplot(3, 3, i+1)
        plt.imshow(img)
        keypoints = sample['keypoints']
        for j in range(len(keypoints)):
            plt.plot(keypoints[j, 0], keypoints[j, 1], 'b.')
    plt.show()
    plt.close()


def eval_keypoints_plot(file, image, outputs, orig_keypoints):
    """
    This function plots the regressed (predicted) keypoints and the actual 
    keypoints after each validation epoch for one image in the batch.
    """
    # detach the image, keypoints, and output tensors from GPU to CPU
    #IPython.embed()
    image = image.detach().cpu()
    image = image.squeeze(0) ## drop the dimension because no longer need it for model 
    outputs = outputs #.detach().cpu().numpy()
    orig_keypoints = orig_keypoints #.detach().cpu().numpy()#orig_keypoints.detach().cpu().numpy()
    # just get a single datapoint from each batch
    #img = image[0]
    output_keypoint = outputs[0] ## don't know why but it is technically nested
    img = np.array(image, dtype='float32')
    img = np.transpose(img, (1, 2, 0))
    plt.imshow(img)
    
    output_keypoint = output_keypoint.reshape(-1, 2)
    orig_keypoints = orig_keypoints.reshape(-1, 2)
    for p in range(output_keypoint.shape[0]):
        if p == 0: 
            plt.plot(orig_keypoints[p, 0], orig_keypoints[p, 1], 'b.',  markersize=20)
            plt.plot(output_keypoint[p, 0], output_keypoint[p, 1], 'orange', markersize=20) ## top
        else:
            plt.plot(orig_keypoints[p, 0], orig_keypoints[p, 1], 'b.',  markersize=20)
            plt.plot(output_keypoint[p, 0], output_keypoint[p, 1], 'g.', markersize=20) ## bottom
    plt.savefig(f"{config.OUTPUT_PATH}/eval/eval_{file}.png")
    plt.close()


#def object_keypoint_similarity()
def vis_keypoints(image, keypoints, color=(0,255,0), diameter=15):
    image = image.copy()

    for (x, y) in keypoints:
        cv2.circle(image, (int(x), int(y)), diameter, (0, 255, 0), -1)
        
    #plt.figure(figsize=(8, 8))
    #plt.axis('off')
    plt.imshow(image)
    plt.show()
    plt.close()

