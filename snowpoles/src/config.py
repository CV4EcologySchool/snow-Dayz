import torch
import os
# constant paths
#ROOT_PATH =    '/Volumes/CatBreen/CV4ecology/SNEX20_TLI_resized' #'/datadrive/vmData/SNEX20_TLI' # '/Volumes/CatBreen/CV4ecology/SNEX20_TLI' # 'datadrive/data/SNEX20_TLI'
#OUTPUT_PATH = '/Volumes/CatBreen/CV4ecology/snow_poles_outputs_resized' #'/datadrive/vmData/snow_poles_outputs'

ROOT_PATH = '/datadrive/vmData/SNEX20_TLI_resized'
OUTPUT_PATH = '/datadrive/vmData/snow_poles_outputs_resized_LRe5_BS64_E100_clean_SNEX_IN'
#OUTPUT_PATH = '/Users/catherinebreen/Documents/Chapter1/dendrite_outputs/IN/snow_poles_outputs_resized_LRe4_BS64_clean_wWAOK_IN'

# learning parameters
BATCH_SIZE = 64 #64 #4 #32
LR = 0.00001  # #0.0001 lower to factor of 10
EPOCHS = 100 #100
#DEVICE = torch.device('mps')  #should be cuda on VMs
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# train/test split
TEST_SPLIT = 0.1  ## could update for the cameras that we want to hold out as validation
# show dataset keypoint plot
SHOW_DATASET_PLOT = False
DOMAIN = True ## True for IN DOMAIN; False for Out of Domain
SNEX = True

### in my datasheet it is columns 3, 4, 5, 6, so we will use range 3:7
## or we can name them directly

keypointColumns = ['x1', 'y1', 'x2', 'y2'] ## update