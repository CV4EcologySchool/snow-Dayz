import torch
import os
# constant paths
# ROOT_PATH =    '/Volumes/CatBreen/CV4ecology/SNEX20_TLI_resized_clean' #'/datadrive/vmData/SNEX20_TLI' # '/Volumes/CatBreen/CV4ecology/SNEX20_TLI' # 'datadrive/data/SNEX20_TLI'
# OUTPUT_PATH = '/Volumes/CatBreen/CV4ecology/snow_poles_outputs_resized' #'/datadrive/vmData/snow_poles_outputs'

# snowfreetbl_path = '/Users/catherinebreen/Documents/Chapter1/WRRsubmission/snowfree_table.csv'
# manual_labels_path = '/Users/catherinebreen/Documents/Chapter1/WRRsubmission/SNEX20_SD_TLI_clean.csv'
# native_res_path = '/Users/catherinebreen/Documents/Chapter1/WRRsubmission/nativeRes.csv'
# ##res_info_path = '/Users/catherinebreen/Documents/Chapter1/WRRsubmission/resolution_info'
# datetime_info = '/Users/catherinebreen/Documents/Chapter1/WRRsubmission/labeledImgs_datetime_info.csv'


ROOT_PATH = '/datadrive/vmData/SNEX20_TLI_resized_clean'
#OUTPUT_PATH = '/datadrive/vmData/snow_poles_outputs_resized_FT_5_LRe4_BS64_E100_clean_Aug' #snow_poles_outputs_resized_FT_10_LRe4_BS64_E100_clean'
OUTPUT_PATH = '/datadrive/vmData/snow_poles_outputs_resized_nonFT5p_LRe4_BS64_E100_clean_OKonly' #snow_poles_outputs_resized_FT_10_LRe4_BS64_E100_clean'
#OUTPUT_PATH = '/datadrive/vmData/snow_poles_outputs_resized_LRe5_BS64_E100_clean'
snowfreetbl_path = '/datadrive/vmData/snowfree_table.csv'
manual_labels_path = '/datadrive/vmData/manuallylabeled.csv' #'/datadrive/vmData/SNEX20_SD_TLI_clean.csv'
datetime_info = '/datadrive/vmData/labeledImgs_datetime_info.csv' #'/datdrive/vmData/native_res/native_res'
##res_info_path = '/datadrive/vmData/resolution_info'
native_res_path = '/datadrive/vmData/nativeRes.csv'

#OUTPUT_PATH = '/Users/catherinebreen/Documents/Chapter1/dendrite_outputs/IN/snow_poles_outputs_resized_LRe4_BS64_clean_wWAOK_IN'

# learning parameters
BATCH_SIZE = 64 #64 #4 #32
LR = 0.0001 #0.00001  # #0.0001 lower to factor of 10
EPOCHS = 100 #100
#DEVICE = torch.device('mps')  #should be cuda on VMs
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# train/test split
TEST_SPLIT = 0.1  ## could update for the cameras that we want to hold out as validation
# show dataset keypoint plot
SHOW_DATASET_PLOT = False
AUG = False ## True for Aug; False for None

keypointColumns = ['x1', 'y1', 'x2', 'y2'] ## update

# Fine-tuning set-up
FINETUNE = True ## True for test/val dataset to be the subset 
FT_PATH = '/datadrive/vmData/snow_poles_outputs_resized_LRe4_BS64_E100_clean_SNEX_IN' ## model that you want to fine tune
FT_sample = 5
