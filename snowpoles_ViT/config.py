
###
# images = '/Users/catherinebreen/Dropbox/Chapter1/WRRsubmission/data/448res/SNEX20_TLI_resized_clean'
# #labels = '/Users/catherinebreen/code/snow-Dayz/snowpoles_ViT/snowex_sd_withfilenames.csv'
# labels = '/Users/catherinebreen/Dropbox/Chapter1/WRRsubmission/data/448res/SNEX20_TLI_resized_clean/snowex_ak_sd_withfilenames.csv'
images = '/datadrive/vmData/SNEX20_TLI_resized_clean'
labels = '/datadrive/snowex_ak_sd_withfilenames.csv'

split = 'camera' # or traditional (for 75/10/10)

# training hyperparameters
image_size = [224, 224]
num_epochs = 200
batch_size = 64
learning_rate = 1e-4
weight_decay = 0.001

#output_path = "/Users/catherinebreen/Dropbox/snowpoles_ViT_outputs/ViT_bs2_co_wa_ak"
output_path = "/datadrive/vmData/snowpoles_ViT_outputs/ViT_bs64_coak-wa-wasplit_l1loss"

## sample train run 
# python