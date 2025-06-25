
###
images = '/Users/catherinebreen/Dropbox/Chapter1/WRRsubmission/data/448res/SNEX20_TLI_resized_clean'
labels = '/Users/catherinebreen/code/Chapter1/snowex_sd_withfilenames.csv'

# training hyperparameters
image_size = [224, 224]
num_epochs = 5
batch_size = 6
learning_rate = 1e-4
weight_decay = 0.001

output_path = "/Users/catherinebreen/code/Chapter1"
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

