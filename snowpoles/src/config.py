import torch
# constant paths
ROOT_PATH = 'datadrive/data/SNEX20_TLI'
OUTPUT_PATH = '../outputs'
# learning parameters
BATCH_SIZE = 32
LR = 0.001
EPOCHS = 30
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# train/test split
TEST_SPLIT = 0.1
# show dataset keypoint plot
SHOW_DATASET_PLOT = True

### in my datasheet it is columns 3, 4, 5, 6, so we will use range 3:7
## or we can name them directly

keypointColumns = ['x1', 'y1', 'x2', 'y2']