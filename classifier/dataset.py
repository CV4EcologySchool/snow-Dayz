'''
    PyTorch dataset class for COCO-CT-formatted datasets. Note that you could
    use the official PyTorch MS-COCO wrappers:
    https://pytorch.org/vision/master/generated/torchvision.datasets.CocoDetection.html

    We just hack our way through the COCO JSON files here for demonstration
    purposes.

    See also the MS-COCO format on the official Web page:
    https://cocodataset.org/#format-data

    2022 Benjamin Kellenberger
'''

import os
import json
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image
import pandas as pd
import glob
from sequenceGenerator import sequenceGenerator
import random
from PIL import Image, ImageFile
import ipdb
import IPython

ImageFile.LOAD_TRUNCATED_IMAGES = True


class CTDataset(Dataset):

    # label class name to index ordinal mapping. This stays constant for each CTDataset instance.
    LABEL_CLASSES = {
        'None': 0,
        'Rain': 1,
        'Snow': 2,
        'Fog': 3,
        'Other': 4
    }
    LABEL_CLASSES_BINARY = {
        'None': 0,
        'Rain': 1,
        'Snow': 1,
        'Fog': 1,
        'Other': 1
    }

############################# columns to drop #######################
##### need to figure out how to load in config file or something else 
    seqSliding = [2831, 3211, 3365, 3632, 3838, 3893, 4450, 4492, 4973, 5366, 6365, 6367, 6456, 6488, 6497, 6511, 6522, 6524, 6530, 6542, 6559, 6607, 6629, 9361, 9734, 11875, 11939, 12147, 13363, 13367, 13432, 14792, 14844, 15115]
    seq6hr = [2080, 2099, 2795, 2799, 2813, 2831, 2946, 3032, 3104, 3211, 3363, 3365, 3375, 3384, 3445, 3457, 3470, 3631, 3632, 3680, 3690, 3724, 3744, 3754, 3763, 3764, 3778, 3792, 3817, 3819, 3833, 3834, 3838, 3870, 3871, 3892, 3893, 3904, 3905, 3906, 3915, 3978, 4032, 4033, 4034, 4102, 4143, 4144, 4263, 4274, 4275, 4316, 4384, 4450, 4484, 4492, 4518, 4589, 4590, 4618, 4623, 4625, 4647, 4670, 4671, 4681, 4702, 4703, 4753, 4766, 4796, 4815, 4820, 4842, 4874, 4882, 4895, 4897, 4901, 4907, 4908, 4912, 4934, 4953, 4954, 4973, 4979, 4980, 4981, 4984, 4985, 4986, 4996, 5185, 5334, 5366, 5409, 5411, 6299, 6305, 6365, 6367, 6374, 6376, 6384, 6386, 6411, 6420, 6426, 6427, 6428, 6429, 6454, 6456, 6464, 6468, 6476, 6478, 6482, 6488, 6491, 6493, 6495, 6497, 6498, 6503, 6504, 6506, 6510, 6511, 6517, 6522, 6524, 6527, 6529, 6530, 6538, 6540, 6542, 6543, 6546, 6552, 6556, 6557, 6559, 6560, 6563, 6564, 6567, 6578, 6579, 6582, 6585, 6589, 6591, 6594, 6607, 6615, 6623, 6629, 6637, 6647, 8427, 8650, 8655, 8656, 8657, 8658, 8659, 8660, 8661, 8662, 8663, 8664, 8665, 8666, 8667, 8668, 8669, 8670, 8671, 8672, 8673, 8674, 8675, 8676, 8677, 8678, 8679, 8680, 8681, 8682, 8683, 8684, 8685, 8686, 8687, 8688, 8689, 8696, 8706, 8717, 8728, 8739, 8750, 8761, 8772, 8783, 8801, 8811, 8822, 8833, 8844, 8855, 8866, 8877, 8888, 8895, 8903, 8914, 8925, 8936, 8947, 8958, 8969, 8980, 8990, 8996, 9005, 9016, 9027, 9038, 9049, 9060, 9070, 9081, 9092, 9097, 9106, 9117, 9128, 9139, 9150, 9161, 9172, 9183, 9194, 9198, 9200, 9201, 9202, 9203, 9204, 9205, 9206, 9207, 9208, 9209, 9210, 9211, 9212, 9213, 9214, 9215, 9216, 9217, 9218, 9219, 9220, 9221, 9222, 9223, 9224, 9225, 9226, 9227, 9228, 9229, 9230, 9231, 9232, 9233, 9234, 9235, 9236, 9237, 9238, 9239, 9240, 9241, 9242, 9243, 9244, 9245, 9246, 9247, 9248, 9249, 9250, 9251, 9252, 9253, 9254, 9255, 9256, 9262, 9273, 9284, 9295, 9299, 9346, 9354, 9356, 9361, 9362, 9363, 9364, 9457, 9642, 9734, 10108, 10400, 10401, 10405, 10423, 10457, 10494, 10496, 10497, 10513, 10517, 10562, 10866, 10967, 11168, 11269, 11370, 11675, 11786, 11843, 11855, 11866, 11869, 11870, 11875, 11905, 11914, 11921, 11931, 11938, 11939, 11942, 11950, 11959, 11998, 12018, 12033, 12043, 12057, 12066, 12069, 12071, 12072, 12119, 12121, 12124, 12127, 12128, 12131, 12146, 12147, 12158, 12167, 12175, 12183, 12201, 12260, 12266, 12268, 12285, 12288, 12293, 12411, 12413, 12418, 12426, 12464, 12487, 12501, 12503, 12506, 12589, 12600, 12605, 12750, 12813, 12831, 12836, 12853, 12858, 12863, 12889, 12943, 12985, 12992, 12998, 13014, 13027, 13057, 13061, 13094, 13099, 13106, 13107, 13117, 13211, 13217, 13218, 13327, 13328, 13331, 13352, 13354, 13356, 13358, 13363, 13366, 13367, 13369, 13371, 13378, 13428, 13429, 13431, 13432, 13433, 13435, 13436, 13438, 13447, 13468, 13470, 13492, 13502, 13506, 13617, 13633, 13641, 13654, 13660, 13665, 13666, 13668, 13673, 13683, 13686, 13697, 13703, 13707, 13711, 13729, 13730, 13756, 13791, 13863, 13867, 13874, 13908, 13909, 13927, 13932, 13944, 13947, 13966, 13967, 13972, 13973, 13974, 13975, 13978, 13979, 14002, 14009, 14023, 14035, 14054, 14056, 14060, 14066, 14069, 14074, 14086, 14109, 14123, 14127, 14148, 14163, 14171, 14177, 14205, 14218, 14222, 14223, 14247, 14275, 14279, 14299, 14310, 14313, 14350, 14368, 14400, 14403, 14404, 14549, 14552, 14554, 14560, 14564, 14569, 14572, 14578, 14638, 14641, 14643, 14645, 14661, 14664, 14666, 14671, 14677, 14689, 14691, 14693, 14738, 14750, 14760, 14784, 14787, 14788, 14791, 14792, 14811, 14839, 14844, 14846, 14849, 14860, 14870, 14880, 14891, 14893, 14901, 14911, 14920, 14921, 14970, 14979, 14982, 14983, 14985, 14994, 15008, 15013, 15016, 15023, 15039, 15058, 15077, 15082, 15085, 15089, 15090, 15102, 15105, 15107, 15108, 15115, 15133, 15139, 15163, 15177, 15178, 15185, 15189, 15200, 15204, 15218, 15231, 15233, 15236, 15238, 15239, 15241, 15251, 15262, 15263, 15356, 15363, 15378, 15449, 15450, 15451, 15452, 15453, 15454, 15455, 15456, 15457, 15459, 15465, 15474, 15475, 15477, 15485, 15495]
    seq12hr = [2080, 2099, 2795, 2799, 2813, 2831, 2946, 3032, 3104, 3211, 3363, 3365, 3375, 3384, 3445, 3457, 3470, 3631, 3632, 3680, 3690, 3724, 3744, 3754, 3763, 3764, 3778, 3792, 3817, 3819, 3833, 3834, 3838, 3870, 3871, 3892, 3893, 3904, 3905, 3906, 3915, 3978, 4032, 4033, 4034, 4102, 4143, 4144, 4263, 4274, 4275, 4316, 4384, 4450, 4484, 4492, 4518, 4589, 4590, 4618, 4623, 4625, 4647, 4670, 4671, 4681, 4702, 4703, 4753, 4766, 4796, 4815, 4820, 4842, 4874, 4882, 4895, 4897, 4901, 4907, 4908, 4912, 4934, 4953, 4954, 4973, 4979, 4980, 4981, 4984, 4985, 4986, 4996, 5185, 5334, 5366, 5409, 5411, 6299, 6305, 6365, 6367, 6374, 6376, 6384, 6386, 6411, 6420, 6426, 6427, 6428, 6429, 6454, 6456, 6464, 6468, 6476, 6478, 6482, 6488, 6491, 6493, 6495, 6497, 6498, 6503, 6504, 6506, 6510, 6511, 6517, 6522, 6524, 6527, 6529, 6530, 6538, 6540, 6542, 6543, 6546, 6552, 6556, 6557, 6559, 6560, 6563, 6564, 6567, 6578, 6579, 6582, 6585, 6589, 6591, 6594, 6607, 6615, 6623, 6629, 6637, 6647, 8427, 8650, 8655, 8656, 8657, 8658, 8659, 8660, 8661, 8662, 8663, 8664, 8665, 8666, 8667, 8668, 8669, 8670, 8671, 8672, 8673, 8674, 8675, 8676, 8677, 8678, 8679, 8680, 8681, 8682, 8683, 8684, 8685, 8686, 8687, 8688, 8689, 8696, 8706, 8717, 8728, 8739, 8750, 8761, 8772, 8783, 8801, 8811, 8822, 8833, 8844, 8855, 8866, 8877, 8888, 8895, 8903, 8914, 8925, 8936, 8947, 8958, 8969, 8980, 8990, 8996, 9005, 9016, 9027, 9038, 9049, 9060, 9070, 9081, 9092, 9097, 9106, 9117, 9128, 9139, 9150, 9161, 9172, 9183, 9194, 9198, 9200, 9201, 9202, 9203, 9204, 9205, 9206, 9207, 9208, 9209, 9210, 9211, 9212, 9213, 9214, 9215, 9216, 9217, 9218, 9219, 9220, 9221, 9222, 9223, 9224, 9225, 9226, 9227, 9228, 9229, 9230, 9231, 9232, 9233, 9234, 9235, 9236, 9237, 9238, 9239, 9240, 9241, 9242, 9243, 9244, 9245, 9246, 9247, 9248, 9249, 9250, 9251, 9252, 9253, 9254, 9255, 9256, 9262, 9273, 9284, 9295, 9299, 9346, 9354, 9356, 9361, 9362, 9363, 9364, 9457, 9642, 9734, 10108, 10400, 10401, 10405, 10423, 10457, 10494, 10496, 10497, 10513, 10517, 10562, 10866, 10967, 11168, 11269, 11370, 11675, 11786, 11843, 11855, 11866, 11869, 11870, 11875, 11905, 11914, 11921, 11931, 11938, 11939, 11942, 11950, 11959, 11998, 12018, 12033, 12043, 12057, 12066, 12069, 12071, 12072, 12119, 12121, 12124, 12127, 12128, 12131, 12146, 12147, 12158, 12167, 12175, 12183, 12201, 12260, 12266, 12268, 12285, 12288, 12293, 12411, 12413, 12418, 12426, 12464, 12487, 12501, 12503, 12506, 12589, 12600, 12605, 12750, 12813, 12831, 12836, 12853, 12858, 12863, 12889, 12943, 12985, 12992, 12998, 13014, 13027, 13057, 13061, 13094, 13099, 13106, 13107, 13117, 13211, 13217, 13218, 13327, 13328, 13331, 13352, 13354, 13356, 13358, 13363, 13366, 13367, 13369, 13371, 13378, 13428, 13429, 13431, 13432, 13433, 13435, 13436, 13438, 13447, 13468, 13470, 13492, 13502, 13506, 13617, 13633, 13641, 13654, 13660, 13665, 13666, 13668, 13673, 13683, 13686, 13697, 13703, 13707, 13711, 13729, 13730, 13756, 13791, 13863, 13867, 13874, 13908, 13909, 13927, 13932, 13944, 13947, 13966, 13967, 13972, 13973, 13974, 13975, 13978, 13979, 14002, 14009, 14023, 14035, 14054, 14056, 14060, 14066, 14069, 14074, 14086, 14109, 14123, 14127, 14148, 14163, 14171, 14177, 14205, 14218, 14222, 14223, 14247, 14275, 14279, 14299, 14310, 14313, 14350, 14368, 14400, 14403, 14404, 14549, 14552, 14554, 14560, 14564, 14569, 14572, 14578, 14638, 14641, 14643, 14645, 14661, 14664, 14666, 14671, 14677, 14689, 14691, 14693, 14738, 14750, 14760, 14784, 14787, 14788, 14791, 14792, 14811, 14839, 14844, 14846, 14849, 14860, 14870, 14880, 14891, 14893, 14901, 14911, 14920, 14921, 14970, 14979, 14982, 14983, 14985, 14994, 15008, 15013, 15016, 15023, 15039, 15058, 15077, 15082, 15085, 15089, 15090, 15102, 15105, 15107, 15108, 15115, 15133, 15139, 15163, 15177, 15178, 15185, 15189, 15200, 15204, 15218, 15231, 15233, 15236, 15238, 15239, 15241, 15251, 15262, 15263, 15356, 15363, 15378, 15449, 15450, 15451, 15452, 15453, 15454, 15455, 15456, 15457, 15459, 15465, 15474, 15475, 15477, 15485, 15495]
    seq24hr = [2773, 2799, 2831, 2946, 3032, 3104, 3211, 3363, 3365, 3384, 3444, 3457, 3468, 3469, 3631, 3632, 3680, 3690, 3724, 3744, 3754, 3762, 3763, 3764, 3778, 3785, 3792, 3817, 3819, 3838, 3845, 3870, 3871, 3893, 3906, 3915, 3978, 4027, 4033, 4034, 4102, 4143, 4144, 4248, 4263, 4274, 4275, 4316, 4373, 4384, 4430, 4444, 4450, 4484, 4492, 4518, 4572, 4589, 4590, 4623, 4647, 4670, 4671, 4681, 4702, 4703, 4753, 4766, 4796, 4815, 4820, 4842, 4874, 4882, 4892, 4897, 4901, 4907, 4908, 4934, 4953, 4954, 4973, 4979, 4980, 4981, 4984, 4985, 4986, 4996, 5185, 5232, 5366, 5411, 6282, 6298, 6365, 6367, 6372, 6384, 6385, 6419, 6426, 6454, 6456, 6463, 6467, 6474, 6475, 6482, 6488, 6490, 6492, 6495, 6497, 6498, 6501, 6502, 6506, 6510, 6511, 6517, 6522, 6524, 6526, 6529, 6530, 6536, 6539, 6541, 6542, 6544, 6552, 6556, 6557, 6558, 6559, 6561, 6562, 6565, 6578, 6579, 6580, 6584, 6587, 6591, 6594, 6607, 6613, 6621, 6629, 6635, 6645, 8427, 9346, 9354, 9356, 9361, 9734, 10400, 10404, 10423, 10457, 10494, 10496, 10497, 10513, 10517, 10562, 10967, 11168, 11269, 11370, 11675, 11855, 11866, 11870, 11875, 11905, 11914, 11920, 11931, 11939, 11942, 11950, 11959, 11962, 11998, 12018, 12033, 12043, 12057, 12066, 12069, 12071, 12118, 12124, 12127, 12128, 12131, 12145, 12147, 12158, 12167, 12175, 12201, 12260, 12266, 12267, 12288, 12293, 12411, 12413, 12418, 12487, 12503, 12506, 12589, 12605, 12750, 12831, 12836, 12853, 12858, 12863, 12888, 12992, 12998, 13012, 13027, 13057, 13061, 13094, 13099, 13107, 13117, 13211, 13217, 13328, 13331, 13352, 13353, 13355, 13357, 13363, 13365, 13367, 13368, 13370, 13378, 13428, 13429, 13431, 13432, 13433, 13435, 13436, 13438, 13447, 13470, 13488, 13490, 13502, 13504, 13617, 13633, 13641, 13654, 13660, 13665, 13666, 13668, 13673, 13683, 13686, 13697, 13703, 13707, 13711, 13730, 13756, 13791, 13863, 13867, 13874, 13908, 13944, 13947, 13966, 13972, 13973, 13974, 13975, 14054, 14056, 14060, 14066, 14069, 14074, 14086, 14109, 14123, 14127, 14148, 14163, 14171, 14177, 14205, 14218, 14222, 14223, 14247, 14275, 14279, 14299, 14310, 14313, 14346, 14400, 14404, 14549, 14551, 14554, 14560, 14564, 14569, 14572, 14576, 14577, 14638, 14641, 14642, 14645, 14661, 14665, 14671, 14677, 14689, 14690, 14692, 14694, 14738, 14750, 14760, 14784, 14787, 14788, 14791, 14792, 14839, 14844, 14846, 14849, 14860, 14880, 14891, 14901, 14911, 14921, 14970, 14979, 14982, 14983, 14985, 15008, 15058, 15082, 15085, 15089, 15090, 15102, 15105, 15115, 15139, 15163, 15178, 15189, 15200, 15204, 15218, 15231, 15233, 15235, 15238, 15241, 15251, 15262, 15263, 15356, 15362, 15449, 15450, 15451, 15452, 15453, 15454, 15455, 15456, 15457, 15459, 15465, 15474, 15475, 15484, 15495]

    def __init__(self, labels, cfg, split='train', folder='train'):
        '''
            Constructor. Here, we collect and index the dataset inputs and
            labels.
        '''
        self.data_root = cfg['data_root']
        self.split = split
        self.folder = folder
        self.sequenceType = cfg['sequenceType']
        self.transform = Compose([              # Transforms. Here's where we could add data augmentation (see Björn's lecture on August 11).
            Resize((cfg['image_size'])),        # For now, we just resize the images to the same dimensions...
            ToTensor()                          # ...and convert them to torch.Tensor.
        ])
        self.drop = cfg['drop']
        
        # index data into list
        self.data = []

        def delete_multiple_element(list_object, indices):
            for idx in indices:
                if idx < len(list_object):
                    list_object.pop(idx)
            return list_object


        # load annotation file
        self.annoPath = os.path.join(
            self.data_root, labels) ############# should i set this as an input?? 

        meta = pd.read_csv(self.annoPath)
        meta = meta[meta['Weather'] != 'Fog'] ### could also drop 'other'
        meta = meta[meta['Weather'] != 'Other']
        meta = meta[meta['File'] != '2015_04_05_09_00_00.jpg']
        meta = meta.drop_duplicates().reset_index() ## maybe I should keep the original indices??

        ## add a check to make sure it exists in the folder of interest
        list_of_images = glob.glob(os.path.join(self.data_root,'train_resized/*')) ####UPDATED
        list_of_images = pd.Series(list_of_images)
        list_of_images = pd.DataFrame(list_of_images.str.split('/', expand=True)[5])

        if self.sequenceType == 'None':
            for file, weather in zip(meta['File'], meta['Weather']):
                #if random.uniform(0.0, 1.0) <= 0.99:
                    #continue
                    #(random.uniform(0.0, 1.0) <= 0.005) and
                if (sum(list_of_images == file) > 0): ## make sure there is the file in the train folder
                    imgFileName = file
                    if cfg['num_classes'] == 2:
                        self.data.append([imgFileName, self.LABEL_CLASSES_BINARY[weather]])
                    else: self.data.append([imgFileName, self.LABEL_CLASSES[weather]]) ## why label index and not label?

######################### sequences #################
        if self.sequenceType != 'None':
            for file, weather in zip(meta['File'], meta['Weather']):
                ## (random.uniform(0.0, 1.0) <= 0.001) and 
                if sum(list_of_images == file) > 0: ## make sure there is the file in the image (train) folder
                    imgFileName = file
                    before, file, after = sequenceGenerator(meta, file, sequenceType = self.sequenceType)
                    imgFileName = file 
                    if cfg['num_classes'] == 2:
                        imgFileName = file
                        self.data.append([[before, imgFileName, after], self.LABEL_CLASSES_BINARY[weather]])
                    else: self.data.append([[before, imgFileName, after], self.LABEL_CLASSES[weather]]) ## why label index and not label?
        print(len(self.data))
        IPython.embed()
############ drop identical image sequences ######################
#### these were pullled after running sequence Generator so the indices are unique to that
        if (self.sequenceType == 'sliding') and (self.drop == 'True'): self.data = delete_multiple_element(self.data, self.seqSliding)
        if (self.sequenceType == '3-6hr') and (self.drop == 'True'): self.data = delete_multiple_element(self.data, self.seq6hr)
        if (self.sequenceType == '6-12hr') and (self.drop == 'True'): self.data = delete_multiple_element(self.data, self.seq12hr)
        if (self.sequenceType == '12-24hr') and (self.drop == 'True'): self.data = delete_multiple_element(self.data, self.seq24hr)


    def __len__(self):
        '''
            Returns the length of the dataset.
        '''
        return len(self.data)

    def __shape__(self):
        return (self.data)

    def __sequenceType__(self):
        return (self.sequenceType)

    def __getitem__(self, idx):
        '''
            Returns a single data point at given idx.
            Here's where we actually load the image.
        '''
        image_name, label = self.data[idx]              # see line 57 above where we added these two items to the self.data list

        # load image
        if self.sequenceType == 'None':
        #try:
            image_path = os.path.join(self.data_root, self.folder, image_name) ## should specify train folder and get image name 
            img = Image.open(image_path).convert('RGB')     # the ".convert" makes sure we always get three bands in Red, Green, Blue order
            #except: pass
            # transform: see lines 31ff above where we define our transformations
            img_tensor = self.transform(img)
            #print(img_tensor.shape)
        
    ######################################## sequences ##########################
        ##import IPython ## for testing : may need to install: pip install IPython
        ##IPython.embed() ## for testing
        #ipdb.set_trace()
       
        #IPython.embed()
        if self.sequenceType != 'None':
            before, image_name, after = image_name

            image_path1 = os.path.join(self.data_root, self.folder, before) ## should specify train folder and get image name 
            #print(image_path1)
            #print(self.data_root)
            #print(self.folder)
            image_path2 = os.path.join(self.data_root, self.folder, image_name)
            image_path3 = os.path.join(self.data_root, self.folder, after) ####

            img1 = Image.open(image_path1).convert('L')     # the ".convert" makes sure we always get three bands in Red, Green, Blue order
            img2 = Image.open(image_path2).convert('L')
            img3 = Image.open(image_path3).convert('L')
            #(print(img3.size))
            #except: pass

            # transform: see lines 31ff above where we define our transformations
            img_tensor1 = self.transform(img1)
            img_tensor2 = self.transform(img2)
            img_tensor3 = self.transform(img3)

            img_tensor = torch.cat([img_tensor1, img_tensor2,img_tensor3], dim = 0) ### 

############################################################################# kadjfldsf

        return img_tensor, label




  
