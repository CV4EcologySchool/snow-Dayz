'''
    Training script. Here, we load the training and validation datasets (and
    data loaders) and the model and train and validate the model accordingly.

    2022 Benjamin Kellenberger
'''

import os
import argparse
import yaml
import glob
from tqdm import trange
import numpy as np 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler
from torch.optim import SGD
#from torch.utils.tensorboard import SummaryWriter 
from sklearn.metrics import balanced_accuracy_score
import IPython

# show model progress on tensorboard

# let's import our own classes and functions!
from dataset import CTDataset, train_test_split
from model import CustomResNet50


def create_dataloader(cfg, dataframe, labels):
    ###### added labels and folder argument
    '''
        Loads a dataset according to the provided split and wraps it in a
        PyTorch DataLoader object.
    '''
    #labels = os.path.join(self.data_root, labels) ## if the full path above doesn't work
    #dataset_instance = CTDataset(cfg, split)        # create an object instance of our CTDataset class

    dataset_instance = CTDataset(labels=labels, cfg=cfg, dataframe=dataframe)
    
    dataLoader = DataLoader(
            dataset=dataset_instance,
            batch_size=cfg['batch_size'],
            shuffle=True,
            num_workers=cfg['num_workers']
        )
    return dataLoader



def load_model(cfg):
    '''
        Creates a model instance and loads the latest model state weights.
    '''
    model_instance = CustomResNet50(cfg['num_classes'])         # create an object instance of our CustomResNet18 class

    # load latest model state
    dir = os.path.join(cfg['output_path'], cfg['exp_name'], 'model_states')
    model_states = glob.glob(f'{dir}/*.pt')
    # if len(model_states): # pass
    #     # at least one save state found; get latest
    #     model_epochs = [int(m.split('/')[-1].split('.')[0]) for m in model_states]
    #     start_epoch = max(model_epochs)

    #     # load state dict and apply weights to model
    #     print(f'Resuming from epoch {start_epoch}')
    #     state = torch.load(open(f'{dir}/{start_epoch}.pt', 'rb'), map_location='cpu')
    #     model_instance.load_state_dict(state['model'])

    # else:
    #     # no save state found; start anew
    #     print('Starting new model')
    #     start_epoch = 0

    start_epoch = 0

    return model_instance, start_epoch



def save_model(cfg, epoch, model, stats, args): ## dir
    # make sure save directory exists; create if not

    dir = os.path.join(cfg['output_path'], cfg['exp_name'], 'model_states')
   # full_save_path = os.path.join(dir, 'model_states')
    os.makedirs(dir, exist_ok=True) ####update here!
#### it was just dir, 'model_states"
    # get model parameters and add to stats...
    stats['model'] = model.state_dict()

    # ...and save
    torch.save(stats, open(f'{dir}/{epoch}.pt', 'wb')) ## {args.exp_dir}/{args.exp_name}/model_states
    

##model_states
def setup_optimizer(cfg, model):
    '''
        The optimizer is what applies the gradients to the parameters and makes
        the model learn on the dataset.
    '''
    optimizer = SGD(model.parameters(),
                    lr=cfg['learning_rate'],
                    weight_decay=cfg['weight_decay'])
    return optimizer



def train(cfg, dataLoader, model, optimizer):
    '''
        Our actual training function.
    '''

    device = cfg['device']

    # put model on device
    model.to(device) ##### ask should I be doing model.cuda??? ****


    # put the model into training mode
    # this is required for some layers that behave differently during training
    # and validation (examples: Batch Normalization, Dropout, etc.)
    model.train()

    # loss function
    #criterion = nn.CrossEntropyLoss() 
    
    # WEIGHTS
    # Define class weights and move them to the correct device
    class_weights = torch.tensor([1.0, 3.0], device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # LOGIT

    # running averages
    loss_total, oa_total = 0.0, 0.0                         # for now, we just log the loss and overall accuracy (OA)

    # iterate over dataLoader
    progressBar = trange(len(dataLoader))
    for idx, (data, labels) in enumerate(dataLoader):       # see the last line of file "dataset.py" where we return the image tensor (data) and label

        # put data and labels on device
        data, labels = data.to(device), labels.to(device)

        # forward pass
        prediction = model(data)

        # reset gradients to zero
        optimizer.zero_grad()

        # loss
        ## if labels == 1:
           #     loss = criterion(prediction, labels) * 2
           # else:  loss = criterion(prediction, labels)
        loss = criterion(prediction, labels)

        # backward pass (calculate gradients of current batch)
        loss.backward()

        # apply gradients to model parameters
        optimizer.step()

        # log statistics
        loss_total += loss.item()                       # the .item() command retrieves the value of a single-valued tensor, regardless of its data type and device of tensor

        pred_label = torch.argmax(prediction, dim=1)    # the predicted label is the one at position (class index) with highest predicted value
        
        oa = torch.mean((pred_label == labels).float()) # OA: number of correct predictions divided by batch size (i.e., average/mean)
        oa_total += oa.item()

        progressBar.set_description(
            '[Train] Loss: {:.2f}; OA: {:.2f}%'.format(  
                loss_total/(idx+1),
                100*oa_total/(idx+1)
            )
        )
        progressBar.update(1)
    
    # end of epoch; finalize
    progressBar.close()
    loss_total /= len(dataLoader)           # shorthand notation for: loss_total = loss_total / len(dataLoader)
    oa_total /= len(dataLoader)
    #ba_total /= len(dataLoader)

    return loss_total, oa_total



def validate(cfg, dataLoader, model):
    '''
        Validation function. Note that this looks almost the same as the training
        function, except that we don't use any optimizer or gradient steps.
    '''
    
    device = cfg['device']
    model.to(device)

    # put the model into evaluation mode
    # see lines 103-106 above
    model.eval()
    
    criterion = nn.CrossEntropyLoss()   # we still need a criterion to calculate the validation loss

    # running averages
    loss_total, oa_total = 0.0, 0.0     # for now, we just log the loss and overall accuracy (OA)

    # iterate over dataLoader
    progressBar = trange(len(dataLoader))
    
    with torch.no_grad():               # don't calculate intermediate gradient steps: we don't need them, so this saves memory and is faster
        for idx, (data, labels) in enumerate(dataLoader):

            # put data and labels on device
            data, labels = data.to(device), labels.to(device)

            # forward pass
            prediction = model(data)

            # loss
            loss = criterion(prediction, labels)

            # log statistics
            loss_total += loss.item()

            pred_label = torch.argmax(prediction, dim=1)
            oa = torch.mean((pred_label == labels).float())
            oa_total += oa.item()

            progressBar.set_description(
                '[Val ] Loss: {:.2f}; OA: {:.2f}%'.format(
                    loss_total/(idx+1),
                    100*oa_total/(idx+1)
                )
            )
            progressBar.update(1)
    
    # end of epoch; finalize
    progressBar.close()
    loss_total /= len(dataLoader)
    oa_total /= len(dataLoader)

    return loss_total, oa_total



def main():

    # Argument parser for command-line arguments:
    # python ct_classifier/train.py --config configs/exp_resnet18.yaml
    parser = argparse.ArgumentParser(description='Train deep learning model.')
    parser.add_argument('--config', help='Path to config file', default='configs/exp_resnet50_2classes.yaml')
 
    #
    args = parser.parse_args()

    #example command line usage:  
    # train.py --config path/to/config --exp_dir path/to/exp_dir --exp_name path/to/exp_name

    # load config
    print(f'Using config "{args.config}"')
    cfg = yaml.safe_load(open(args.config, 'r'))


   # if folder experiment folder/name DNE, make folder and copy args.config to the folder using os
   # if folder not in 
   ######################################################### this is technically in twice (make directory is in save model)
    # if not os.path.exists(cfg['output_path']) #os.path.join(cfg['data_root'], args.exp_dir)):
    #     os.makedirs(cfg['output_path']) #args.exp_dir, exist_ok=True) 
    save_path = os.path.join(cfg['output_path'],cfg['exp_name'])
    if not os.path.exists(f'{save_path}'): #os.path.join(cfg['data_root'], args.exp_dir, args.exp_name)):
        os.makedirs(f'{save_path}', exist_ok=True) #os.path.join(cfg['data_root'], args.exp_dir, args.exp_name), exist_ok=True) 

    print(f'Saving results to "{save_path}"')

    # check if GPU is available
    device = cfg['device']
    if device != 'cpu' and not torch.cuda.is_available():
        print(f'WARNING: device set to "{device}" but CUDA not available; falling back to CPU...')
        cfg['device'] = 'cpu'

    train_data, val_data = train_test_split(cfg, images_path = cfg['data_root'], labels = cfg['labels'])

    # initialize data loaders for training and validation set
    dl_train = create_dataloader(cfg, dataframe = train_data, labels = cfg['labels']) #folder=args.train_folder
    print('dl_train',dl_train.__len__())
    dl_test = create_dataloader(cfg, dataframe = val_data, labels = cfg['labels']) #folder=args.val_folder
    print('dl_val',dl_test.__len__())
    ## dl_test.dataset.__getitem__(1) 
    # initialize model
    model, current_epoch = load_model(cfg)

    # set up model optimizer
    optim = setup_optimizer(cfg, model)

    # writer = SummaryWriter(comment = args.exp_name) #log_dir=os.path.join(save_path, 'runs'))

    previousLoss = np.inf

    # we have everything now: data loaders, model, optimizer; let's do the epochs!
    numEpochs = cfg['num_epochs']

    #lossEpoch = []
    #bestLoss = max(int(lossEpoch))
    best_loss_val = np.inf
    best_loss_val_epoch = 0 # index of the epoch

    while current_epoch < numEpochs:
        current_epoch += 1
        print(f'Epoch {current_epoch}/{numEpochs}')

        loss_train, oa_train = train(cfg, dl_train, model, optim)
        loss_val, oa_val = validate(cfg, dl_test, model)

        # writer.add_scalar('Loss/Train', loss_train, current_epoch)
        # writer.add_scalar('OA/Train', oa_train, current_epoch)
        # writer.add_scalar('Loss/Val', loss_val, current_epoch)
        # writer.add_scalar('OA/Val', oa_val, current_epoch)

        # combine stats and save
        stats = {
            'loss_train': loss_train,
            'loss_val': loss_val,
            'oa_train': oa_train,
            'oa_val': oa_val
        }
                                                    
#################### early stopping #################
#### like Suzanne's code but looks for a plateau (over 10 epochs rather just not going down by a lot)
        if loss_val < best_loss_val:
                best_loss_val = loss_val
                best_loss_val_epoch = current_epoch
        elif current_epoch > best_loss_val_epoch + 10:
                save_model(cfg, current_epoch, model, stats, args)
                break
       # delta = previousLoss - loss_val
       # if delta < 1e-3:
       #     break
       # previousLoss = loss_val
        
    ## just the last epoch
    save_model(cfg, current_epoch, model, stats, args)
    




if __name__ == '__main__':
    # This block only gets executed if you call the "train.py" script directly
    # (i.e., "python ct_classifier/train.py").
    main()
