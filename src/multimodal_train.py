import torch
import logging
import os
import argparse
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from pytorchtools import EarlyStopping
from sklearn.metrics import mean_absolute_error

from multimodal_CNN import CNN
from multimodal_loader import get_data, get_data_subject
from utils import get_mpjpe


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

# Check if CUDA is available and choose device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_epoch(model, train_loader, criterion, optimizer):
    model.train()
    train_losses = []
    for batch_data, batch_labels in train_loader:
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)
        # forward
        batch_preds = model(batch_data)
        loss = criterion(batch_preds, batch_labels)
        # set the gradient to zero
        optimizer.zero_grad()
        # do backward prop
        loss.backward()
        # update the gradient
        optimizer.step()
        # record training loss
        train_losses.append(loss.item())
    return train_losses

def validate_model(model, test_loader, criterion):
    model.eval()
    valid_losses = []
    total_p1_mpjpe, total_p2_mpjpe, total_mae, cnt = 0, 0, 0, 0
    num_joints = 17
    num_dim = 3
    
    for batch_data, batch_labels in test_loader:
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)

        with torch.no_grad():
            batch_preds = model(batch_data)
            loss = criterion(batch_preds, batch_labels)
            valid_losses.append(loss.item())

            batch_preds = np.array(batch_preds.cpu()).reshape(-1, num_dim, num_joints)
            batch_labels = np.array(batch_labels.cpu()).reshape(-1, num_dim, num_joints)

            p1_mpjpe, p2_mpjpe = get_mpjpe(batch_preds, batch_labels)
            mae = mean_absolute_error(batch_preds.reshape(-1, num_dim*num_joints), batch_labels.reshape(-1, num_dim*num_joints))

            total_p1_mpjpe += p1_mpjpe
            total_p2_mpjpe += p2_mpjpe
            total_mae += mae
            cnt += 1


    avg_p1_mpjpe = total_p1_mpjpe / cnt
    avg_p2_mpjpe = total_p2_mpjpe / cnt
    avg_mae = total_mae / cnt

    return valid_losses, avg_p1_mpjpe, avg_p2_mpjpe, avg_mae


def main(protocol, datasplit, modality = 'radar', total_epochs=100, patience=20, learning_rate=0.001, betas=(0.5, 0.999)):
    subject_list = ['subject' + str(i) for i in range(1,21)]
    all_test_subject_list = [['subject17', 'subject13', 'subject11', 'subject15'],
    ['subject9', 'subject7', 'subject20', 'subject8'],
    ['subject3', 'subject16', 'subject7', 'subject2']]


    random_seed_list = [665,666,667]

    #radar or imu
    outdir = f'model/{modality}'
    os.makedirs(outdir, exist_ok=True)
    
    # total error for paper
    total_error = []    
    for idx, random_seed in enumerate(random_seed_list):
        # protocol 1: everything, protocol 2: only 10 movements
        num_joints = 17
        num_dim = 3
        
        if datasplit == 1:
            train_loader, test_loader = get_data(subject_list, protocol, modality, random_seed)
            split = 'random'
        elif datasplit == 2:
            test_subject_list = all_test_subject_list[idx]
            train_loader, test_loader = get_data_subject(test_subject_list, protocol, modality) 
            split = 'subject' 
        
        # initialize the model
        if modality == 'radar':
            model = CNN(channels=5, neurons=51, fc_input_size=6272).to(device)
        elif modality == 'imu':
            model = CNN(channels=1, neurons=51, fc_input_size=2304).to(device)

        # Declaring Loss and Optimizer
        criterion = F.l1_loss
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=betas)

        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=patience, verbose=True, path=os.path.join(outdir, f"{modality}_protocol{str(protocol)}_datasplit{str(datasplit)}.pt"))
        
        # to track the average training loss per epoch as the model trains
        avg_train_losses = []
        # to track the average validation loss per epoch as the model trains
        avg_valid_losses = [] 

        # for the main training
        for epoch_count in range(total_epochs):
            train_losses = train_epoch(model, train_loader, criterion, optimizer)
            valid_losses, avg_p1_mpjpe, avg_p2_mpjpe, avg_mae = validate_model(model, test_loader, criterion)

            # calculate average loss over an epoch
            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)
            
            epoch_len = len(str(total_epochs))
            
            print_msg = (f'[{epoch_count:>{epoch_len}}/{epoch_count:>{epoch_len}}] ' +
                        f'train_loss: {train_loss:.5f} ' +
                        f'valid_loss: {valid_loss:.5f}')
            
            logger.info(print_msg)
            logger.info(f'Validation P1 MPJPE: {avg_p1_mpjpe:.5f}, P2 MPJPE: {avg_p2_mpjpe:.5f}, MAE: {avg_mae:.5f}')
            
            # clear lists to track next epoch
            train_losses = []
            valid_losses = []
            
            # early_stopping needs the validation loss to check if it has decresed, 
            # and if it has, it will make a checkpoint of the current model
            early_stopping(valid_loss, model)
            
            if early_stopping.early_stop:
                logger.info(f"Early stopping at epoch: {epoch_count}")
                break

        total_error.append([avg_p1_mpjpe*1000, avg_p2_mpjpe*1000])

    total_error.append(np.mean(total_error, 0))
    total_error.append(np.std(total_error, 0))

    # save results
    os.makedirs(f'{outdir}/results', exist_ok=True)
    np.save(f'{outdir}/results/{split}_split_protocol{str(protocol)}.npy', np.array(total_error))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('-p', '--protocol', type=int, default = 2,
           help='protocol 1 means with walking and others, protocol 2 means only 10 movements')
    p.add_argument('-s', '--datasplit', type=int, default = 1,
        help='1 means random setting, 2 means split by subjects')
    p.add_argument('-m', '--modality', type=str, default = 'radar',
        help='radar or imu')
    p.add_argument('-e', '--total_epochs', type=int, default = 50,
        help='total number of training epochs')
    p.add_argument('-pt', '--patience', type=int, default = 20,
        help='patience for early stopping')
    p.add_argument('-lr', '--learning_rate', type=float, default = 0.001,
        help='learning rate for optimizer')
    p.add_argument('-b', '--betas', type=tuple, default = (0.5, 0.999),
        help='betas for Adam optimizer')
    main(**vars(p.parse_args()))