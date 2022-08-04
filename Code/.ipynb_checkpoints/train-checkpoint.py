#Import all necessary packages
import os
import sys
sys.path.insert(1,'/home/gridsan/jmaalouf/packages/chemprop_01') #The different version of chemprop have different atom features implemented

from model.models import MainModel_1_hyperopt as MainModel
from model.utils import create_logger
from model.datautils import Dataset, collate
from model.parsing import parse_train_args
from chemprop.models.mpn import MPN
from chemprop.args import TrainArgs
from chemprop.features.featurization import set_extra_atom_fdim

import torch
from torch import nn
import torch.distributed as dist
from torch.multiprocessing import Process
from torch.nn.parallel import DistributedDataParallel
from torch import optim
from torch.utils.data import DataLoader

import argparse
import socket
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime
from rdkit import RDLogger   #Delete, or put this somewhere else
RDLogger.DisableLog('rdApp.*') # turn off RDKit warning message 

#Double check that the correct atom features fuction is being used
# from chemprop.features.featurization import atom_features
# sys.stdout.write(inspect.getsource(atom_features))

def train(rank, world_size, hostname,args):

    hyperparams={
    'lr':args.lr,
    'epochs':args.n_epochs,
    'num_mods':args.n_fold,
    'batch_size':int(args.batch_size/world_size),
    'chemprop_feature_set':'01'
    }

    #Set directory
    path=os.getcwd()
    mydrive=os.path.abspath(os.path.join(path, os.pardir))
    ModelFolder=args.log_dir
    os.makedirs(f'{mydrive}/Models/{ModelFolder}',exist_ok=True)
    
    
    #Build Graph Convolutional neural network model Pytorch using ChemProp. This is also called an Message passing neural network (MPN)
    #and thus some of the naming in the model is called MPN
    # For details on Chemprop's MPN model, see: https://github.com/chemprop/chemprop/blob/master/chemprop/models/mpn.py
    # For details on important model arguments, see: https://github.com/chemprop/chemprop/blob/736530ac2ca42069088069f38571dc7a23e7e1d9/chemprop/args.py#L215
    # Separate MPNs can be implemented in same MPN class for multiple molecules: https://github.com/chemprop/chemprop/releases/tag/v1.1.0

    # Model hyperparameters
    hyper = TrainArgs()
    hyper.dataset_type=args.task
    hyper.hidden_size = args.hidden_size # Dimensionality of hidden layers in MPN
    hyper.depth = args.depth # Number of message passing steps
    hyper.bias = False # Whether to add bias to MPN linear layers
    hyper.dropout = args.dropout # MPN Dropout probability
    hyper.activation = 'ReLU' # MPN Activation function. Options: ['ReLU', 'LeakyReLU', 'PReLU', 'tanh', 'SELU', 'ELU']
    hyper.ffn_activation='ReLU' # FFN Activation function. Options: ['ReLU', 'LeakyReLU', 'PReLU', 'tanh', 'SELU', 'ELU']
    hyper.aggregation = args.graph_pool # MPN Aggregation scheme for atomic vectors into molecular vectors. options: ['mean', 'sum', 'norm']
    hyper.aggregation_norm = 100 # For MPN norm aggregation, number by which to divide summed up atomic features
    hyper.device=torch.device(rank%2)
    hyper.ffn_hidden_size = args.ffn_hidden_size # Dimensionality of hidden layer in readout MLP
    hyper.ffn_num_layers = args.ffn_depth
    hyper.output_size=args.n_out
    # Required settings to pass in custom atom and molecular features
    hyper.atom_descriptors = None # when using default atom features
    #hyper.atom_descriptors = 'feature' # when custom atom features passed in as np arrays
    hyper.overwrite_default_atom_features = False # Augment default atom features with custom ones instead of overwriting
    set_extra_atom_fdim(0)
    
    
    
    logger = create_logger('train', f'{mydrive}/Models/{args.log_dir}')
    
    logger.info('Hyperparamters Used During Training ...')
    logger.info(f'data_path = {args.data_path}')
    logger.info(f'log_dir = {args.log_dir}')
    logger.info(f'dataset_type = {args.task}')
    logger.info(f'hidden_size = {args.hidden_size}')
    logger.info(f'lr = {args.lr}')
    logger.info(f'depth = {args.depth}')
    logger.info(f'dropout = {args.dropout}')
    logger.info(f'activation = LeakyReLU')
    logger.info(f'aggregation = {args.graph_pool}')
    
    logger.info(f'ffn_hidden_size = {args.ffn_hidden_size}')
    logger.info(f'ffn_num_layers = {args.ffn_depth}')
    logger.info(f'output_size = {args.n_out}')
    logger.info(f'epochs = {args.n_epochs}')
    
    logger.info(f'num_mods = {args.n_fold}')
    logger.info(f'batch_size = {args.batch_size}')
    
    
    

    """Import Data Set"""
    data_path=args.data_path
    df = pd.read_csv(f'{mydrive}{data_path}')
    X=df.iloc[:,0].to_numpy()
    n=len(X)
    X=np.reshape(X,(n,1))
    y=df.iloc[:,1:].to_numpy()
    

    

    """ Model development. """

    def loop(model, loader, epoch, model_num, evaluation=False):

        if evaluation:
            model.eval()
            mode = "eval"
        else:
            model.train()
            mode = 'train'
        batch_losses = []

        for data in loader:

            x, y = data
            y=torch.tensor(y).to(device) 
            pred = model(x)
            loss = (pred.squeeze()-y.squeeze()).pow(2).mean()

            if not evaluation:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_losses.append(loss.item())

        return np.array(batch_losses).mean()


    #We will train num_mods random splits for the number of epochs specified below.
    
    """ Begin Training """
    
    num_mods=hyperparams['num_mods']
    model_train_losses=[];
    model_val_losses=[];


    for i in range(num_mods):
      
        os.makedirs(f'{mydrive}/Models/{ModelFolder}/model_{i}',exist_ok=True)

        # Create model
        train_losses = []
        val_losses = []
        model = MainModel(hyper)
        device=rank%2
        model=model.to(device)
        model=DistributedDataParallel(model,device_ids=[rank%2]) 
        optimizer = optim.Adam(model.parameters(), lr=hyperparams['lr'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True)

        # Randomly Split data
        X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.2,shuffle=True)
        X_train, X_val, y_train, y_val= train_test_split(X_train,y_train, test_size=0.125,shuffle=True)

        # Build dataset
        traindata = Dataset(X_train,y_train)
        valdata = Dataset(X_val,y_val)
        testdata = Dataset(X_test,y_test)
        
        # Build dataloader
        batchsize = hyperparams['batch_size']
        train_sampler=torch.utils.data.distributed.DistributedSampler(traindata,num_replicas=world_size,rank=rank%2)
        train_loader = DataLoader(dataset=traindata,batch_size=batchsize,collate_fn=collate,sampler=train_sampler,num_workers=world_size )
        val_loader = DataLoader(dataset=valdata,batch_size=batchsize,shuffle=True,collate_fn=collate,num_workers=world_size)
        test_loader = DataLoader(dataset=testdata,batch_size=batchsize,shuffle=True,collate_fn=collate)

        epochs=hyperparams['epochs']

        logger.info(f'Beginning to Train model {i} for {epochs} epochs')
        savepoints=[1,2,50,100,125,150,175]
        for epoch in range(epochs):
            train_loss = loop(model, train_loader, epoch,i)
            val_loss = loop(model, val_loader, epoch,i, evaluation=True)
            scheduler.step(val_loss)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            logger.info(f'Training epoch number {epoch}: Training Loss = {train_loss}')
            logger.info(f'Validation epoch number {epoch}: Val Loss = {val_loss}')
            if epoch in savepoints:
                #print the time
                now=datetime.now()
                current_time = now.strftime("%H:%M:%S")
                #logger.info(f'Current time= {current_time}')
                
                #save checkpoint
                torch.save(model.state_dict(), f'{mydrive}/Models/{ModelFolder}/model_{i}/model_{i}_epoch_{epoch}.pt')
                #logger.info(f'Reached savepoint epoch: {epoch}')


        torch.save(model.state_dict(), f'{mydrive}/Models/{ModelFolder}/model_{i}/model_{i}.pt')

        """Plot Results"""

        # import matplotlib.pyplot as plt

        # Training and validation loss during training
        plt.figure()
        plt.plot(train_losses)
        plt.plot(val_losses)
        plt.legend(['Training','Validation'])
        plt.title("MSE During Training")
        plt.ylabel("MSE")
        plt.xlabel("Epoch")
        #plt.ylim((0,10))
        plt.savefig( f'{mydrive}/Models/{ModelFolder}/model_{i}/epochcurve_{i}.png')
        plt.close()

        # True vs. predicted values for training and test data

        model.eval()
        y_train_all = np.array([0])
        y_train_pred = np.array([[0]])
        y_test_all = np.array([0])
        y_test_pred = np.array([[0]])

        # Get training predictions
        for data in train_loader:
            x_train, y_train = data
            y_pred = model(x_train)

            y_train_pred = np.concatenate((y_train_pred, y_pred.cpu().detach().numpy())) 
            
            y_train_all = np.concatenate((y_train_all, np.squeeze(y_train,axis=1)))

        # Get test predictions
        for data in test_loader:
            x_test, y_test = data
            y_pred = model(x_test)

            y_test_pred = np.concatenate((y_test_pred, y_pred.cpu().detach().numpy()))
            y_test_all = np.concatenate((y_test_all, np.squeeze(y_test,axis=1)))

        # Remove placeholder 0
        y_train_all = y_train_all[1:]
        y_test_all = y_test_all[1:]
        y_train_pred = y_train_pred[1:,:]
        y_test_pred = y_test_pred[1:,:]

        # Calculate MSE
        mse_train = np.mean( np.power( (y_train_pred.squeeze()-y_train_all.squeeze()), 2 ) )
        mse_test = np.mean( np.power( (y_test_pred.squeeze()-y_test_all.squeeze()), 2 ) )
        
        logger.info(f'Test MSE for model {i} is {mse_test}')
        # Plot
        plt.figure()
        plt.scatter(y_train_pred, y_train_all, label='train (MSE = {:.2f})'.format(mse_train), alpha=0.6)
        plt.scatter(y_test_pred, y_test_all, label='test (MSE = {:.2f})'.format(mse_test), alpha=0.6)
        plt.plot(y_train_pred,y_train_pred)
        plt.title("True vs. Predicted Value")
        plt.ylabel("True")
        plt.xlabel("Predicted")
        plt.legend()

        #Save arrays
        np.save(f'{mydrive}/Models/{ModelFolder}/model_{i}/y_train_true.npy',y_train_all)
        np.save(f'{mydrive}/Models/{ModelFolder}/model_{i}/y_train_pred.npy',y_train_pred)
        np.save(f'{mydrive}/Models/{ModelFolder}/model_{i}/y_test_true.npy',y_test_all)
        np.save(f'{mydrive}/Models/{ModelFolder}/model_{i}/y_test_pred.npy',y_test_pred)

        plt.savefig( f'{mydrive}/Models/{ModelFolder}/model_{i}/parity_{i}.png')
        plt.close()
        model_train_losses.append(train_losses);
        model_val_losses.append(val_losses);
    
    #Plot epoch curve
    
    
    model_train_losses_arr=np.array(model_train_losses) #2D array of # models by Num epochs
    model_val_losses_arr=np.array(model_val_losses)

    model_train_losses_arr_avg=np.mean(model_train_losses_arr,axis=0)
    model_val_losses_arr_avg=np.mean(model_val_losses_arr, axis=0)

    model_train_losses_arr_std=np.std(model_train_losses_arr,axis=0)
    model_val_losses_arr_std=np.std(model_val_losses_arr, axis=0)
    
    #save epoch curve arrays
    np.save(f'{mydrive}/Models/{ModelFolder}/train_losses_arr.npy',model_train_losses_arr)
    np.save(f'{mydrive}/Models/{ModelFolder}/val_losses_arr.npy',model_val_losses_arr)
    
    plt.figure()
    plt.plot(np.array(range(epochs)),model_train_losses_arr_avg.squeeze())
    plt.plot(np.array(range(epochs)),model_val_losses_arr_avg.squeeze())
    plt.legend(['Training','Validation'])
    plt.title("MSE During Training Average over 10 models")
    plt.ylabel("MSE")
    plt.xlabel("Epoch")
    #plt.ylim((0,3))
    epoch_nums = np.array([i for i in range(epochs)])
    
    plt.fill_between(epoch_nums, model_train_losses_arr_avg-model_train_losses_arr_std, model_train_losses_arr_avg+model_train_losses_arr_std, alpha=0.2)
    plt.fill_between(epoch_nums, model_val_losses_arr_avg-model_val_losses_arr_std, model_val_losses_arr_avg+model_val_losses_arr_std, alpha=0.2)
    plt.savefig( f'{mydrive}/Models/{ModelFolder}/AverageEpochCurve.png')
    plt.close()




def get_dist_env():
    if 'OMPI_COMM_WORLD_SIZE' in os.environ:
        world_size = int(os.getenv('OMPI_COMM_WORLD_SIZE'))
    else:
        world_size = int(os.getenv('SLURM_NTASKS'))

    if 'OMPI_COMM_WORLD_RANK' in os.environ:
        global_rank = int(os.getenv('OMPI_COMM_WORLD_RANK'))
    else:
        global_rank = int(os.getenv('SLURM_PROCID'))
    return global_rank, world_size


if __name__ == "__main__":
    # Dont change the following :
    global_rank, world_size = get_dist_env()
    
    if global_rank==0:
        print(f'The global rank is {global_rank}')
        print(f'The world_size is {world_size}')
    
    hostname = socket.gethostname()
    args = parse_train_args()
    # You have run dist.init_process_group to initialize the distributed environment
    # Always use NCCL as the backend. Gloo performance is pretty bad and MPI is currently
    # unsupported (for a number of reasons).     
    dist.init_process_group(backend='nccl', rank=global_rank, world_size=world_size)

    # now run your distributed training code
    print('here')
    train(global_rank, world_size, hostname,args)
    


