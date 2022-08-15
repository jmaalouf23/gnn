import os
import sys
sys.path.insert(1,'/data1/groups/manthiram_lab/Utils')

from .datautils import Dataset, collate

import logging
import numpy as np
import matplotlib.pyplot as plt
import plot as pl
import torch
from torch import nn
import torch.distributed as dist
from torch.multiprocessing import Process
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from argparse import Namespace

def create_logger(name: str, log_dir: str = None) -> logging.Logger:
    """
    Creates a logger with a stream handler and file handler.
    
    Params
    name: The name of the logger.
    log_dir: The directory in which to save the logs.
    
    Returns
    The logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Set logger
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    fh = logging.FileHandler(os.path.join(log_dir, name + '.log'))
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    return logger

def loop(model, loader, loss, device, optimizer, evaluation=False): #consider changing name to eval
    
    """
    Given a model, if evaliation = False this function "trains" the model with a specific loss function
    and updates the model prameters. If evaluation = True then function evaluates the model on the data in the data loader.
    
    Params
    
    model:  a PyTorch model
    loader: a PyTorch data loader
    device: the device on which the model and data reside
    optimizer : PyTorch optimizer used to update the model weights
    evaluation : whether or not the model is in eval or training mode
    
    returns: the average loss over the entire batch as a numpy array.
    
    """

    if evaluation:
        model.eval()
        mode = "eval"
    else:
        model.train()
        mode = "train"
    batch_losses = []

    for data in loader:
        x, y = data
        y=torch.tensor(y).to(device) 
        pred = model(x)
        #loss = (pred.squeeze()-y.squeeze()).pow(2).mean()
        result=loss(pred.float(),y.float())
        if not evaluation:
            optimizer.zero_grad()
            result.backward()
            optimizer.step()

        batch_losses.append(result.item())

    return np.array(batch_losses).mean()   



def get_loss_func(args: Namespace) -> nn.Module:
    
    """
    Gets the loss function corresponding to a given dataset type.

    :param args: Namespace containing the dataset type ("classification" or "regression").
    :return: A PyTorch loss function.
    """
    if args.task == 'classification':
        return nn.BCELoss(reduction='mean')

    if args.task == 'regression':
        return nn.MSELoss(reduction='mean')

    
def construct_loader(X, y, world_size, rank, num_fold, args):
    
    mydrive=os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    batchsize=int(args.batch_size/world_size)
    """Split Data"""
    if args.split_path is not None:
        #split data based on split indices
        splits = np.load(f'{mydrive}/Splits/{args.split_path}{num_fold}.npy',allow_pickle=True)
        train_indices = splits[0]
        val_indices   = splits[1]
        test_indices  = splits[2]

        X_train, y_train = (X[train_indices,:],y[train_indices,:])
        X_val, y_val = (X[val_indices,:],y[val_indices,:])
        X_test,y_test= (X[test_indices,:],y[test_indices,:])
    else:
    # If split path not specified then randomly split data
        X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.2,shuffle=True)
        X_train, X_val, y_train, y_val= train_test_split(X_train,y_train, test_size=0.125,shuffle=True)                

    # Build dataset
    traindata = Dataset(X_train,y_train)
    valdata = Dataset(X_val,y_val)
    testdata = Dataset(X_test,y_test)

    # Build dataloader
    train_sampler=torch.utils.data.distributed.DistributedSampler(traindata,num_replicas=world_size,rank=rank%2)
    train_loader = DataLoader(dataset=traindata,batch_size=batchsize,collate_fn=collate,sampler=train_sampler,num_workers=world_size )
    val_loader = DataLoader(dataset=valdata,batch_size=batchsize,shuffle=True,collate_fn=collate,num_workers=world_size)
    test_loader = DataLoader(dataset=testdata,batch_size=batchsize,shuffle=True,collate_fn=collate) 
    
    
    
    
    
    return train_loader, val_loader, test_loader
    
def get_dist_env() -> tuple:
    
    """
    Get environment variables necessary for multi GPU training with PyTorch.
    
    Returns:
    
    world_size  : Number of processes for your training. Often the number of GPUs you want to allocate for training.
    global_rank : The unique ID given to a process, so that other processes know how to identify a particular process. 
                  Local rank is the a unique local ID for processes running in a single node.

    Ex: Suppose we run our training in 2 nodes and each with 4 GPUs. The world size is 4*2=8. 
    The ranks for the processes will be [0, 1, 2, 3, 4, 5, 6, 7]. In each node, the local rank will be [0, 1, 2, 3]
    
    """
    
    if 'OMPI_COMM_WORLD_SIZE' in os.environ:
        world_size = int(os.getenv('OMPI_COMM_WORLD_SIZE'))
    else:
        world_size = int(os.getenv('SLURM_NTASKS'))

    if 'OMPI_COMM_WORLD_RANK' in os.environ:
        global_rank = int(os.getenv('OMPI_COMM_WORLD_RANK'))
    else:
        global_rank = int(os.getenv('SLURM_PROCID'))
    return global_rank, world_size


def make_learning_curve(epochs: int, train_losses: list, val_losses: list, save_path: str,fill_between: bool = False,model_train_losses_arr_avg=[], model_train_losses_arr_std=[], model_val_losses_arr_avg=[],model_val_losses_arr_std=[]) -> None:
    
    """
    Plots learning curve given val and training loss arrays

    :param epochs: number of epochs.
    :param train_losses: list containing training loss for each epoch
    :param val_losses: list containing val loss for each epoch.
    :param save_path: directory where figure will be saved
    :param fill_between: whether or not the learning curve will include shaded regions indicating error bars
    
    Returns
    Does not return anything.
    """
    
    fig,ax=plt.subplots(1,1,figsize=(6,6))
    plt.plot(range(epochs),train_losses)
    plt.plot(range(epochs),val_losses)
    plt.legend(['Training','Validation'])
        
    if fill_between:
        epoch_nums = np.array([i for i in range(epochs)])
        plt.fill_between(epoch_nums, model_train_losses_arr_avg-model_train_losses_arr_std, model_train_losses_arr_avg+model_train_losses_arr_std, alpha=0.2)
        plt.fill_between(epoch_nums, model_val_losses_arr_avg-model_val_losses_arr_std, model_val_losses_arr_avg+model_val_losses_arr_std, alpha=0.2)
    pl.set(ax,title="MSE During Training",xlabel="Epoch",ylabel="MSE",labelsize=14,fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    
def make_parity(y_train_pred:np.ndarray ,y_train_all: np.ndarray ,y_test_pred: np.ndarray,y_test_all: np.ndarray,save_path: str, mse_train: np.ndarray , mse_test: np.ndarray ) -> None:
    
    """
    Creates a parity plot of training and test predictions from a machine learning model trained by ylide_gnn code. 
    
    Params
    
    y_train_pred : training predictions list
    y_train_all  : training true values list
    y_test_pred  : test prediction values list
    y_test_all   : test true values list
    mse_train    : array of mean standard error of training data 
    mse_test     : array of mean standard error of test data
    
    Returns
    Does not return anything.
    """
    
    n,d=np.shape(y_test_pred)
    fig,axs=plt.subplots(1,d,figsize=(20,7))
    
    if hasattr(axs, '__iter__') == False:
        axs=[axs]

    for i,a in enumerate(axs):
        axs[i].scatter(y_train_all[:,i], y_train_pred[:,i],label=f'train (MSE = {mse_train[i]:.3f})', alpha=0.6)
        axs[i].scatter(y_test_all[:,i], y_test_pred[:,i], label=f'test (MSE = {mse_test[i]:.3f})', alpha=0.6)
        axs[i].plot(y_train_all[:,i],y_train_all[:,i])
        axs[i].legend()
        pl.set(axs[i],title="True vs. Predicted Value",xlabel="True",ylabel="Predicted",labelsize=16,fontsize=16)
    plt.subplots_adjust(wspace=0.4)
    fig.savefig(save_path,bbox_inches='tight')
    plt.close(fig)
    
    
    