import os
import sys
import math
import numpy as np
import pandas as pd
import optuna
import joblib
import socket

from model.models import MainModel
from model.utils import create_logger, make_learning_curve,make_parity
from model.datautils import Dataset, collate
from model.parsing import parse_train_args,add_train_args, modify_train_args

import torch
from torch import nn, optim
import torch.distributed as dist
from torch.multiprocessing import Process
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from argparse import ArgumentParser
import logging

def optimize(trial, args, rank, world_size,hostname):

    setattr(args, 'hidden_size', int(trial.suggest_discrete_uniform('hidden_size', 300, 1200, 300)))
    setattr(args, 'depth', int(trial.suggest_discrete_uniform('depth', 2, 6, 1)))
    #setattr(args, 'dropout', int(trial.suggest_discrete_uniform('dropout', 0, 1, 0.2)))
    setattr(args, 'lr', trial.suggest_loguniform('lr', 1e-5, 1e-3))
    setattr(args, 'batch_size', int(trial.suggest_categorical('batch_size', [25, 50, 100])))
    #setattr(args, 'graph_pool', trial.suggest_categorical('graph_pool', ['sum', 'mean', 'max', 'attn', 'set2set']))
    setattr(args, 'ffn_hidden_size', int(trial.suggest_discrete_uniform('ffn_hidden_size', 300, 2100, 300)))
    setattr(args, 'ffn_depth', int(trial.suggest_discrete_uniform('ffn_depth', 2, 6, 1)))

    setattr(args, 'log_dir', os.path.join(args.hyperopt_dir, str(trial._trial_id)))
    
    
    lr=args.lr
    num_folds=args.n_fold
    epochs=args.n_epochs
    batchsize=int(args.batch_size/world_size)
    
    modify_train_args(args)
    best_val_loss_lst = []
    torch.manual_seed(args.pytorch_seed)
    
    """Import Data Set"""
    data_path=args.data_path
    df = pd.read_csv(f'{mydrive}{data_path}')
    X=df.iloc[:,0].to_numpy()
    n=len(X)
    X=np.reshape(X,(n,1))
    y=df.iloc[:,1:args.n_out + 1].to_numpy()
    
    
    
    for n_fold in range(args.n_fold):
        """Split Data"""
        
        if args.split_path is not None:
            #split data based on split indices
            splits = np.load(f'{mydrive}/Splits/{args.split_path}{n_fold}.npy',allow_pickle=True)
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
        

        # create model, optimizer, scheduler, and loss fn
        for model_index in range(args.ensemble):
            
            model = MainModel(args,rank)
            device=rank%2
            model=model.to(device)
            model=DistributedDataParallel(model,device_ids=[rank%2]) 
            optimizer = optim.Adam(model.parameters(), lr=lr)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=8, verbose=True)    
            
            best_val_loss = math.inf
            best_epoch = 0

        # record args, optimizer, and scheduler info

        # train
            for epoch in range(0, args.n_epochs):
                train_loss = loop(model, train_loader, epoch, device, optimizer)
                val_loss = loop(model, val_loader, epoch, device, optimizer, evaluation=True)
                    
                if val_loss <= best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch

            best_val_loss_lst.append(best_val_loss)

    return sum(best_val_loss_lst)/len(best_val_loss_lst)



def loop(model, loader, epoch, device, optimizer, evaluation=False): #consider changing name to eval

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
        loss = (pred.squeeze()-y.squeeze()).pow(2).mean()

        if not evaluation:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        batch_losses.append(loss.item())

    return np.array(batch_losses).mean() 


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




if __name__ == '__main__':

    parser = ArgumentParser()
    add_train_args(parser)
    parser.add_argument('--hyperopt_dir', type=str,
                        help='Directory to save all results')
    parser.add_argument('--n_trials', type=int, default=25,
                        help='Number of hyperparameter choices to try')
    parser.add_argument('--restart', action='store_true', default=False,
                        help='Whether or not to resume study from previous .pkl file')
    args = parser.parse_args()
    
    path=os.getcwd()
    mydrive=os.path.abspath(os.path.join(path, os.pardir))
    ModelFolder=args.hyperopt_dir
    working_dir=f'{mydrive}/Models/{ModelFolder}'
    os.makedirs(working_dir,exist_ok=True) 

    if not os.path.exists(args.hyperopt_dir):
        os.makedirs(args.hyperopt_dir)

    logger = logging.getLogger()
    
    
    
    logger.setLevel(logging.INFO)  # Setup the root logger.
    #logger.addHandler(logging.FileHandler(os.path.join(args.hyperopt_dir, "hyperopt.log"), mode="w"))
    logger.addHandler(logging.FileHandler(os.path.join(working_dir, "hyperopt.log"), mode="w"))

    optuna.logging.enable_propagation()  # Propagate logs to the root logger.
    optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.

    if args.restart:
        #study = joblib.load(os.path.join(args.hyperopt_dir, "study.pkl"))
        study = joblib.load(os.path.join(working_dir, "study.pkl"))
    else:
        study = optuna.create_study(
            pruner=optuna.pruners.HyperbandPruner(min_resource=5, max_resource=args.n_epochs, reduction_factor=2),
            sampler=optuna.samplers.CmaEsSampler(warn_independent_sampling=False)
        )
    #joblib.dump(study, os.path.join(args.hyperopt_dir, "study.pkl"))
    joblib.dump(study, os.path.join(working_dir, "study.pkl"))
    
    
    
    global_rank, world_size = get_dist_env()    
    if global_rank==0:
        print(f'The global rank is {global_rank}')
        print(f'The world_size is {world_size}')
    
    hostname = socket.gethostname()
    # You have run dist.init_process_group to initialize the distributed environment always use NCCL as the backend. 
    #Gloo performance is pretty bad and MPI is currently unsupported (for a number of reasons).     
    
    dist.init_process_group(backend='nccl', rank=global_rank, world_size=world_size)

    logger.info("Running optimization...")
    study.optimize(lambda trial: optimize(trial, args, global_rank, world_size,hostname), n_trials=args.n_trials)
