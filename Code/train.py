import os
import sys

from model.models import MainModel,MainModel_2
from model.utils import create_logger, make_learning_curve, make_parity, loop, get_dist_env, get_loss_func, construct_loader, Standardizer
from model.parsing import parse_train_args

import torch
from torch import nn, optim
import torch.distributed as dist
from torch.multiprocessing import Process
from torch.nn.parallel import DistributedDataParallel

import socket
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from rdkit import RDLogger   
RDLogger.DisableLog('rdApp.*') # turn off RDKit warning message 


def train(rank, world_size, hostname, args):
    
    if world_size > 1:
        assert  args.distributed == True ,'WORLD SIZE IS GREATER THAN 1 BUT DISTRIBUTED SET TO FALSE'
    elif world_size == 1:
        assert args.distributed == False,'WORLD SIZE IS 1 BUT DISTRIBUTED SET TO TRUE'
    
    path=os.getcwd()
    mydrive=os.path.abspath(os.path.join(path, os.pardir))
    ModelFolder=args.log_dir
    os.makedirs(f'{mydrive}/Models/{ModelFolder}',exist_ok=True)   
    
    
    logger = create_logger('train', f'{mydrive}/Models/{args.log_dir}')
       

    """Import Data Set"""
    df = pd.read_csv(f'{mydrive}{args.data_path}')
    X=df.iloc[:,0:args.number_of_molecules].to_numpy()
    n=np.shape(X)
    X=np.reshape(X,(n[0],args.number_of_molecules))
    y=df.iloc[:,args.number_of_molecules:args.number_of_molecules+args.n_out + 1].to_numpy()
    

    """ Model development. """
    
    if rank==0:
        logger.info(f'Beginning Training on {world_size} GPUs ...')
        logger.info(f'Hyperparamerters are ...')
        
        for arg in vars(args):
            logger.info(f'{arg}: {getattr(args, arg)}')
    
    model_train_losses=[];
    model_val_losses=[];


    for fold in range(args.n_fold):
        
        device=rank%2
        if args.distributed:
            train_loader, train_loader_all,train_sampler,val_loader, test_loader, mu, std = construct_loader(X, y, world_size, rank, fold, args) 
        else:
            train_loader,val_loader, test_loader, mu, std = construct_loader(X, y, world_size, rank, fold, args)
            train_loader_all=train_loader
            train_sampler=None
        standardizer= Standardizer(mu,std,device)
        
        if rank==0:
            logger.info(f'Mean used for standardization: {mu}')
            logger.info(f'Standard deviation used for standardization: {std}')
        
        os.makedirs(f'{mydrive}/Models/{ModelFolder}/fold_{fold}',exist_ok=True)

       
        np.save(f'{mydrive}/Models/{ModelFolder}/fold_{fold}/mean.npy',mu)
        np.save(f'{mydrive}/Models/{ModelFolder}/fold_{fold}/std.npy',std)
                
        for model_index in range(args.ensemble):
      
            os.makedirs(f'{mydrive}/Models/{ModelFolder}/fold_{fold}/model_{model_index}',exist_ok=True)
        

            # Create model
            train_losses = []
            val_losses = []
            
            if args.gnn_type=='mpn':
                model = MainModel(args,rank)
            elif args.gnn_type=='mpn2':
                model = MainModel_2(args,rank)
                
            model=model.to(device)
            if args.distributed:
                model=DistributedDataParallel(model,device_ids=[rank%2],find_unused_parameters=True) 
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=8, verbose=True)
            loss=get_loss_func(args)
            
            if rank ==0:
                logger.info(f'\nOptimizer parameters are:\n{optimizer}\n')
                logger.info(f'Scheduler state dict is:')
                
            
                for key, value in scheduler.state_dict().items():
                    logger.info(f'{key}: {value}')
                logger.info('\n')


                logger.info(f'Beginning to Train fold {fold} model {model_index} for {args.n_epochs} epochs \n')

            for epoch in range(args.n_epochs):
                
                train_loss = loop(epoch, model, train_loader, train_sampler, loss, device, optimizer,standardizer,args)
                val_loss = loop(epoch, model, val_loader, train_sampler, loss, device, optimizer, standardizer, args,evaluation=True)
                scheduler.step(val_loss)

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                
                if rank==0:
                    logger.info(f'Training epoch number {epoch}: Training Loss = {train_loss}')
                    logger.info(f'Validation epoch number {epoch}: Val Loss = {val_loss}')

                if val_loss == min(val_losses):
                    
                    if args.distributed and rank==0:
                        #torch.distributed.barrier()
                        torch.save(model.module.state_dict(), f'{mydrive}/Models/{ModelFolder}/fold_{fold}/model_{model_index}/best_model.pt')
                    elif not args.distributed: 
                        torch.save(model.state_dict(), f'{mydrive}/Models/{ModelFolder}/fold_{fold}/model_{model_index}/best_model.pt')
            
            if rank==0:
                logger.info(f'Best model occurs at epoch {val_losses.index(min(val_losses))} with val loss = {min(val_losses)} ')
            

                """Load Best Model & Get Predictions"""

                #if rank ==0:

                if args.gnn_type=='mpn':
                    model = MainModel(args,rank).to(rank%2)
                elif args.gnn_type=='mpn2':
                    model = MainModel_2(args,rank).to(rank%2)

                state_dict=torch.load(f'{mydrive}/Models/{ModelFolder}/fold_{fold}/model_{model_index}/best_model.pt', map_location=f'cuda:{device}')
                model.load_state_dict(state_dict)
                model.eval()     


                y_train_all  = []
                y_train_pred = []
                y_test_all   = []
                y_test_pred  = []

                for data in train_loader_all:
                    x_train, y_train = data
                    y_train_all.append(np.vstack(tuple(y_train)))
                    y_pred = standardizer(model(x_train),rev=True)
                    y_train_pred.append(y_pred.cpu().detach().numpy())


                y_train_pred= np.vstack(tuple(y_train_pred))
                y_train_all= np.vstack(tuple(y_train_all))


                for data in test_loader:
                    x_test, y_test = data
                    y_test_all.append(np.vstack(tuple(y_test)))
                    y_pred = standardizer(model(x_test),rev=True)
                    y_test_pred.append(y_pred.cpu().detach().numpy())

                y_test_pred= np.vstack(tuple(y_test_pred))
                y_test_all= np.vstack(tuple(y_test_all)) 

                mse_train = np.mean( np.power( y_train_pred-y_train_all, 2 ), axis = 0)
                mse_test = np.mean( np.power( y_test_pred-y_test_all, 2 ), axis = 0)

                mae_train = np.mean( np.abs(y_train_pred-y_train_all), axis = 0)
                mae_test = np.mean( np.abs(y_test_pred-y_test_all), axis = 0)

                logger.info(f'Test MAE for fold {fold} model {model_index} is {mae_test} \n')


                """Plot Results"""
                if args.include_plots:
                    make_learning_curve(args.n_epochs,train_losses, val_losses, f'{mydrive}/Models/{ModelFolder}/fold_{fold}/model_{model_index}/epochcurve_{fold}_{model_index}.png')
                    make_parity(y_train_pred,y_train_all,y_test_pred,y_test_all,f'{mydrive}/Models/{ModelFolder}/fold_{fold}/model_{model_index}/parity_{fold}_{model_index}.png',mae_train, mae_test)

                model_train_losses.append(train_losses);
                model_val_losses.append(val_losses);

                np.save(f'{mydrive}/Models/{ModelFolder}/fold_{fold}/model_{model_index}/y_train_true.npy',y_train_all)
                np.save(f'{mydrive}/Models/{ModelFolder}/fold_{fold}/model_{model_index}/y_train_pred.npy',y_train_pred)
                np.save(f'{mydrive}/Models/{ModelFolder}/fold_{fold}/model_{model_index}/y_test_true.npy',y_test_all)
                np.save(f'{mydrive}/Models/{ModelFolder}/fold_{fold}/model_{model_index}/y_test_pred.npy',y_test_pred)


            #Plot epoch curve averaged over all ensemble models within a given fold

            model_train_losses_arr=np.array(model_train_losses) #2D array of Num_ensembles by Num_epochs
            model_val_losses_arr=np.array(model_val_losses)

            model_train_losses_arr_avg=np.mean(model_train_losses_arr,axis=0)
            model_val_losses_arr_avg=np.mean(model_val_losses_arr, axis=0)

            model_train_losses_arr_std=np.std(model_train_losses_arr,axis=0)
            model_val_losses_arr_std=np.std(model_val_losses_arr, axis=0)

            #save average epoch curve arrays
            np.save(f'{mydrive}/Models/{ModelFolder}/fold_{fold}/train_losses_arr.npy',model_train_losses_arr)
            np.save(f'{mydrive}/Models/{ModelFolder}/fold_{fold}/val_losses_arr.npy',model_val_losses_arr)
            
            if args.include_plots:
                make_learning_curve(args.n_epochs, model_train_losses_arr_avg.squeeze(), model_val_losses_arr_avg.squeeze(), f'{mydrive}/Models/{ModelFolder}/fold_{fold}/AverageEpochCurve.png',fill_between=True,model_train_losses_arr_avg= model_train_losses_arr_avg, model_train_losses_arr_std= model_train_losses_arr_std, model_val_losses_arr_avg= model_val_losses_arr_avg,model_val_losses_arr_std= model_val_losses_arr_std)


    

if __name__ == "__main__":

    global_rank, world_size = get_dist_env()    
    
    hostname = socket.gethostname()
    args = parse_train_args()
    
    # Run dist.init_process_group to initialize the distributed environment always use NCCL as the backend. 
    # Gloo performance is pretty bad and MPI is currently unsupported (for a number of reasons).
    
    dist.init_process_group(backend='nccl', rank=global_rank, world_size=world_size)
    
    train(global_rank, world_size, hostname, args)
    


