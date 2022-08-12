import os
import sys
sys.path.insert(1,'/data1/groups/manthiram_lab/Utils')
import logging
import numpy as np
import matplotlib.pyplot as plt
import plot as pl


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


def make_learning_curve(epochs: int, train_losses: list, val_losses: list, save_path: str,fill_between: bool = False,model_train_losses_arr_avg=[], model_train_losses_arr_std=[], model_val_losses_arr_avg=[],model_val_losses_arr_std=[]):
    
    """
    Plots learning curve given val and training loss

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
    
    
def make_parity(y_train_pred,y_train_all,y_test_pred,y_test_all,save_path,mse_train, mse_test):
    
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
    
    
    