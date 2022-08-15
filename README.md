# ylide_gnn

This repository uses chemprop's development of a message passing neural network (https://github.com/chemprop/chemprop) wrapped in custom code to add an mlp readout layer able to map to multiple properties at once to predict thermodynamic properties of ylide redox mediators. Currently, the models have been trained to predict properties that are important for redox mediator performance including redox potential, deprotonation free energy, and Hydrogen abstraction energy.

## How to Use

To train a model, run `python train.py --data_path data_path --log_dir log_dir`

`data_path` is the path to the directory where the .csv file containing the data and labels

Additional specifications that can be made by passing in arguments can be found in `Code/model/parsing.py` and can be implemented as follows. 

`python train.py --data_path data_path --log_dir log_dir --lr 1e-3`

In this case we are setting the initial learning rate to 1e-3.


### Data
Data should come in the form of a .csv file where the first column are SMILES strings of the molecules, and each target property is an additional column. The number of target properties read should be specified by the `n_out` argument. For example, if `n_out' is 2, the script will read the next two columns after the SMILES string column, even if there are 3 additional columns after the SMILES string column.

### Data Splits
If `split_path` is not specified, the script will perform a random split on the data in a 80:10:20 ratio making up the training, validation, and test sets respectively.

The user can also specify a directory with `split_path` where numpy arrays containing the split indices can be found. Each split should be named `split_i.npy` where i is the index of the split starting at 0. Each `split_i.npy` should contain 3 numpy arrays, the train indices, validation indices, and the test indices. The indices correspond to the indices in the data.csv file. The `split_path` should include `split_` in the name.

### Multi GPU Training

MultiGPU training is built in and run with PyTorch's DistributedDataParallel module. Code automatically detects number of nodes and GPUs allocated to the job and splits up training accordingly. This can usually be specified by the appropriate `#SBATCH` command if running on a server with slurm. 

### Hyperparameter Tuning

Hyperparameters can be tuned by running `python hyperopt.py` which uses optuna to implement a Bayesian Hyperparameter optimization algorithm. 

