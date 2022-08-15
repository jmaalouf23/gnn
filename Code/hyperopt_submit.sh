#!/bin/bash
#SBATCH --job-name hyperopt
#SBATCH -o %x-%j.log
#SBATCH -N 1
#SBATCH -c 20
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:volta:1

source /etc/profile
module load anaconda/2021a
source activate ml
module load mpi/openmpi-4.0
module load cuda/10.1
module load nccl/2.5.6-cuda10.1

export MPI_FLAGS="--tag-output --bind-to socket -map-by core -mca btl ^openib -mca pml ob1 -x PSM2_GPUDIRECT=1 -x NCCL_NET_GDR_LEVEL=5 -x NCCL_P2P_LEVEL=5 -x NCCL_NET_GDR_READ=1"

#Set some environment variables needed by torch.distributed
export MASTER_ADDR=$(hostname -s)
#Get unused port
export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

echo "MASTER_ADDR : ${MASTER_ADDR}"
echo "MASTER_PORT : ${MASTER_PORT}"

# Do not use the torch.distributed.launch utility. Use mpirun as shown below to launch your code. 



data_path='/Data/Ylide_Gas_SpecialCases_Removed.csv'
log_dir='ylide/hyperopt/expt0'
split_path='ylide/80_10_20/split_'
gnn_type='mpn'

#-----GNN Parameters-----------------
message='sum'
pool='sum'

#-----Entire Model Parameters--------
n_epochs=100
num_workers=20
n_out=3
n_ensemble=1
n_fold=5
#----Hyperopt Parameters------
n_trials=50



echo "Start time: $(date '+%Y-%m-%d_%H:%M:%S')"


mpirun ${MPI_FLAGS} python hyperopt.py --data_path $data_path --split_path $split_path --hyperopt_dir $log_dir --gnn_type $gnn_type --graph_pool $pool --ensemble $n_ensemble  --n_fold $n_fold --n_epochs $n_epochs --n_trials $n_trials --scaled_err True --n_out $n_out

echo "End time: $(date '+%Y-%m-%d_%H:%M:%S')"


