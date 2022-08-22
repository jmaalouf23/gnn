#!/bin/bash
#SBATCH --job-name H_dist
#SBATCH -o %x-%j.log
#SBATCH -N 2
#SBATCH -c 20
#SBATCH --tasks-per-node=2
#SBATCH --gres=gpu:volta:2

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

data_path='/Data/thermo/Hf.csv'
log_dir='H/4gpu_test_distributed'
split_path='thermo/80_10_20/split_'
lr=1e-3


#GNN Parameters-----------------
gnn_type='mpn'
depth=4
hidden_size=1200

# FFN Parameters----------------
ffn_depth=4
ffn_hiddensize=1200

#Entire Model Parameters--------
n_epochs=10
batch_size=50
num_workers=20
n_out=1
ensemble=1
n_fold=1
dropout=0
number_of_molecules=1
distributed=True

mpirun ${MPI_FLAGS} python train.py  --data_path $data_path --log_dir $log_dir --split_path $split_path --gnn_type $gnn_type --lr $lr --ffn_depth $ffn_depth --ffn_hidden_size $ffn_hiddensize --ensemble $ensemble  --n_fold $n_fold --n_out $n_out --n_epochs $n_epochs --dropout $dropout --number_of_molecules $number_of_molecules --num_workers $num_workers --distributed 
