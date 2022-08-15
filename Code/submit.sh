#!/bin/bash
#SBATCH --job-name test
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


data_path='/Data/Ylide_Gas_SpecialCases_Removed_redox.csv'
log_dir='ylide/test/expt8'
split_path='ylide/80_10_20/split_'
lr=1e-3


#GNN Parameters-----------------
gnn_type='mpn'
depth=3
hidden_size=600

# FFN Parameters----------------
ffn_depth=3
ffn_hiddensize=600

#Entire Model Parameters--------
n_epochs=2
batch_size=50
num_workers=20
n_out=1
ensemble=2
n_fold=2
dropout=0


mpirun ${MPI_FLAGS} python train.py  --data_path $data_path --log_dir $log_dir --split_path $split_path --gnn_type $gnn_type --lr $lr --ffn_depth $ffn_depth --ffn_hidden_size $ffn_hiddensize --ensemble $ensemble  --n_fold $n_fold --n_out $n_out --n_epochs $n_epochs --dropout $dropout
