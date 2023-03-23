#!/bin/bash

#SBATCH -o /home/scratch/full_test.out
#SBATCH -p partition=ba100
#SBATCH --gpus=1

export CUDA_VISIBILE_DEVICES=2
export CHECKPOINT_DIR=/brtx/603-nvme2/jgualla1/full_train_5/ 
export TRAINING_CONFIG=/home/jgualla1/why_ambiguity_2023/miso_uds/miso/training_config/transformer/no_syntax/base.jsonnet

./experiments/decomp_train.sh -a train
