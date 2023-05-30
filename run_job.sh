#!/bin/bash

#SBATCH -o /home/scratch/full_test.out
#SBATCH -p partition=ba100
#SBATCH --gpus=4

export CUDA_VISIBLE_DEVICES=4
#export MODEL_DIR=/brtx/603-nvme2/jgualla1/$1/ckpt.
export CHECKPOINT_DIR=/brtx/603-nvme2/jgualla1/$1
#export TEST_DATA=dev 
#export TRAINING_CONFIG=/home/jgualla1/why_ambiguity_2023/miso_uds/test/configs/overfit_decomp_transformer.jsonnet
export TRAINING_CONFIG=/home/jgualla1/why_ambiguity_2023/miso_uds/miso/training_config/transformer/no_syntax/base.jsonnet

./experiments/decomp_train.sh -a train
#./experiments/decomp_train.sh -a spr_eval #/brtx/603-nvme2/jgualla1/full_run_bool_loss
