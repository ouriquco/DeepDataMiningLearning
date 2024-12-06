#!/bin/bash
#SBATCH --job-name=huggingface_qa_training
#SBATCH --output=./slurm_exp_logs/output/huggingface_qa_training.out
#SBATCH --error=./slurm_exp_logs/error/huggingface_qa_traininge.err
#SBATCH -c 2
#SBATCH -n 1
#SBATCH --partition=gpu
#SBATCH --nodelist=g16

# Choose correct MIG partition if applicable
# export CUDA_VISIBLE_DEVICES=0

#Initialize Conda
source /home/010892622/miniconda3/etc/profile.d/conda.sh

# # set up the environment
# conda activate mtr2

# training script
cd /data/cmpe258-sp24/010892622/DeepDataMiningLearning/nlp/
python huggingfaceSequence4_qa.py
