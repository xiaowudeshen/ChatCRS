#!/bin/bash
#PBS -N LLAMA2_full_CRS
#PBS -l select=1:ncpus=16:ngpus=1:mem=100gb
#PBS -l walltime=20:00:00
#PBS -j oe
#PBS -P personal-e0134107
#PBS -q ai
#PBS -o log/LLAMA2-normal-crs.log


# The following environment variables will be preset after your job is submitted and started.
################################################# 
echo PBS: qsub is running on $PBS_O_HOST
echo PBS: executing queue is $PBS_QUEUE
echo PBS: working directory is $PBS_O_WORKDIR
echo PBS: job identifier is $PBS_JOBID
echo PBS: job name is $PBS_JOBNAME
echo PBS: node file is $PBS_NODEFILE
echo PBS: current home directory is $PBS_O_HOME
echo PBS: PATH = $PBS_O_PATH
#################################################


source activate lora


cd $PBS_O_WORKDIR
# cd alpaca_finetuning_v1
[ -d log ] || mkdir log

nvidia-smi

python ChatCRS.py --config configs/CHATCRS_DuE_CRS.yaml  --model LLAMA2 --huggingface_key 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

# python ChatCRS.py --config configs/CHATCRS_DuE_CHAT.yaml  --model LLAMA2 --huggingface_key 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

# python ChatCRS.py --config configs/CHATCRS_DuE_GOAL.yaml  --model LLAMA2 --huggingface_key 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

# python ChatCRS.py --config configs/CHATCRS_DuE_TOPIC.yaml  --model LLAMA2 --huggingface_key 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

# python ChatCRS.py --config configs/CHATCRS_DuE_KNOWLEDGE.yaml  --model LLAMA2 --huggingface_key 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

# python ChatCRS.py --config configs/CHATCRS_DuE_REC.yaml  --model LLAMA2 --huggingface_key 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