#!/bin/bash
#PBS -N CHATGPT_full_CRS
#PBS -l select=1:ncpus=16:ngpus=1:mem=100gb
#PBS -l walltime=40:00:00
#PBS -j oe
#PBS -P personal-e0134107
#PBS -q normal
#PBS -o log/CHATGPT-Sat-CRS.log


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

# for task in CRS CHAT GOAL REC TOPIC KNOWLEDGE
# do 
#     python ChatCRS.py --config configs/CHATCRS_DuE_${task}.yaml  --model CHATGPT --openai_api_key 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
# done


python ChatCRS.py --config configs/CHATCRS_DuE_CRS.yaml  --model CHATGPT --openai_api_key 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 --quick_test 1000
# python ChatCRS.py --config configs/CHATCRS_DuE_CRS_GOAL.yaml  --model CHATGPT --openai_api_key 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
# python ChatCRS.py --config configs/CHATCRS_DuE_CRS_REC.yaml  --model CHATGPT --openai_api_key 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
# python ChatCRS.py --config configs/CHATCRS_DuE_CRS_TOPIC.yaml  --model CHATGPT --openai_api_key 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

# python ChatCRS.py --config configs/CHATCRS_DuE_GOAL.yaml  --model CHATGPT --openai_api_key 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
# python ChatCRS.py --config configs/CHATCRS_DuE_REC.yaml  --model CHATGPT --openai_api_key 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
# python ChatCRS.py --config configs/CHATCRS_DuE_TOPIC.yaml  --model CHATGPT --openai_api_key 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

# for task in GOAL TOPIC REC
# do
#     python ChatCRS.py --config configs/CHATCRS_DuE_CRS_${task}.yaml  --model CHATGPT --openai_api_key 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
# done