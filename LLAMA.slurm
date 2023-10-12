#!/bin/bash
#SBATCH --time=200:10:00

#SBATCH --gres=gpu:1
#SBATCH -p new
#SBATCH -w ttnusa12


python models.py


# python finetune.py \
#     --base_model 'decapoda-research/llama-7b-hf' \
#     --data_path 'yahma/alpaca-cleaned' \
#     --output_dir './lora-alpaca'



# python generate.py \
#     --load_8bit \
#     --base_model 'decapoda-research/llama-7b-hf' \
#     --lora_weights 'tloen/alpaca-lora-7b'