#!/bin/bash
#SBATCH --time=200:10:00

#SBATCH --gres=gpu:1
#SBATCH -p new
#SBATCH -w ttnusa12


python ChatCRS.py --config configs/CHATCRS_DuE_CRS.yaml  --openai_api_key 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 --quick_test 5 

python ChatCRS.py --config configs/CHATCRS_DuE_KNOWLEDGE.yaml  --openai_api_key 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 --quick_test 5 

python ChatCRS.py --config configs/CHATCRS_DuE_CHAT.yaml  --openai_api_key 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 --quick_test 5

python ChatCRS.py --config configs/CHATCRS_DuE_TOPIC.yaml  --openai_api_key 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 --quick_test 5 

python ChatCRS.py --config configs/CHATCRS_DuE_GOAL.yaml  --openai_api_key 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 --quick_test 5 

python ChatCRS.py --config configs/CHATCRS_DuE_REC.yaml  --openai_api_key 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 --quick_test 5 

python ChatCRS.py --config configs/LLAMA_DuE_CRS.yaml  --quick_test 5 --huggingface_key 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