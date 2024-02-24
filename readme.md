# ChatCRS
# ChatCRS: Incorporating External Knowledge and Goal Guidance for Conversational Recommender Systems

## Abstract:
This paper aims to efficiently enable Large Language Models (LLMs) to use external knowledge and goal guidance in conversational recommender system (CRS) tasks. Advanced LLMs (e.g., ChatGPT) are limited in CRS tasks for 1) generating grounded responses with recommendation-oriented knowledge,
or 2) proactively guiding users through different dialogue goals. In this work, we first analyze those limitations through a comprehensive evaluation to assess LLMs' intrinsic capabilities, showing the necessity of incorporating external knowledge and goal guidance which contribute significantly to the recommendation accuracy and language quality. In light of this finding, we propose a novel ChatCRS framework to decompose the complex CRS task into several sub-tasks through the implementation of 1) a knowledge retrieval agent using a tool-augmented approach to reason over external Knowledge Bases and 2) a goal-planning agent for dialogue goal prediction. Experimental results on two CRS datasets reveal that ChatCRS sets new state-of-the-art benchmarks, improving language quality of informativeness by 17% and proactivity by 27%, and achieving a tenfold enhancement in recommendation accuracy over LLM-based CRS

## Method:


## Citation
Our ChatCRS is out and it can be accessed with the [LINK]()
```console
@article{ChatCRS
}
```


## Experiments
**Dataset**
```console
❱❱❱ python dataTloader.py
```
* --Create data for DuRecDial, data will be saved in the "data_p" folder
```console
❱❱❱ python data_loader_T.py
```
* --Create data for TG-Redial, data will be saved in the "data" folder

**Empirical Analysis**
```console
## Running direct Genenration using OPENAI ChatGPT/GPT4
❱❱❱ python ChatCRS.py --config configs_DG/CHATCRS_DuE_CRS_GOAL.yaml  --model CHATGPT --openai_api_key [YOUR OPENAI API KEY] --save_dir result_DG
❱❱❱ python ChatCRS.py --config configs_DG/CHATCRS_DuE_CRS_KNOWLEDGE.yaml  --model CHATGPT --openai_api_key  [YOUR OPENAI API KEY] --save_dir result_DG/KG1
❱❱❱ python ChatCRS.py --config configs_DG/CHATCRS_DuE_REC_KNOWLEDGE.yaml  --model CHATGPT --openai_api_key  [YOUR OPENAI API KEY] --save_dir result_DG/KG1

## Running direct Genenration using LLaMA or other Huggingface LLMs
❱❱❱ python ChatCRS.py --config configs_DG/CHATCRS_DuE_CRS_GOAL.yaml  --model LLAMA2 --huggingface_key [YOUR HuggingFace KEY] --save_dir result_DG
❱❱❱ python ChatCRS.py --config configs_DG/CHATCRS_DuE_CRS_KNOWLEDGE.yaml  --model LLAMA2 --huggingface_key [YOUR HuggingFace KEY] --save_dir result_DG/KG1
❱❱❱ python ChatCRS.py --config configs_DG/CHATCRS_DuE_REC_KNOWLEDGE.yaml  --model LLAMA2 --huggingface_key [YOUR HuggingFace KEY] --save_dir result_DG/KG1

```

* --config: Your config file that stores the necessary configuration for running the experiments
* --model: Your LLM for running the empirical analysis
* --openai_api_key: Your own OPENAI API KEY
* --huggingface_key Your own HuggingFace KEY
