# ChatCRS
# ChatCRS: Incorporating External Knowledge and Goal Guidance for Conversational Recommender Systems

## Abstract:
This paper aims to efficiently enable Large Language Models (LLMs) to use external knowledge and goal guidance in conversational recommender system (CRS) tasks. Advanced LLMs (e.g., ChatGPT) are limited in CRS tasks for 1) generating grounded responses with recommendation-oriented knowledge,
or 2) proactively guiding users through different dialogue goals. In this work, we first analyze those limitations through a comprehensive evaluation to assess LLMs' intrinsic capabilities, showing the necessity of incorporating external knowledge and goal guidance which contribute significantly to the recommendation accuracy and language quality. In light of this finding, we propose a novel ChatCRS framework to decompose the complex CRS task into several sub-tasks through the implementation of 1) a knowledge retrieval agent using a tool-augmented approach to reason over external Knowledge Bases and 2) a goal-planning agent for dialogue goal prediction. Experimental results on two CRS datasets reveal that ChatCRS sets new state-of-the-art benchmarks, improving language quality of informativeness by 17% and proactivity by 27%, and achieving a tenfold enhancement in recommendation accuracy over LLM-based CRS

## Method:
<p align="center">
<img src="figs/figure.png" width="100%" />

</p>

Overview of **UNO-DST** which consists of two periods: 1) **joint training** for both task A (slot value prediction) and B (slot type prediction), and 2) **self-training** in the unknown target domain. **Step 1**: Generation of slot values and types from tasks A and B; **Step 2**: Selection of good samples using cycle consistency between two tasks; **Step 3**: Fine-turning the PLM with selected samples.

<p align="center">
<img src="figs/figure1.png" width="100%" />

</p>

Generalise **UNO-DST** to Large Language Models and ChatGPT. We utilise both In-Context Learning and Conversational approaches. Check the results in our paper.

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
❱❱❱ python ChatCRS.py --config_file [Your configuration file]
```
* --config_file: Your config file that stores the necessary configuration for running the experiments
