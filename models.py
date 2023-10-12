import json
import os 
import openai
import json
import tqdm
import time
from huggingface_hub import login


#First model ChatGPT3.5
def MODEL1_GEN(message2, api_key):
    message1 = "You are a good NLP model."
    openai.api_key = api_key
    message_dict = [{"role": "system", "content": message1}, {"role": "user", "content": message2}]
    chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages = message_dict)
    # print(chat_completion)

    token_length = chat_completion["usage"]["total_tokens"]
    output_messages = chat_completion["choices"][0]["message"]["content"]
    # print(response['choices'][0]['message']['content'])
    return output_messages, token_length

#Second model ChatGPT4
def MODEL2_GEN(message2, api_key):
    message1 = "You are a good NLP model."
    openai.api_key = api_key
    message_dict = [{"role": "system", "content": message1}, {"role": "user", "content": message2}]
    chat_completion = openai.ChatCompletion.create(model="gpt-4", messages = message_dict)
    # print(chat_completion)

    token_length = chat_completion["usage"]["total_tokens"]
    output_messages = chat_completion["choices"][0]["message"]["content"]
    # print(response['choices'][0]['message']['content'])
    return output_messages, token_length

#Third model LLAMA
def MODEL3_GEN(message2, api_key = None):
    '''
    refer the code from LLAMA-alpaca
    import is only done within the function to benefit the user who only use CHATGPT

    '''
    import os
    import sys

    import fire
    # import gradio as gr
    import torch
    import transformers
    from peft import PeftModel
    from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
    from tqdm import tqdm
    # from utils.callbacks import Iteratorize, Stream
    # from utils.prompter import Prompter
    # from data_loader_DST import prepare_data
    import json
    import argparse
    from config import get_args
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(device)
    try:
        if torch.backends.mps.is_available():
            device = "mps"
    except:  # noqa: E722
        pass
    print('Strating main function')
    # args = get_args()
    # args = vars(args)
    base_model = 'huggyllama/llama-7b'
    lora_weights = "tloen/alpaca-lora-7b"
    # prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
        #    torch_dtype=torch.float16,
        )


    # prompt = prompter.generate_prompt(instruction, input)
    inputs = tokenizer(message2, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    token_length = input_ids.shape[1] # number of tokens in the prompt
    temperature=0.1
    top_p=0.75
    top_k=40
    num_beams=4
    max_new_tokens=128
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)

    return output, token_length







#Fourth model LLAMA2
def MODEL4_GEN(message2, api_key = None):
    '''code for LLAMA2'''
    import os
    import sys

    import fire
    # import gradio as gr
    import torch
    import transformers
    from peft import PeftModel
    from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
    from tqdm import tqdm
    # from utils.callbacks import Iteratorize, Stream
    # from utils.prompter import Prompter
    # from data_loader_DST import prepare_data
    import json
    import argparse
    from config import get_args
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(device)
    try:
        if torch.backends.mps.is_available():
            device = "mps"
    except:  # noqa: E722
        pass
    print('Strating main function')
    # args = get_args()
    # args = vars(args)
    base_model = 'meta-llama/Llama-2-7b-hf'
    lora_weights = "meta-llama/Llama-2-7b-hf"
    # prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        # model = PeftModel.from_pretrained(
        #     model,
        #     lora_weights,
        # #    torch_dtype=torch.float16,
        # )


    # prompt = prompter.generate_prompt(instruction, input)
    inputs = tokenizer(message2, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    token_length = input_ids.shape[1] # number of tokens in the prompt
    temperature=0.1
    top_p=0.75
    top_k=40
    num_beams=4
    max_new_tokens=128
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)

    return output, token_length


#Fifth model "YOU OWN FUTURE MODEL"
def MODEL5_GEN(message2, api_key = None):
    '''
    You are supposed to insert any model you want, in a text-to-text format.
    Input: the context from data loader
    Output: the generated repsposne for evaluation
    '''

    return None
    













def generate_inference(model_name, test_loader, api_key, evaluation, data_d, task):
    '''code related to the model'''
    if model_name == "ChatGPT3.5":
        generate = MODEL1_GEN
    elif model_name == "ChatGPT4":
        generate = MODEL2_GEN
    elif model_name == "LLAMA":
        generate = MODEL3_GEN
    elif model_name == "LLAMA2":
        generate = MODEL4_GEN
    elif model_name == "YOU OWN FUTURE MODEL":
        generate = MODEL5_GEN
    else:
        print("Model name not found, we are unable to perform inference")
        


    print("Step 2: RUNNING INFERENCE ON SELECTED MODEL AND DATALOADER")
    print("Model: ", model_name, "Dataset: ", data_d)
    # need to include another value of time taken for the programme
    '''Code related to the data loader'''
    total_token = 0
    counter = 0
    with open(file_name, "r") as f:
        dials = json.load(f)
        for dial_id, dial_dict in tqdm.tqdm(dials.items()):
            counter += 1
            for turn_id, turn_dict in dial_dict.items():
                output_messages, token_length = generate(turn_dict["Prompt"])

                continue
                '''Code related to the generation'''
                return None


def test_model(model_name, test_loader, api_key, evaluation, data_d, task):
    '''code related to the model'''
    if model_name == "ChatGPT3.5":
        generate = MODEL1_GEN
    elif model_name == "ChatGPT4":
        generate = MODEL2_GEN
    elif model_name == "LLAMA":
        generate = MODEL3_GEN
    elif model_name == "LLAMA2":
        generate = MODEL4_GEN
    elif model_name == "YOU OWN FUTURE MODEL":
        generate = MODEL5_GEN
    else:
        print("Model name not found, we are unable to perform inference")


    input_text = "你好，你可以说中文吗？"
    input_text = "Hello, what's your name?"
    output_messages, token_length = generate(input_text, api_key)
    print("Output Message:", output_messages)


if __name__ == "__main__":
    
    login( token = "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")
    model_name = "ChatGPT4"
    model_name = "ChatGPT3.5"
    model_name = "LLAMA"
    model_name = "LLAMA2"
    test_loader = None
    api_key = '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'
    evaluation = None
    data_d = None
    task = None
    test_model(model_name, test_loader, api_key, evaluation, data_d, task)



