import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import torch
import json
import os 
import openai
import json
import tqdm
import time
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_max_memory():
    """Get the maximum memory available for the current GPU for loading models."""
    free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
    max_memory = f'{free_in_GB-6}GB'
    n_gpus = torch.cuda.device_count()
    max_memory = {i: max_memory for i in range(n_gpus)}
    return max_memory


def load_model(model_name_or_path, dtype=torch.float16, int8=False, reserve_memory=10):
    # Load a huggingface model and tokenizer
    # dtype: torch.float16 or torch.bfloat16
    # int8: whether to use int8 quantization
    # reserve_memory: how much memory to reserve for the model on each gpu (in GB)

    # Llama: set up the root dir
    # open_source_models = ["llama", "alpaca"]
    # if any([m in model_name_or_path for m in open_source_models]):
    #     model_name_or_path = os.path.join(os.environ["LLAMA_ROOT"], model_name_or_path)

    # Load the FP16 model
    
    logger.info(f"Loading {model_name_or_path} in {dtype}...")
    if int8:
        logger.warn("Use LLM.int8")
    start_time = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map='auto',
        torch_dtype=dtype,
        max_memory=get_max_memory(),
        load_in_8bit=int8,
    )
    logger.info("Finish loading in %.2f sec." % (time.time() - start_time))

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)

    # Fix OPT bos token problem in HF
    if "opt" in model_name_or_path:
        tokenizer.bos_token = "<s>"
    tokenizer.padding_side = "left"

    return model, tokenizer




class LLM_model:

    def __init__(self, args):
        self.args = args

        if args.model not in ["CHATGPT", "GPT4", "LLAMA", "LLAMA2"]:
            raise Exception("Your model is either not in the supported model list or you need to implement your own model for evaluation")
        # breakpoint()
        if args.model in ['CHATGPT', 'GPT4']:
            args.openai_api = True
        
            import openai
            if args.openai_api_key is None:
                raise Exception("You need to provide an OpenAI API key")
            OPENAI_API_KEY = args.openai_api_key
            openai.api_key = OPENAI_API_KEY

            self.tokenizer = AutoTokenizer.from_pretrained("gpt2", fast_tokenizer=False)
            self.total_tokens = 0
        else:
            if args.model == "LLAMA":
                model_name_or_path = "huggyllama/llama-13b"
            elif args.model == "ALPACA":
                model_name_or_path = "tloen/alpaca-lora-7b"
            elif args.model == "LLAMA2":
                from huggingface_hub import login
                login(token = args.huggingface_key)
                model_name_or_path = "meta-llama/Llama-2-7b-hf"
            else:
                raise Exception("Your model is either not in the supported model list or you need to implement your own model for evaluation")
                '''
                Place reserved for implementin your own models
                Need to specify the path that save your own model weights, model config and model tokenizer
                Need to match the AutoModelForCausalLM and AutoTokenizer of the current transformers version
                '''

            self.model, self.tokenizer = load_model(model_name_or_path, dtype=torch.float16, int8=False)

        self.prompt_exceed_max_length = 0
        self.fewer_than_50 = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    
    def generate(self, prompt, max_tokens, stop=None):
        args = self.args
        if max_tokens <=50:
            self.prompt_exceed_max_length += 1
            logger.warning("Prompt exceeds max length and return an empty string as answer. If this happens too many times, it is suggested to make the prompt shorter")
            return ""
        if max_tokens < 50:
            self.fewer_than_50 += 1
            logger.warning("The model can at most generate < 50 tokens. If this happens too many times, it is suggested to make the prompt shorter")

        if args.openai_api:
                # For OpenAI's ChatGPT API, we need to convert text prompt to chat prompt
            prompt = [
                {'role': 'system', 'content': "You are an excellent conversational recommender that helps the user achieve recommendation-related goals through conversations."},
                {'role': 'user', 'content': prompt}
            ]
            if args.model == "CHATGPT":
                # deploy_name = "gpt-3.5-turbo"
                deploy_name = "gpt-3.5-turbo-1106"
            elif args.model == "GPT4":
                deploy_name = "gpt-4"
            
            is_ok = False
            retry_count = 0
            while not is_ok:
                retry_count += 1
                try:
                    response = openai.ChatCompletion.create(
                        model=deploy_name,
                        messages=prompt,
                        temperature=args.temperature,
                        max_tokens=max_tokens,
                        stop=stop,
                        top_p=args.top_p,
                    )
                    is_ok = True
                except Exception as error:
                    if retry_count <= 5:
                        logger.warning(f"OpenAI API retry for {retry_count} times ({error})")
                        continue
                    print(error)
            self.prompt_tokens += response['usage']['prompt_tokens']
            self.completion_tokens += response['usage']['completion_tokens']   
            logger.info(f"Prompt tokens: {response['usage']['prompt_tokens']}, Completion tokens: {response['usage']['completion_tokens']}")    
            # self.total_tokens += response['usage']['total_tokens']
            return response['choices'][0]['message']['content']
            
        else:
            inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
            stop = [] if stop is None else stop
            stop = list(set(stop + ["\n", "Ċ", "ĊĊ", "<0x0A>"])) # In Llama \n is <0x0A>; In OPT \n is Ċ
            stop_token_ids = list(set([self.tokenizer._convert_token_to_id(stop_token) for stop_token in stop] + [self.model.config.eos_token_id]))
            if "LLAMA" in args.model:
                stop_token_ids.remove(self.tokenizer.unk_token_id)
            outputs = self.model.generate(
                **inputs,
                do_sample=True, temperature=args.temperature, top_p=args.top_p, 
                max_new_tokens=max_tokens,
                num_return_sequences=1,
                eos_token_id=stop_token_ids
            )
            generation = self.tokenizer.decode(outputs[0][inputs['input_ids'].size(1):], skip_special_tokens=True)
            return generation


        

# def generate_inference(model_name, test_loader, api_key, evaluation, data_d, task):
#     '''code related to the model'''
#     if model_name == "ChatGPT3.5":
#         generate = MODEL1_GEN
#     elif model_name == "ChatGPT4":
#         generate = MODEL2_GEN
#     elif model_name == "LLAMA":
#         generate = MODEL3_GEN
#     elif model_name == "LLAMA2":
#         generate = MODEL4_GEN
#     elif model_name == "YOU OWN FUTURE MODEL":
#         generate = MODEL5_GEN
#     else:
#         print("Model name not found, we are unable to perform inference")
        


#     print("Step 2: RUNNING INFERENCE ON SELECTED MODEL AND DATALOADER")
#     print("Model: ", model_name, "Dataset: ", data_d)
#     # need to include another value of time taken for the programme
#     '''Code related to the data loader'''
#     total_token = 0
#     counter = 0
#     with open(file_name, "r") as f:
#         dials = json.load(f)
#         for dial_id, dial_dict in tqdm.tqdm(dials.items()):
#             counter += 1
#             for turn_id, turn_dict in dial_dict.items():
#                 output_messages, token_length = generate(turn_dict["Prompt"])

#                 continue
#                 '''Code related to the generation'''
#                 return None




# if __name__ == "__main__":
    
#     login( token = "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")
#     model_name = "ChatGPT4"
#     model_name = "ChatGPT3.5"
#     model_name = "LLAMA"
#     model_name = "LLAMA2"
#     test_loader = None
#     api_key = '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'
#     evaluation = None
#     data_d = None
#     task = None
#     test_model(model_name, test_loader, api_key, evaluation, data_d, task)



