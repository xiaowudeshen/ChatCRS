#Copyright, this script is modified from ALCE project in Princeton
# Modified by: Victor Li for ChatCRS project
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
import configparser
import json
from models import LLM_model
from data_loader import prepare_test_data
from config import get_args
import time
import numpy as np
import os
from tqdm import tqdm

def make_demo(item, prompt, test_prompt, instruction=None, guidance = None, test=False):
    # For demo prompt
    

    prompt = prompt.replace("{INST}", instruction).replace("{Q}", item['Input'])
    if guidance is not None:
        prompt = prompt.replace("{G}", item["guide_message"])

    if not test:
        answer = "\n" + "\n".join(item["Output"]) if isinstance(item["Output"], list) else item["Output"]
        prompt = prompt.replace("{A}", "").rstrip() + answer
    else:
        prompt = prompt.replace("{A}", "").rstrip() # remove any space or \n

    return prompt


def prepare_prompt_data(args, prompt_data):
        # Generate the demonstration part
    head_prompt = ""
    train_ids = np.random.choice(len(prompt_data["demos"]), args.shot, replace=False)
    for train_id in train_ids:
        train_item = prompt_data["demos"][train_id]
        if args.with_guidance:
            head_prompt += make_demo(
                train_item, prompt=prompt_data["demo_prompt"], test_prompt=prompt_data["test_prompt"], 
                instruction=prompt_data["Instructions"] + prompt_data["Task"] + prompt_data["Rule"], guidance = prompt_data["guide_message"], test= False
            )
        else:

            head_prompt += make_demo(
                train_item, prompt=prompt_data["demo_prompt"], test_prompt=prompt_data["test_prompt"], 
                instruction=prompt_data["Instructions"] + prompt_data["Task"] + prompt_data["Rule"], guidance = None, test= False
            )
        head_prompt += prompt_data["demo_sep"]
    print(head_prompt)
    return head_prompt


def main():
    #load config and args
    args = get_args()


    
    #random seed
    np.random.seed(args.seed)
    logger.info("Loaded configuration for this experiment")
    logger.info("Loading model...")
    #load model
    # breakpoint()
    CRS = LLM_model(args)

   
    if "GPT4" in args.model:
        args.max_length = 8192
    else:
        args.max_length = 4096
    logger.info("Loaded model. Model_Name: %s, Max token Length: %d", args.model, args.max_length)
    #load data
    prompt_data = json.load(open(args.prompt_file))
    if os.path.exists(args.eval_file):
        eval_data = json.load(open(args.eval_file))
    else:
        eval_data = prepare_test_data(args)

    #create prompt for demo 
    CRS_prompt = prepare_prompt_data(args, prompt_data)

    # Sample quick test
    if args.quick_test is not None:
        eval_ids = np.random.choice(len(eval_data), args.quick_test, replace=False)
        eval_data = [eval_data[int(idx)] for idx in eval_ids]

    logger.info("Generating prompts for some quick tests of the data...") 
    for idx, eval_item in enumerate(tqdm(eval_data)):
        if args.with_guidance:
            eval_data[idx]['prompt'] = CRS_prompt + make_demo(
                eval_item, prompt=prompt_data["demo_prompt"], test_prompt=prompt_data["test_prompt"], 
                instruction=prompt_data["Instructions"] + prompt_data["Task"] + prompt_data["Rule"], guidance = prompt_data["guide_message"], test= False
            )
        else:
            eval_data[idx]['prompt'] = CRS_prompt + make_demo(
                eval_item, prompt=prompt_data["demo_prompt"], test_prompt=prompt_data["test_prompt"], 
                instruction=prompt_data["Instructions"] + prompt_data["Task"] + prompt_data["Rule"], guidance = None,  test= True
            )
    # breakpoint()
    logger.info("Ready to run the experiment, total number of testing smaples:%d", len(eval_data))

    #run evaluation
    for idx, item in enumerate(tqdm(eval_data)):
        prompt = item['prompt']
        print(prompt)
        prompt_len = len(CRS.tokenizer.tokenize(prompt))

        if idx == 0:
            print(prompt)

        output_array = []
        for _ in range(args.num_samples):
            if args.interactive:
                print("============ Interactive =============")
                output_answer = ""
                # doc_list = item['docs']

                interactive_prompt = prompt.rstrip() + "\n" # Start a new line
                # inline_doc = ""
                num_turn = 0
                further_instruction = ""
                
                # doc_history = []
                while True:
                    
                    num_turn += 1
                    new_prompt = interactive_prompt + further_instruction
                    new_prompt_len = len(CRS.tokenizer.tokenize(new_prompt))

                    if idx == 0:
                        print(f"-------------- Step {num_turn} prompt --------------")
                        print(new_prompt)
                        print("-----------------------------")

                    output = CRS.generate(new_prompt, min(args.max_new_tokens, args.max_length-new_prompt_len), stop=["\n", "\n\n"])

                    print("Output: ", output)
                    # if len(inline_doc) > 0:
                        # output = "Output: " + output # "Output: " was included in inline_doc
                    # inline_doc = "" # Delete inline_doc after use
                    # interactive_prompt += output + "\n"
                    

                    print("============ Interactive =============")
                    logger.info(f"Model output: \"{output}\"")
                    print("Do you want to exit or provide more guidance to the model?")
                    key_in = Input("Type 'exit' to exit or type 'more' to provide more guidance with [REC, TOPIC, GOAL, KNOWLEDGE] to the model: ")
                    
                    if key_in == "exit":
                        break
                    fix_guidance = False
                    while True:
                        for gui in ["REC", "TOPIC", "GOAL", "KNOWLEDGE"]:
                            if gui in key_in:
                                fix_guidance = True
                                further_instruction = key_in
                                break
                        if fix_guidance:
                            break
                        else:
                            print("Thx for your answer, would you like to provide guidance in the standard format or using free text?")
                            key_in = Input("Re-enter your guidance in the standard format or type 'free: Your guidance' to use free text: ")
                            if "free:" in key_in:
                                further_instruction = key_in.replace("free:", "")
                                break

                    
                    if num_turn >= args.max_turn:
                        logger.warning("Reach maximum number of turns. Terminate now.")
                        break
                
                # if "qampari" in args.eval_file:
                #     output_answer = output_answer.rstrip().rstrip(",")
                # output_array.append(output_answer)
                # item['prompt'] = interactive_prompt
                # item['doc_history'] = doc_history
            else: 
                # breakpoint()
                output_array.append(CRS.generate(prompt, min(args.max_new_tokens, args.max_length-prompt_len)))
                item['prompt'] = prompt
            
            # output_array[-1] = output_array[-1].replace("<|im_end|>", "").rstrip()
            # if output_array[-1].endswith("End."):
            #     output_array[-1] = output_array[-1][:-len("End.")]

            logger.info(f"Prompt length={prompt_len}")
            logger.info(f"Question: {item['Input']}")
            logger.info(f"Gold answer: {item['Output']}")
            logger.info(f"Final model output: {output_array[-1]}") 
        
        item['output'] = output_array if len(output_array) > 1 else output_array[0]
        # breakpoint()
    logger.info(f"#Cases when prompts exceed max length: {CRS.prompt_exceed_max_length}")
    logger.info(f"#Cases when max new tokens < 50: {CRS.fewer_than_50}")

    #Save results
    
    model_name = args.model
    if "/" in model_name:
        model_name = model_name.split("/")[-1]
    name = f"{args.dataset_name}-{model_name}-{args.task}-gui{args.guidance}-shot{args.shot}-{args.seed}"
    
    if args.quick_test is not None:
        name += f"-quick_test{args.quick_test}"
    
    if args.num_samples > 1:
        name += f"-sample{args.num_samples}"
    if args.force_cite_show:
        name += f"-forceciteshow"

    
    
       #Calculate price for API usage
    if args.openai_api:
        logger.info(f"Token used: prompt {CRS.prompt_tokens}; completion {CRS.completion_tokens}")
        if "CHATGPT" in args.model:
            p_price, c_price = 0.0015, 0.002
        if "GPT4" in args.model:
            p_price, c_price = 0.03, 0.06   
        else:
            logger.warn("Cannot find model price")
            p_price, c_price = 0, 0

        eval_data["total_cost"] = CRS.prompt_tokens / 1000 * p_price + CRS.completion_tokens / 1000 * c_price        

        logger.info(f"Unit price (Oct 16, 2023, prompt/completion): {p_price}/{c_price}")
        logger.info(f"Total cost: %.1f" % (eval_data["total_cost"]))
    
    
    eval_data = {
        "args": args.__dict__,
        "data": eval_data,
    }
    # save all to the results folder
    if not os.path.exists("result"):
        os.makedirs("result")
    json.dump(eval_data, open("result/" + name + ".json", "w"), indent=4)


 
    

    

if __name__ == "__main__":
    # args = get_args()
    # args.prompt_file = "instructions/template_response.json"
    # prompt_data = json.load(open(args.prompt_file))
    # prepare_prompt_data(args, prompt_data)
    main()

''' 
Overall logic for ChatCRS Evaluation Process:

- Get the config file for the model, task, data, and evaluation (ChatCRS.py; Configs; config.py)
    - check the format for any error case
    - save the config as variables

- Load the data (data_loader.py)
    - for each dataset, load the tasks (if task is contained in the dataset)
    - for each task, design sepecific instructions for the task
    - prepare necessary supporting task data and save at default location for evaluation

- Load the model (models.py)
    - for each model, load the model
    - run model on the data and save for evaluation
     
- Load the evaluation (evaluation.py)
    - for each save results (model, dataset, tasks), load the evaluation
    - for each task, run the evaluation
        - Metric-based evaluation
        - Plot the results and graphs
    - for response generation task (additional task):
        - evaluation on each supporting task
        - prepare scripts for human evaluation
'''
