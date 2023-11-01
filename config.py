# Copyright: modified from ALCE project from Princeton University

import argparse
import yaml

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to the config file")
    parser.add_argument("--prompt_file", type=str, help="Path to the prompt file")
    parser.add_argument("--eval_file", type=str, help="Path to the eval file")
    parser.add_argument("--quick_test", type=int, default=None, help="Quickly test a few examples")
    parser.add_argument("--save_dir", type=str, default= 'result', help="Path to save the results")


    # Dataset and task setting
    parser.add_argument("--dataset_name", type=str, help="The name of the dataset used, DuRecDial_ENGLISH, DuRecDial_CHINESE, TG-Redial_CHINESE, OpenKnowKG, ReDial")
    parser.add_argument("--task", type=str, help="The task to run, CRS, CHAT, REC, TOPIC, GOAl, KNOWLEDGE")
    parser.add_argument("--with_guidance", type=bool, default=False, help="Whether guidance in the task is used")
    # parser.add_argument("--with_COT", type=bool, default=False, help="Whether COT is used")
    parser.add_argument("--guidance", type=str, default=None, help="The guidance to use, choose from TOPIC, GOAL, REC, KNOWLEDGE")
    # ICL setting
    # parser.add_argument("--ndoc", type=int, help="Number of documents")
    parser.add_argument("--shot", default = 2, type=int, help="Number of ICL demonstrations")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the random number generator")
    # parser.add_argument("--no_doc_in_demo", type=bool, default=False, help="Whether to remove the documents in the demos")
    # parser.add_argument("--fewer_doc_in_demo", type=bool, default=False, help="Whether to use fewer documents in the demos")
    # parser.add_argument("--ndoc_in_demo", type=int, default=None, help="When using --fewer_doc_in_demo, use this to designate how many docs in demo")

    # Model and name
    # parser.add_argument("--dataset_name", type=str, help="Name of the dataset (for saving)")
    parser.add_argument("--tag", type=str, help="Tag of run (for saving)")
    parser.add_argument("--model", type=str, help="Model to use, choose from ChatGPT, GPT4, LLAMA, LLAMA2")
    parser.add_argument("--openai_api", type=bool, default=False, help="Whether to use OpenAI API")
    parser.add_argument("--openai_api_key", type=str, default=None, help="Whether to use OpenAI API")
    parser.add_argument("--huggingface_key", type=str, default=None, help="Whether to use OpenAI API")
    # parser.add_argument("--azure", action="store_true", default=False, help="Azure openai API")

    # Decoding
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature for decoding")
    parser.add_argument("--top_p", type=float, default=1.0, help="Nucleus sampling top-p")
    parser.add_argument("--max_new_tokens", type=int, default=300, help="Max number of new tokens to generate in one step")
    parser.add_argument("--max_length", type=int, default=2048, help="Max length the model can take. Should set properly wrt the model to avoid position overflow.")
    parser.add_argument("--num_samples", type=int, default=1, help="Sample multiple answers.")

    # Use summarization/extraction of the documents
    parser.add_argument("--use_shorter", type=str, default=None, help="Whether to use summary data or extraction data for documents. Option: None, `summary`, `extraction`")

    # Interactive
    parser.add_argument("--interactive", type=bool, default=False, help="Whether to run in interactive mode")
    parser.add_argument("--interactive_query", type=str, default=None, help="The query to use in interactive mode, either `doc_id` (corresponding to interact in paper) or `search` (corresponding to inlinesearch in paper).")
    parser.add_argument("--retriever", type=str, default=None, help="When using interactive search mode, which retriever to use. Options: `tfidf`, `gtr-t5-large`")
    parser.add_argument("--retriever_device", type=str, default="cuda", help="Where to put the dense retriever if using. Options: `cuda`, `cpu`")
    parser.add_argument("--retrieve_in_all_docs", type=bool, default=False, help="Retrieve in all documents instead of just top ndoc")
    parser.add_argument("--max_turn", type=int, default=10, help="Max number of all actions")
    parser.add_argument("--max_doc_show", type=int, default=3, help="Max number of documents to show at one time.")
    parser.add_argument("--force_cite_show", type=bool, default=False, help="Force citing the documents that are shown to the model")


    # Load config
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config)) if args.config is not None else {}
    parser.set_defaults(**config)
    args = parser.parse_args()
    for k in args.__dict__:
        print(f"{k}: {args.__dict__[k]}")

    args = parser.parse_args()
    return args
