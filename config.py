# Copyright (c) Facebook, Inc. and its affiliates

import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="t5-small", help="Path, url or short name of the model")
    parser.add_argument("--data_dir", type=str, default="../durecdial", help="Dataset Directory")
    parser.add_argument('--data_name', default='durecdial', type=str, help="dataset name")
    parser.add_argument("--model_name_or_path", default='fnlp/bart-base-chinese', type=str, help="model name or path")
    parser.add_argument("--output_dir", default='output', type=str,  help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--data_dir", default='../data', type=str, help="The data directory.")
    parser.add_argument("--cache_dir", default='/projdata1/info_fil/ydeng/bert', type=str, help="The cache directory.")
    
    parser.add_argument("--max_seq_length", default=512, type=int,help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=100, type=int, help="The maximum total output sequence length.")
    
    # parser.add_argument("--saving_dir", type=str, default="save", help="Path for saving")
    # parser.add_argument("--train_batch_size", type=int, default=16, help="Batch size for training")
    # parser.add_argument("--meta_batch_size", type=int, default=1, help="Batch size for meta training")
    # parser.add_argument("--dev_batch_size", type=int, default=8, help="Batch size for validation")
    # parser.add_argument("--test_batch_size", type=int, default=8, help="Batch size for test")
    # parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Accumulate gradients on several steps")
    # parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    # parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    # parser.add_argument("--n_epochs", type=int, default=5, help="Number of training epochs")
    # parser.add_argument("--seed", type=int, default=577, help="Random seed")
    # parser.add_argument("--verbose", action='store_true', help="continual baseline")
    # parser.add_argument("--length", type=int, default=50, help="Batch size for validation")
    # parser.add_argument("--max_history", type=int, default=2, help="max number of turns in the dialogue")
    # parser.add_argument("--GPU", type=int, default=8, help="number of gpu to use")
    # parser.add_argument("--model_name", type=str, default="t5", help="use t5 or bart?")
    # parser.add_argument("--slot_lang", type=str, default="none", help="use 'none', 'human', 'naive', 'value', 'question', 'slottype' or 'binary' slot description")
    # parser.add_argument("--fewshot", type=float, default=0.0, help="data ratio for few shot experiment")
    # parser.add_argument("--fix_label", action='store_true')
    # parser.add_argument("--except_domain", type=str, default="none", help="hotel, train, restaurant, attraction, taxi")
    # parser.add_argument("--only_domain", type=str, default="none", help="hotel, train, restaurant, attraction, taxi")
    # parser.add_argument("--threshold", type=float, default=0.4)
    # parser.add_argument("--semi", action='store_true')
    # parser.add_argument("--mode", type=str, default="train")
    # parser.add_argument("--turn_att", type=str, default="euclidean")
    # parser.add_argument("--max_dist", type=int, default=100)
    # parser.add_argument("--joint_training", type=str, default="none", help="slot_type_discriminator_summary")
    # parser.add_argument("--self_training", type=str, default="R1", help="round of self-training, user R1 or R2 or R3")
    

    args = parser.parse_args()
    return args
