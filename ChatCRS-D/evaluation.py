#date Oct, 2023
#author: Victor Li
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
import re
import os
import json
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from collections import Counter
import torch
import torch.nn.functional as F
from sklearn import metrics
import jieba
import string
import re
from config import get_args
import tqdm

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(preds, refs):
    f1s = []
    for pred_items, gold_items in zip(preds, refs):
        common = Counter(gold_items) & Counter(pred_items)
        num_same = sum(common.values())
        if num_same == 0:
            f1 = 0
        else:
            precision = 1.0 * num_same / len(pred_items)
            recall = 1.0 * num_same / len(gold_items)
            f1 = (2 * precision * recall) / (precision + recall)
        f1s.append(f1)
    return sum(f1s)/len(f1s)

def distinct(seqs):

    intra_dist1, intra_dist2 = [], []
    unigrams_all, bigrams_all = Counter(), Counter()
    for seq in seqs:
        unigrams = Counter(seq)
        bigrams = Counter(zip(seq, seq[1:]))
        intra_dist1.append((len(unigrams)+1e-12) / (len(seq)+1e-5))
        intra_dist2.append((len(bigrams)+1e-12) / (max(0, len(seq)-1)+1e-5))

        unigrams_all.update(unigrams)
        bigrams_all.update(bigrams)

    inter_dist1 = (len(unigrams_all)+1e-12) / (sum(unigrams_all.values())+1e-5)
    inter_dist2 = (len(bigrams_all)+1e-12) / (sum(bigrams_all.values())+1e-5)
    intra_dist1 = np.average(intra_dist1)
    intra_dist2 = np.average(intra_dist2)
    return intra_dist1, intra_dist2, inter_dist1, inter_dist2

def perplexity(logits, targets, weight=None, padding_idx=None, device=None):
    """
    logits: (batch_size, max_len, vocab_size)
    targets: (batch_size, max_len)
    """
    batch_size = logits.size(0)
    if weight is None and padding_idx is not None:
        weight = torch.ones(logits.size(-1), device=device)
        weight[padding_idx] = 0
    nll = F.nll_loss(input=logits.view(-1, logits.size(-1)),
                     target=targets.contiguous().view(-1),
                     weight=weight,
                     reduction='none')
    nll = nll.view(batch_size, -1).sum(dim=1)
    if padding_idx is not None:
        word_cnt = targets.ne(padding_idx).float().sum()
        nll = nll / word_cnt
    ppl = nll.exp()
    return ppl

def topic_f1_score(pred_pt, gold_pt):
    ps = []
    rs = []
    f1s = []
    for pred_labels, gold_labels in zip(pred_pt, gold_pt):
        if len(pred_labels) == 0:
            pred_labels.append('empty')
        if len(gold_labels) == 0:
            gold_labels.append('empty')
        tp = 0
        for t in pred_labels:
            if t in gold_labels:
                tp += 1
        r = tp / len(gold_labels)
        p = tp / len(pred_labels)
        try:
            f1 = 2 * p * r / (p + r)
        except ZeroDivisionError:
            f1 = 0
        ps.append(p)
        rs.append(r)
        f1s.append(f1)
    p = sum(ps) / len(ps)
    r = sum(rs) / len(rs)
    f1 = sum(f1s) / len(f1s)
    scores = [p, r, f1]

    return scores


def topic_hit_score(pred_pt, gold_pt):
    hits1 = []
    hits3 = []
    hits5 = []
    for pred_labels, gold_labels in zip(pred_pt, gold_pt):
        if len(gold_labels) == 0:
            continue
        if len(set(pred_labels[:1])&set(gold_labels)) > 0:
            hits1.append(1)
        else:
            hits1.append(0)
        if len(set(pred_labels[:3])&set(gold_labels)) > 0:
            hits3.append(1)
        else:
            hits3.append(0)
        if len(set(pred_labels[:5])&set(gold_labels)) > 0:
            hits5.append(1)
        else:
            hits5.append(0)
    hits1 = sum(hits1)/len(hits1)
    hits3 = sum(hits3)/len(hits3)
    hits5 = sum(hits5)/len(hits5)
    return [hits1, hits3, hits5]

def goal_f1_score(pred_pt, gold_pt, dataset):
    
    # breakpoint()
    def get_metrics(y, y_pre, dataset):
        if dataset == 'DuRecDial_ENGLISH':
            macro_f1 = metrics.f1_score(y, y_pre, average='macro')
            macro_precision = metrics.precision_score(y, y_pre, average='macro')
            macro_recall = metrics.recall_score(y, y_pre, average='macro')
            return macro_precision, macro_recall, macro_f1
    
    
    all_scores = list(get_metrics(gold_pt, pred_pt, dataset))

    return all_scores

def ndcg_score(preds, refs):
    ndcg10 = []
    ndcg50 = []
    for pred, ref in zip(preds, refs):
        #if 0 in ref:
        #    continue
        score10 = 0.0
        score50 = 0.0
        for rank, item in enumerate(pred):
            if item in ref:
                if rank < 10:
                    score10 += 1.0/np.log2(rank+2)
                if rank < 50:
                    score50 += 1.0/np.log2(rank+2)
        
        norm = 0.0
        for rank in range(len(ref)):
            norm += 1.0/np.log2(rank+2)
        ndcg10.append(score10/max(0.3,norm))
        ndcg50.append(score50/max(0.3,norm))
    ndcg10 = sum(ndcg10)/len(ndcg10)
    ndcg50 = sum(ndcg50)/len(ndcg50)
    return [ndcg10, ndcg50]

def mrr_score(preds, refs):
    mrr10 = []
    mrr50 = []
    for pred, ref in zip(preds, refs):
        #if 0 in ref:
        #    continue
        score10 = 0.0
        score50 = 0.0
        for rank, item in enumerate(pred):
            if item in ref:
                if rank < 10:
                    score10 = 1.0/ (rank+1.0)
                    score50 = 1.0/ (rank+1.0)
                    break
                if rank < 50:
                    score50 = 1.0/ (rank+1.0)
                    break
        mrr10.append(score10)
        mrr50.append(score50)
    mrr10 = sum(mrr10)/len(mrr10)
    mrr50 = sum(mrr50)/len(mrr50)
    return [mrr10, mrr50]

def bleu_cal(sen1, tar1):
    bleu1 = sentence_bleu([tar1], sen1, weights=(1, 0, 0, 0))
    bleu2 = sentence_bleu([tar1], sen1, weights=(0, 1, 0, 0))
    bleu = sentence_bleu([tar1], sen1)
    return bleu1, bleu2, bleu


def tgredial_bleu(tokenized_gen, tokenized_tar):
    bleu1_sum, bleu2_sum, bleu_sum, count = 0, 0, 0, 0
    for sen, tar in zip(tokenized_gen, tokenized_tar):
        bleu1, bleu2, bleu = bleu_cal(sen, tar)
        bleu1_sum += bleu1
        bleu2_sum += bleu2
        bleu_sum += bleu
        count += 1

    return bleu_sum / count, bleu1_sum / count, bleu2_sum / count


def is_json(myjson):
  try:
    json.loads(myjson)
  except ValueError as e:
    return False
  return True


def automatic_evaluation(args, path_to_data, json_format=False, special_format=False, special_task = None):
    task_dic = {
        "DuRecDial_ENGLISH": ["CRS", "CHAT", "REC", "TOPIC", "GOAL", "KNOWLEDGE"],
        "DuRecDial_CHINESE": ["CRS", "CHAT", "REC", "TOPIC", "GOAL"],
        "TG-Redial_CHINESE": ["CRS", "CHAT", "REC", "TOPIC", "GOAL"]
    }
    if special_task is not None:
        args.task = special_task
        args.dataset_name = "DuRecDial_ENGLISH"
        print(f"Running evaluatin for special task: {args.task}")
    else:
        for key in task_dic:
            
            if key in path_to_data:
                args.dataset_name = key
                break
        for task in task_dic[args.dataset_name]:
            if f"-{task}-" in path_to_data:
                args.task = task
                break   
        logger.info(f"#Running evaluatin for dataset: {args.dataset_name} and task: {args.task}")

    with open(path_to_data, 'r', encoding='utf-8') as infile:
        data = json.load(infile)
    # if data["args"]["guidance"] is not None:
    #     logger.info(f"#Running evaluatin for guidance: {data['args']['guidance']}")
        
    A_golds = []
    A_preds = []
    
    '''
    For different dataset and task, use different metrics for evaluation
    '''
    if args.dataset_name == 'DuRecDial_ENGLISH':
        if args.task in ['CRS', 'CHAT']:
            #running result for F1, blue, distinct
            #run bleu

            ####first make sure that the output is in json format and load correct data for evaluation
            format_error = 0
            for d in data["data"]:
                # breakpoint()
                if type(d["Output"]) is list:
                    A_golds.append([normalize_answer(item) for item in d["Output"]])
                else:
                    A_golds.append(normalize_answer(d["Output"]))
                if type(d["output"]) is list:
                    A_preds.append([normalize_answer(item) for item in d["output"]])
                else:
                    
                    if json_format:
                        if is_json(d["output"]) and "System Response" in json.loads(d["output"]):
                            dj = json.loads(d["output"])
                            # if "system Resonse" not in dj:
                            #     print(dj)
                            #     breakpoint()
                            A_preds.append(dj["System Response"])  # process for json format
                        else: 
                            format_error += 1
                            A_golds.pop()
                    elif special_format:
                        if len(d["output"].split("system response is")) == 2:
                            output = d["output"].split("system response is")[1].replace("]", "").replace("[", "").replace("\"", "").strip()
                            A_preds.append(normalize_answer(output))  # process for natural langauage
                        else:
                            # breakpoint()
                            format_error += 1
                            A_golds.pop()
                    else:
                        A_preds.append(normalize_answer(d["output"]))  # process for natural langauage
                            

            print('format error: ', format_error)
            print(len(A_golds), len(A_preds))

            bleu_preds = A_preds
            bleu_refs = [[gold] for gold in A_golds]
            bleu_score = corpus_bleu(bleu_refs, bleu_preds)
            bleu1 = corpus_bleu(bleu_refs, bleu_preds, weights=(1, 0, 0, 0))
            bleu2 = corpus_bleu(bleu_refs, bleu_preds, weights=(0.5, 0.5, 0, 0))
            
            
            
            bleu_scores = [bleu_score, bleu1, bleu2]
            print('Running BLEU for ' + ' ' + args.dataset_name + '-----------------------------')
            print('BLEU: ', bleu_scores)
            
            dist_scores = list(distinct(A_preds))
            print('Running Dist for ' + ' ' + args.dataset_name + '-----------------------------')
            print('Dist: ', dist_scores)

            f1_scores = [f1_score(A_preds, A_golds)]
            print('Running F1 for ' + ' ' + args.dataset_name + '-----------------------------')
            print('F1: ', f1_scores)

            auto_scores = bleu_scores + dist_scores + f1_scores 
        elif args.task == 'TOPIC':
            refs = A_golds
            if A_preds is not list:
                preds = [pred for pred in A_preds] #if pred is one item
            else:
                preds = A_preds # if prd is a list
            hit_scores = topic_hit_score(preds, refs)
            f1_scores = topic_f1_score(preds, refs)
            print('Running P/R/F1 for ' + ' ' + args.dataset_name + '-----------------------------')
            print('P/R/F1/hits: ', f1_scores, hit_scores)
            auto_scores = f1_scores + hit_scores


        elif args.task == 'GOAL':
            format_error = 0
            file_name = 'data_dict/DuRecDial_ENGLISH_CRS_GOAL.json'
            hit = 0
            
            with open(file_name, 'r', encoding='utf-8') as infile:
                goal_dict = json.load(infile)
            for d in data["data"]:
            # for d in data:
                goal_truth = goal_dict[d["dial-turn_id"]]["guide_message"]
                # goal_truth = d["gold_label"]
                # breakpoint()
                if type(goal_truth) is list:
                    A_golds.append(goal_truth[0].replace('\"', ''))
                else:
                    A_golds.append(goal_truth.replace('\"', '').replace('[', '').replace(']', '').replace('\'', '').strip() )
                if type(d["output"]) is list:
                    A_preds.append(normalize_answer(d["Output"][0]))
                else:
                    if json_format:
                        if is_json(d["output"]):
                            a_preds = []
                            dj = json.loads(d["output"])
                            # if "system Resonse" not in dj:
                            #     print(dj)
                            #     breakpoint()
                            for key, item in dj.items():
                                a_preds.append(item)
                            A_preds.append(a_preds)  # process for json format
                            # breakpoint()
                        else:
                            format_error += 1
                            A_golds.pop()
                    elif special_format:
                        sep_token1 = "and the system response is"
                        sep_token2 = "and the recommendation list is"
                        task_token = "dialogue goal is"
                        if len(d["output"].split(sep_token1)) == 2:
                            # breakpoint()
                            output = d["output"].split(sep_token1)[0]
                            output1 = output.split(task_token)[1]
                            output_goal = output1.replace('\"', '').replace('[', '').replace(']', '').replace('\'', '').strip() 
                            A_preds.append(output_goal)  # process for natural langauage
                            
                        elif len(d["output"].split(sep_token2)) == 2:
                            output = d["output"].split(sep_token2)[0]
                            output1 = output.split(task_token)[1]
                            output_goal = output1.replace('\"', '').replace('[', '').replace(']', '').replace('\'', '').strip() 
                            A_preds.append(output_goal)  # process for natural langauage
                        else:
                            format_error += 1
                            A_golds.pop()
                    else:
                        # A_preds.append(d["pred_label"].replace("]</s>", "").replace('[', '').replace('\"', '').replace('\'', '').strip())  
                        A_preds.append(d["output"].strip().lower())  # process for natural langauage
                
                # print(goal_dict[d["dial-turn_id"]])
                # print(A_golds) 
                # print(A_preds)   
                # breakpoint()
            print('format error: ', format_error)
            print(len(A_golds), len(A_preds))
            
            refs = A_golds
            preds = A_preds
            # print(refs)
            # print(preds)
            accuracy = 0
            for i in range(len(refs)):
                if refs[i] == preds[i]:
                    accuracy += 1
            print("Accuracy: ", accuracy/len(refs))
            # breakpoint()
            f1_scores = goal_f1_score(preds, refs, args.dataset_name)
            print('format error: ', format_error)
            print(len(A_golds), len(A_preds))
            print('Running P/R/F1/Accuracy for ' + ' ' + args.dataset_name + '-----------------------------')
            print('P/R/F1: ', f1_scores, accuracy/len(refs))
            auto_scores = f1_scores


        elif args.task == 'RELATION':
            format_error = 0
            # breakpoint()
            for d in data["data"]:
                goal_truth = d["Output"]
                # breakpoint()
                if type(goal_truth) is list and len(goal_truth) == 3:
                    R = goal_truth[1]
                    if  '-' in R and '2018' in R:
                        A_golds.append("weather")
                    else:
                        A_golds.append(goal_truth[1])
                else:
                    format_error += 1
                    continue
                if type(d["output"]) is list:
                    A_preds.append(normalize_answer(d["Output"][0]))
                else:
                    if json_format:
                        if is_json(d["output"]):
                            a_preds = []
                            dj = json.loads(d["output"])
                            # if "system Resonse" not in dj:
                            #     print(dj)
                            #     breakpoint()
                            for key, item in dj.items():
                                a_preds.append(item)
                            A_preds.append(a_preds)  # process for json format
                            # breakpoint()
                        else:
                            format_error += 1
                            A_golds.pop()
                    elif special_format:
                        task_token = "relation is"
                        if len(d["output"].split(task_token)) == 2:
                            # breakpoint()
                            output1 = d["output"].split(task_token)[1]
                            output_goal = output1.replace('\"', '').replace('[', '').replace(']', '').replace('\'', '').replace('.', '').strip() 
                            A_preds.append(output_goal)  # process for natural langauage
                            
                        
                        else:
                            format_error += 1
                            A_golds.pop()
                    else:
                        A_preds.append([d["output"].strip().lower()])  # process for natural langauage
                
                # print(goal_dict[d["dial-turn_id"]])
                # print(A_golds) 
                # print(A_preds)   
                # breakpoint()
            print('format error: ', format_error)
            print(len(A_golds), len(A_preds))
            
            refs = A_golds
            preds = A_preds
            print(refs)
            print(preds)
            accuracy = 0
            for i in range(len(refs)):
                if refs[i] == preds[i]:
                    accuracy += 1
            print("Accuracy: ", accuracy/len(refs))
            # breakpoint()
            f1_scores = goal_f1_score(preds, refs, args.dataset_name)
            # f1_scores = [f1_score(A_preds, A_golds)]
            print('format error: ', format_error)
            print(len(A_golds), len(A_preds))
            print('Running P/R/F1/Accuracy for ' + ' ' + args.dataset_name + '-----------------------------')
            print('P/R/F1: ', f1_scores, accuracy/len(refs))
            auto_scores = f1_scores

        elif args.task == 'KNOWLEDGE':
            #take knowledge prediction as generation task same as topic prediction
            format_error = 0
            # breakpoint()
            file_name = 'data_dict/DuRecDial_ENGLISH_CRS_KNOWLEDGE.json'
            hit = 0
            
            with open(file_name, 'r', encoding='utf-8') as infile:
                know_dict = json.load(infile)
            for d in data["data"]:
                know_truth = know_dict[d["dial-turn_id"]]["guide_message"].replace('[', '').replace(']', '').split(',')
                if type(know_truth) is list and len(know_truth) == 3:
                    R = know_truth[1]
                    if  '-' in R and '2018' in R:
                        A_golds.append("weather")
                    else:
                        A_golds.append(know_truth[1].strip().replace('\'', '').replace('\"', ''))
                else:
                    format_error += 1
                    continue
                if type(d["output"]) is list:
                    A_preds.append(normalize_answer(d["Output"][0]))
                else:
                    if json_format:
                        if is_json(d["output"]):
                            a_preds = []
                            dj = json.loads(d["output"])
                            # if "system Resonse" not in dj:
                            #     print(dj)
                            #     breakpoint()
                            for key, item in dj.items():
                                a_preds.append(item)
                            A_preds.append(a_preds)  # process for json format
                            # breakpoint()
                        else:
                            format_error += 1
                            A_golds.pop()
                    elif special_format:
                        sep_token1 = "and the system response is"
                        sep_token2 = "and the recommendation list is"
                        task_token = "knowledge triples is"
                        if len(d["output"].split(sep_token1)) == 2:
                            output = d["output"].split(sep_token1)[0]
                            if len(output.split(task_token)) != 2:
                                format_error += 1
                                A_golds.pop()
                                continue
                            output1 = output.split(task_token)[1]
                            output_goal = output1.replace('\"', '').replace('[', '').replace(']', '').replace('\'', '').replace('.', '').strip() 
                            # breakpoint()
                            if len(output_goal.split(','))>=3:
                                R_pred = output_goal.split(',')[1].replace('\'', '').replace('\"', '').strip()
                                A_preds.append(R_pred)  # process for natural langauage
                            else:
                                A_preds.append(output_goal)  # process for natural langauage   

                        elif len(d["output"].split(sep_token2)) == 2:
                            output = d["output"].split(sep_token2)[0]
                            if len(output.split(task_token)) != 2:
                                format_error += 1
                                A_golds.pop()
                                continue
                            output1 = output.split(task_token)[1]
                            output_goal = output1.replace('\"', '').replace('[', '').replace(']', '').replace('\'', '').replace('.', '').strip() 
                            
                            if len(output_goal.split(','))>=3:
                                R_pred = output_goal.split(',')[1].replace('\'', '').replace('\"', '').strip()
                                A_preds.append(R_pred)  # process for natural langauage
                            else:
                                A_preds.append(output_goal)  # process for natural langauage   
                        
                        else:
                            format_error += 1
                            A_golds.pop()
                    else:
                        A_preds.append([d["output"].strip().lower()])  # process for natural langauage
                    # breakpoint()
                    # print(A_golds)
                    # print(A_preds)
                
                # print(goal_dict[d["dial-turn_id"]])
                # print(A_golds) 
                # print(A_preds)   
                # breakpoint()
            print('format error: ', format_error)
            print(len(A_golds), len(A_preds))
            
            refs = A_golds
            preds = A_preds
            accuracy = 0
            for i in range(len(refs)):
                if refs[i] == preds[i]:
                    accuracy += 1
            print("Accuracy: ", accuracy/len(refs))
            # breakpoint()
            f1_scores = goal_f1_score(preds, refs, args.dataset_name)
            # f1_scores = [f1_score(A_preds, A_golds)]
            print('format error: ', format_error)
            print(len(A_golds), len(A_preds))
            print('Running P/R/F1/Accuracy for ' + ' ' + args.dataset_name + '-----------------------------')
            print('P/R/F1: ', f1_scores, accuracy/len(refs))
            auto_scores = f1_scores
            
        elif args.task == 'REC':
            # breakpoint()
            format_error = 0
            for d in data["data"]:
                if type(d["Output"]) is list:
                    A_golds.append([item.strip().lower() for item in d["Output"]])
                else:
                    A_golds.append([d["Output"].strip().lower()])
                # breakpoint()
                ###process unmatch list file
                # d["output"] = d["output"].replace("\"", "").strip('][').split(', ')
                ###process unmatch list file
                if type(d["output"]) is list:
                    # breakpoint()
                    A_preds.append([item.strip().lower() for item in d["output"]])
                else:
                    # breakpoint()
                    if json_format:
                        if is_json(d["output"]):
                            a_preds = []
                            dj = json.loads(d["output"])
                            # if "system Resonse" not in dj:
                            #     print(dj)
                            #     breakpoint()
                            for key, item in dj.items():
                                a_preds.append(item)
                            A_preds.append(a_preds)  # process for json format
                            # breakpoint()
                        else:
                            format_error += 1
                            A_golds.pop()
                    elif special_format:
                        if len(d["output"].split("recommendation list is")) == 2:
                            output = d["output"].split("recommendation list is")[1] 
                            output_list = output.strip().replace('\"', '').replace('[', '').replace(']', '').split(',') 
                            A_preds.append([item.strip().lower() for item in output_list])  # process for natural langauage
                        else:
                            format_error += 1
                            A_golds.pop()
                    else:
                        A_preds.append([d["output"].strip().lower()])  # process for natural langauage
                    
            print('format error: ', format_error)
            print(len(A_golds), len(A_preds))
            

            preds = A_preds
            refs = A_golds

            # if type(A_preds[0]) is not list: 
            #     preds = [pred for pred in A_preds]
            # else:
            #     preds = A_preds.copy()
            # if type(A_golds[0]) is not list: 
            #     refs = [ref for ref in A_golds]
            # else:
            #     refs = A_golds.copy()
            ndcg_scores = ndcg_score(preds, refs)
            mrr_scores = mrr_score(preds, refs)
            print('Running NDCG and MRR for ' + ' ' + args.dataset_name + '-----------------------------')
            print('NDCG@10/NDCG@50/MRR@10/MRR@50: ', ndcg_scores, mrr_scores)
            auto_scores = ndcg_scores + mrr_scores

    return auto_scores
        
    


if __name__ == '__main__':
    # file_dir = 'result_COT'
    
    # file_dir = 'result_goal'
    # file_path = 'goal2.json'
    file_dir = 'result_COT'
    # # file_path = 'DuRecDial_ENGLISH-CHATGPT-CRS-COT-GOAL-shot3-42.json'
    file_path = 'KG/DuRecDial_ENGLISH-CHATGPT-CRS-guiNone-shot3-42.json'
    # file_path = 'KG/DuRecDial_ENGLISH-CHATGPT-REC-guiNone-shot3-42.json'
    # file_dir = 'result_DG/KG1'
    # file_path = 'DuRecDial_ENGLISH-CHATGPT-CRS-guiNone-shot3-42.json'
    # file_path = 'DuRecDial_ENGLISH-CHATGPT-REC-guiNone-shot3-42.json'
    # file_dir = 'result_KGG'
    # file_dir = 'result_KGG'
    # file_path = 'DuRecDial_ENGLISH-CHATGPT-CRS-guiGOAL-shot3-42.json'
    # file_path = 'new1_KG/DuRecDial_ENGLISH-CHATGPT-CRS-guiKNOWLEDGE-shot3-42.json'
    # file_path = 'KG/DuRecDial_ENGLISH-CHATGPT-REC-guiKNOWLEDGE-shot3-42.json'

    ###for LLAMA2-7b
    # file_dir = 'LLAMA2/result_DG/KG'
    # file_path = 'DuRecDial_ENGLISH-LLAMA2-CRS-guiNone-shot3-42.json'
    # file_path = 'DuRecDial_ENGLISH-LLAMA2-REC-guiNone-shot3-42.json'
    # file_dir = 'LLAMA2/result_COT'
    # file_path = 'DuRecDial_ENGLISH-LLAMA2-CRS-COT-GOAL-shot3-42.json'
    # file_path = 'DuRecDial_ENGLISH-LLAMA2-CRS-COT-KG-shot3-42.json'
    # file_path = 'DuRecDial_ENGLISH-LLAMA2-REC-COT-KG-shot3-42.json'
    # file_dir = 'LLAMA2/result_KGG'
    # file_path = 'DuRecDial_ENGLISH-LLAMA2-CRS-guiGOAL-shot3-42.json'
    # file_path = 'KG/DuRecDial_ENGLISH-LLAMA2-CRS-guiKNOWLEDGE-shot3-42.json'
    # file_path = 'KG/DuRecDial_ENGLISH-LLAMA2-REC-guiKNOWLEDGE-shot3-42.json'
    # file_path = 'DuRecDial_ENGLISH-LLAMA2-CHAT-None-shot1-42.json'

    ####for LLAMA2-13b
    # file_dir = 'LLAMA2/result_DG/KG'
    # file_path = 'DuRecDial_ENGLISH-LLAMA2-13-chat-CRS-guiNone-shot3-42.json'
    # file_path = 'DuRecDial_ENGLISH-LLAMA2-13-chat-REC-guiNone-shot3-42.json'
    # file_dir = 'LLAMA2/result_COT'
    # file_path = 'DuRecDial_ENGLISH-LLAMA2-13-chat-CRS-guiNone-shot3-42.json'
    # file_path = 'KG/DuRecDial_ENGLISH-LLAMA2-13-chat-CRS-guiNone-shot3-42.json'
    # file_path = 'KG/DuRecDial_ENGLISH-LLAMA2-13-chat-REC-guiNone-shot3-42.json'
    # file_dir = 'LLAMA2/result_KGG'
    # file_dir = 'result_KGG'
    # file_path = 'DuRecDial_ENGLISH-LLAMA2-13-chat-CRS-guiGOAL-shot3-42.json'
    # file_path = 'new_KG/DuRecDial_ENGLISH-LLAMA2-13-chat-CRS-guiKNOWLEDGE-shot3-42.json'
    # file_path = 'KG/DuRecDial_ENGLISH-LLAMA2-13-chat-REC-guiKNOWLEDGE-shot3-42.json'
    # file_path = 'DuRecDial_ENGLISH-LLAMA2-13-chat-CHAT-None-shot1-42.json'

    # file_dir = "result_know"
    # file_path = "DuRecDial_ENGLISH-LLAMA2-13-chat-KNOWLEDGE-guiKG_R-shot3-42.json"
    # file_path = "DuRecDial_ENGLISH-GPT4-KNOWLEDGE-guiKG_R-shot3-42.json"

    # file_dir = "result_KGG/both"
    # file_path = "DuRecDial_ENGLISH-CHATGPT-CRS-guiBOTH-shot3-42.json"
    # file_path = "DuRecDial_ENGLISH-LLAMA2-CRS-guiBOTH-shot3-42.json"

    # file_dir = "result_ChatCRS"
    # file_path = "DuRecDial_ENGLISH-CHATGPT-REC-guiKNOWLEDGE-shot3-42.json"
    # file_path = "DuRecDial_ENGLISH-LLAMA2-13-chat-CRS-guiBOTH-shot3-42.json"
    # file_path = "DuRecDial_ENGLISH-CHATGPT-CRS-guiBOTH-shot3-42.json"
    # file_path = "DuRecDial_ENGLISH-LLAMA2-13-chat-REC-guiKNOWLEDGE-shot3-42.json"

    # file_dir = "LLAMA2/result_COT"
    # file_path = "DuRecDial_ENGLISH-LLAMA2-13-chat-KNOWLEDGE-guiKG_R-shot3-42.json"
    # file_path = "KG/DuRecDial_ENGLISH-LLAMA2-13-chat-CRS-guiNone-shot3-42.json"

    # file_path = "DuRecDial_ENGLISH-LLAMA2-13-chat-CRS-guiBOTH-shot3-42.json"
    # print("Instructions: You are an excellent conversational recommender that helps the user achieve recommendation-related goals through conversations. Your task is to generate the next system reponse given the dialogue history and satisfy the conversation goal. Please output in json format and system response must satisfy the system dialogue goal provided.\n\nInput: {\nDialogue History: [user]:Hello! Do you know who starred in the movie Flying Dagger?\n[system]:Yes, of course I know that. It's Jimmy Lin.\n[user]:OK, thank you.\n[system]:He is an amazing all-rounder, and he has won Chinese Youth Leader in 2014.\n[user]:He is my favorite star.\n[system]:He has also won the TV Dramas Awards Made in China for Most Appealing Actor in 2011, and was awarded as American International Outstanding Youth in 2003.\n[user]:Impressive! I will always be proud of him.\n[system]:\nSystem Dialogue Goal: ['Movie recommendation']}\n\nOutput: {System Resonse: }Since you like him so much, I wanna recommend to you the movie To Miss with Love, which is starred by him. There are many classic lines in it. You can go to see it.\n\n\nInstructions: You are an excellent conversational recommender that helps the user achieve recommendation-related goals through conversations. Your task is to generate the next system reponse given the dialogue history and satisfy the conversation goal. Please output in json format and system response must satisfy the system dialogue goal provided.\n\nInput: {\nDialogue History: [user]:Hello! Do you know who starred in the movie Flying Dagger?\n[system]:Yes, of course I know that. It's Jimmy Lin.\n[user]:OK, thank you.\n[system]:He is an amazing all-rounder, and he has won Chinese Youth Leader in 2014.\n[user]:He is my favorite star.\n[system]:He has also won the TV Dramas Awards Made in China for Most Appealing Actor in 2011, and was awarded as American International Outstanding Youth in 2003.\n[user]:Impressive! I will always be proud of him.\n[system]:Since you like him so much, I wanna recommend to you the movie To Miss with Love, which is starred by him. There are many classic lines in it. You can go to see it.\n[user]:What kind of movie is it?\n[system]:\nSystem Dialogue Goal: ['Movie recommendation']}\n\nOutput: {System Resonse: }It's a comedy. Very funny!\n\n\nInstructions: You are an excellent conversational recommender that helps the user achieve recommendation-related goals through conversations. Your task is to generate the next system reponse given the dialogue history and satisfy the conversation goal. Please output in json format and system response must satisfy the system dialogue goal provided.\n\nInput: {\nDialogue History: [user]:What time is it now?\n[system]:It's 22 o'clock.\n[user]:Thank you.\n[system]:It's sunny with south wind today. The highest temperature is 12\u2103, and the lowest temperature is 1\u2103.\n[user]:No wonder I feel a little cold at home.\n[system]:\nSystem Dialogue Goal: ['Weather notification']}\n\nOutput: {System Resonse: }Yeah, it's getting colder. Please keep warm!\n\n\nInstructions: You are an excellent conversational recommender that helps the user achieve recommendation-related goals through conversations. Your task is to generate the next system reponse given the dialogue history and satisfy the conversation goal. Please output in json format and system response must satisfy the system dialogue goal provided.\n\nInput: {\nDialogue History: [user]:What is Xun Zhou's star sign?\n[system]:It's Libra.\n[user]:Good for you! You know so much.\n[system]:I also know that she has won the Asian Film Awards for Best Actress.\n[user]:She's my idol. Her acting skills are excellent.\n[system]:\nSystem Dialogue Goal: ['Chat about stars']}\n\nOutput: {System Resonse: }She was born for acting and has won the Chinese Film Media Awards for Best Actress.")
    # print("Instructions: You are an excellent conversational recommender that helps the user achieve recommendation-related goals through conversations. Your task is to generate the next system reponse given the dialogue history and satisfy the conversation goal. Please output in json format and system response must satisfy the system dialogue goal provided.\n\nInput: {\nDialogue History: [user]:Hello! Do you know who starred in the movie Flying Dagger?\n[system]:Yes, of course I know that. It's Jimmy Lin.\n[user]:OK, thank you.\n[system]:He is an amazing all-rounder, and he has won Chinese Youth Leader in 2014.\n[user]:He is my favorite star.\n[system]:He has also won the TV Dramas Awards Made in China for Most Appealing Actor in 2011, and was awarded as American International Outstanding Youth in 2003.\n[user]:Impressive! I will always be proud of him.\n[system]:\nSystem Dialogue Goal: ['Movie recommendation']}\n\nOutput: {System Resonse: Since you like him so much, I wanna recommend to you the movie To Miss with Love, which is starred by him. There are many classic lines in it. You can go to see it.}\n\n\nInstructions: You are an excellent conversational recommender that helps the user achieve recommendation-related goals through conversations. Your task is to generate the next system reponse given the dialogue history and satisfy the conversation goal. Please output in json format and system response must satisfy the system dialogue goal provided.\n\nInput: {\nDialogue History: [user]:Hello! Do you know who starred in the movie Flying Dagger?\n[system]:Yes, of course I know that. It's Jimmy Lin.\n[user]:OK, thank you.\n[system]:He is an amazing all-rounder, and he has won Chinese Youth Leader in 2014.\n[user]:He is my favorite star.\n[system]:He has also won the TV Dramas Awards Made in China for Most Appealing Actor in 2011, and was awarded as American International Outstanding Youth in 2003.\n[user]:Impressive! I will always be proud of him.\n[system]:Since you like him so much, I wanna recommend to you the movie To Miss with Love, which is starred by him. There are many classic lines in it. You can go to see it.\n[user]:What kind of movie is it?\n[system]:\nSystem Dialogue Goal: ['Movie recommendation']}\n\nOutput: {System Resonse: It's a comedy. Very funny!}\n\n\nInstructions: You are an excellent conversational recommender that helps the user achieve recommendation-related goals through conversations. Your task is to generate the next system reponse given the dialogue history and satisfy the conversation goal. Please output in json format and system response must satisfy the system dialogue goal provided.\n\nInput: {\nDialogue History: [user]:What time is it now?\n[system]:It's 22 o'clock.\n[user]:Thank you.\n[system]:It's sunny with south wind today. The highest temperature is 12\u2103, and the lowest temperature is 1\u2103.\n[user]:No wonder I feel a little cold at home.\n[system]:\nSystem Dialogue Goal: ['Weather notification']}\n\nOutput: {System Resonse: Yeah, it's getting colder. Please keep warm!}\n\n\nInstructions: You are an excellent conversational recommender that helps the user achieve recommendation-related goals through conversations. Your task is to generate the next system reponse given the dialogue history and satisfy the conversation goal. Please output in json format and system response must satisfy the system dialogue goal provided.\n\nInput: {\nDialogue History: [user]:What's the height of Jay Chou, do you know?\n[system]:175cm.\n[user]:Good for you.\n[system]:Jay Chou is an entertainment and sports star\n[user]:Yes, he is the God of the whole nation.\n[system]:Yes, he is one of top singers in Taiwan.\n[user]:He has also won eight annual record sales Champions in Greater China area.\n[system]:I'd like to recommend one of his songs, Snake Dance. The whole song combines Egyptian and Middle Eastern folk music styles with hip-hop, as well as seductive singing, creating a unique and charming hip-hop duet song.\n[user]:It sounds very good.\n[system]:Yes, it's a good song.\n[user]:Then I want to hear it.\n[system]:Do you need me to play it for you?\n[user]:Yes.\n[system]:\nSystem Dialogue Goal: ['Play music']}\n\nOutput: {System Resonse: There you go.}")
    # print("Instructions: {INST}\n\nInput: {\n\"Dialogue History\": \"{Q}\"}\n\nOutput: {\"System Dialogue Goal\": \"{G}\",\"System Resonse\": \"{A}\"}")
    # A = "{\"System Dialogue Goal\": \"['Movie recommendation']\",\"System Resonse\": \"Children From The Distant Planet is very interesting. I heard it's about a story between a robot and a monster. You can go to see it.\"}\n"
    # print("You are an excellent conversational recommender that helps the user achieve recommendation-related goals through conversations. Your task is to generate a list of recommendation items that the user likes given the dialogue history. Please limit your response to 100 items in a ranking list without any sentences.\n\nInput: [user]:Hello! Could you tell me who starred in the movie Bruce Lee: A Warrior's Journey?\n[system]:Yes, it's Jackie Chan.\n[user]:Wow, you know so much. Good for you!\n[system]:Hehe, I'm known as an encyclopedia! I also know he is a model worker and a great philanthropist.\n[user]:Yeah, he's really outstanding. And he has won many awards. Do you know what awards they are?\n[system]:Yes. He has won the Teen Choice Award for Choice Chemistry, and the Golden Rooster Award for Best Actor.\n[user]:Wow, that's my favorite idol, who has great acting skills.\n[recommendation item]:\n\nRecommendations: [\"Jackie Chan: My Stunts\", \"Wanzhou Roasted Fish\", \"Hand in Hand\", \"Tough Pill\", \"Sliced Fish in Hot Chili Oil\", \"Liuguosanchun Sichuan Cuisine (Global Financial Center Store)\", \"The Little Girl Under the Streetlight\", \"Blamefully Beautiful\", \"Fly Me to Polaris\", \"Days of Tomorrow\", \"My Dear Son\", \"Producer Cypher\", \"Love Now\", \"Lonely Shadow\", \"The Days of Being Dumb\", \"Beautiful\", \"A Chinese Ghost Story\", \"Heartburn\", \"Hope for Love\", \"Four Girls Sichuan Cuisine\", \"All the Way\", \"When Distance Keeps us Apart\", \"Change Me\", \"All the Things You Never Knew\", \"Hand Phone\", \"Impossible to Miss You\", \"Time and Tide\", \"A Man Of Intention\", \"Run Toward the Future\", \"Where Are You Now\", \"The Crescent\", \"Flower Field Mistake+Sun & Moon of My Heart\", \"Open Fire\", \"Bicycle\", \"Kung Fu Panda 3\", \"To Me the Way\", \"Double Tap\", \"Sauteed Beef Fillet\", \"Murderer\", \"Who's the Keyman\", \"Blessing\", \"Love Net\", \"Golden Armor\", \"Perhaps Love\", \"Sacrifice\", \"Xiangyuxuan New Hunan Cuisine (Shouyihui Store)\", \"I Waited Until the Flower Withered\", \"Happy Camp: Bring Happiness Home\", \"McDull, Prince de la Bun\", \"The Best Voice\", \"Bright  Heart\", \"The Message\", \"Deserve\", \"Silence is Gold\", \"The Moon Represents My Heart\", \"Cold War\", \"Anna Magdalena\", \"July Rhapsody\", \"The Coldest Day\", \"Hollywood Hong-Kong\", \"A West Lake Moment\", \"American Dreams in China\", \"Magician\", \"Call for Love\", \"Last Winner\", \"Kiss Goodbye\", \"A Simple Song\", \"Life and Death Plunder\", \"There Is a Place\", \"Bridge of Faith\", \"New Shaolin Temple\", \"Just Started\", \"The Promise (Mandarin)\", \"Mojin-The Lost Legend\", \"Beef in Sour Soup\", \"Fatty Hu Hunan Restaurant\", \"1:99 Shorts\", \"Days of Being Wild\", \"Left Right Love Destiny\", \"Invisible Target\", \"One Night in Mongkok\", \"Forever Young\", \"Perfect Two\", \"Insanity\", \"The Bullet Vanishes\", \"Mapo Tofu\", \"Wonerful Happiness\", \"She Came to My Concert\", \"Bullet in the Head\", \"A Chinese Ghost Story III:TaoTaoTao\", \"Chop Socky: Cinema Hong Kong\", \"Second Time Around\", \"Happy Running\", \"After Leaving\", \"Thousands of Songs\", \"I Am Not Madame Bovary\", \"Jialinjiang Riverside Sichuan Cuisine\", \"Love in The Buff\", \"A Life Of Fighting Is But A Dream\"]\n\n\nYou are an excellent conversational recommender that helps the user achieve recommendation-related goals through conversations. Your task is to generate a list of recommendation items that the user likes given the dialogue history. Please limit your response to 100 items in a ranking list without any sentences.\n\nInput: [user]:Hi, do you know who the leading actor of the movie Vampire Family is?\n[system]:Jimmy Lin.\n[user]:OK. Thank you very much!\n[system]:Besides being an actor, Jim Lin is also an idol singer.\n[user]:Yes, he is my idol. I like him very much.\n[system]:And he's also a candidate for the 2012 inspirational youth list.\n[user]:Yes. He's an idol who seems not to get old.\n[recommendation item]:\n\nRecommendations:")
    # print("You are an excellent conversational recommender that helps the user achieve recommendation-related goals through conversations. Your task is to genenrate a list of recommendation items that the user likes given the dialogue history and match the provided topic guidance. Please limit your response to 50 items in a ranking list without any sentences.\n\nInput: [user]:Hello! Do you know who starred in the movie Flying Dagger?\n[system]:Yes, of course I know that. It's Jimmy Lin.\n[user]:OK, thank you.\n[system]:He is an amazing all-rounder, and he has won Chinese Youth Leader in 2014.\n[user]:He is my favorite star.\n[system]:He has also won the TV Dramas Awards Made in China for Most Appealing Actor in 2011, and was awarded as American International Outstanding Youth in 2003.\n[user]:Impressive! I will always be proud of him.\n[recommendation item]:\n\nRecommendations: \nTo Miss with Love\nMarinated Fish\nWanzhou Roasted Fish\nHand in Hand\nTough Pill\nSliced Fish in Hot Chili Oil\nLiuguosanchun Sichuan Cuisine (Global Financial Center Store)\nThe Little Girl Under the Streetlight\nBlamefully Beautiful\nFly Me to Polaris\nDays of Tomorrow\nMy Dear Son\nProducer Cypher\nLove Now\nLonely Shadow\nThe Days of Being Dumb\nBeautiful\nA Chinese Ghost Story\nHeartburn\nHope for Love\nFour Girls Sichuan Cuisine\nAll the Way\nWhen Distance Keeps us Apart\nChange Me\nAll the Things You Never Knew\nHand Phone\nImpossible to Miss You\nTime and Tide\nA Man Of Intention\nRun Toward the Future\nWhere Are You Now\nThe Crescent\nFlower Field Mistake+Sun & Moon of My Heart\nOpen Fire\nBicycle\nKung Fu Panda 3\nTo Me the Way\nDouble Tap\nSauteed Beef Fillet\nMurderer\nWho's the Keyman\nBlessing\nLove Net\nGolden Armor\nPerhaps Love\nSacrifice\nXiangyuxuan New Hunan Cuisine (Shouyihui Store)\nI Waited Until the Flower Withered\nHappy Camp: Bring Happiness Home\nMcDull, Prince de la Bun\n\n\nYou are an excellent conversational recommender that helps the user achieve recommendation-related goals through conversations. Your task is to genenrate a list of recommendation items that the user likes given the dialogue history and match the provided topic guidance. Please limit your response to 50 items in a ranking list without any sentences.\n\nInput: [system]:Good afternoon, handsome Fan Zhang.\n[user]:Good afternoon.\n[system]:How's your life recently?\n[user]:Generally speaking, it's very good. I've made progress in my study. I'm very happy.\n\n\nRecommendations:")
    # print( "You are an excellent conversational recommender that helps the user achieve recommendation-related goals through conversations. Your task is to generate a list of recommendation items that the user likes given the dialogue history. Please limit your response to 50 items in a ranking list without any sentences.\n\nInput: [user]:Hello! Could you tell me who starred in the movie Bruce Lee: A Warrior's Journey?\n[system]:Yes, it's Jackie Chan.\n[user]:Wow, you know so much. Good for you!\n[system]:Hehe, I'm known as an encyclopedia! I also know he is a model worker and a great philanthropist.\n[user]:Yeah, he's really outstanding. And he has won many awards. Do you know what awards they are?\n[system]:Yes. He has won the Teen Choice Award for Choice Chemistry, and the Golden Rooster Award for Best Actor.\n[user]:Wow, that's my favorite idol, who has great acting skills.\n[recommendation item]:\n\nRecommendations: [\"Jackie Chan: My Stunts\", \"Wanzhou Roasted Fish\", \"Hand in Hand\", \"Tough Pill\", \"Sliced Fish in Hot Chili Oil\", \"Liuguosanchun Sichuan Cuisine (Global Financial Center Store)\", \"The Little Girl Under the Streetlight\", \"Blamefully Beautiful\", \"Fly Me to Polaris\", \"Days of Tomorrow\", \"My Dear Son\", \"Producer Cypher\", \"Love Now\", \"Lonely Shadow\", \"The Days of Being Dumb\", \"Beautiful\", \"A Chinese Ghost Story\", \"Heartburn\", \"Hope for Love\", \"Four Girls Sichuan Cuisine\", \"All the Way\", \"When Distance Keeps us Apart\", \"Change Me\", \"All the Things You Never Knew\", \"Hand Phone\", \"Impossible to Miss You\", \"Time and Tide\", \"A Man Of Intention\", \"Run Toward the Future\", \"Where Are You Now\", \"The Crescent\", \"Flower Field Mistake+Sun & Moon of My Heart\", \"Open Fire\", \"Bicycle\", \"Kung Fu Panda 3\", \"To Me the Way\", \"Double Tap\", \"Sauteed Beef Fillet\", \"Murderer\", \"Who's the Keyman\", \"Blessing\", \"Love Net\", \"Golden Armor\", \"Perhaps Love\", \"Sacrifice\", \"Xiangyuxuan New Hunan Cuisine (Shouyihui Store)\", \"I Waited Until the Flower Withered\", \"Happy Camp: Bring Happiness Home\", \"McDull, Prince de la Bun\", \"The Best Voice\", \"Bright  Heart\", \"The Message\", \"Deserve\", \"Silence is Gold\", \"The Moon Represents My Heart\", \"Cold War\", \"Anna Magdalena\", \"July Rhapsody\", \"The Coldest Day\", \"Hollywood Hong-Kong\", \"A West Lake Moment\", \"American Dreams in China\", \"Magician\", \"Call for Love\", \"Last Winner\", \"Kiss Goodbye\", \"A Simple Song\", \"Life and Death Plunder\", \"There Is a Place\", \"Bridge of Faith\", \"New Shaolin Temple\", \"Just Started\", \"The Promise (Mandarin)\", \"Mojin-The Lost Legend\", \"Beef in Sour Soup\", \"Fatty Hu Hunan Restaurant\", \"1:99 Shorts\", \"Days of Being Wild\", \"Left Right Love Destiny\", \"Invisible Target\", \"One Night in Mongkok\", \"Forever Young\", \"Perfect Two\", \"Insanity\", \"The Bullet Vanishes\", \"Mapo Tofu\", \"Wonerful Happiness\", \"She Came to My Concert\", \"Bullet in the Head\", \"A Chinese Ghost Story III:TaoTaoTao\", \"Chop Socky: Cinema Hong Kong\", \"Second Time Around\", \"Happy Running\", \"After Leaving\", \"Thousands of Songs\", \"I Am Not Madame Bovary\", \"Jialinjiang Riverside Sichuan Cuisine\", \"Love in The Buff\", \"A Life Of Fighting Is But A Dream\"]\n\n\nYou are an excellent conversational recommender that helps the user achieve recommendation-related goals through conversations. Your task is to generate a list of recommendation items that the user likes given the dialogue history. Please limit your response to 50 items in a ranking list without any sentences.\n\nInput: [user]:Do you know who plays the leading role in  Left Right Love Destiny?\n[system]:Certainly, it's Cecilia Cheung.\n[user]:You know so much.\n[system]:well, let's talk about her. I remember she was called a natural beauty.\n[user]:Yes, she is very beautiful.\n[system]:Actually, her career is better. She was awarded as the Outstanding Artist from Asia at the Chinese Film Festival in New York and has won the Golden Bauhinia Award for Best Leading Actress.\n[user]:Ha ha, her acting skills are perfect.\n[recommendation item]:\n\nRecommendations:")
    
    file_name = os.path.join(file_dir, file_path)
    print(file_name)
    args = get_args()
    # add_guidance(file_name)

    # automatic_evaluation(args, file_name, json_format = False, special_format = True, special_task = 'GOAL') #, special_task = 'KNOWLEDGE'
    automatic_evaluation(args, file_name, json_format = False, special_format = True, special_task = 'KNOWLEDGE') #, special_task = 'KNOWLEDGE'
    # automatic_evaluation(args, file_name, json_format = False, special_format = True)