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
        args.dataset_name = "TG-Redial_CHINESE"
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
    if args.dataset_name == 'TG-Redial_CHINESE':
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
                # if len(d["guide_message"][0][0]) == 2:
                #     A_golds.append(d["guide_message"][0][0][1].strip().lower())
                # else:
                #     A_golds.append(d["guide_message"][0][0].strip().lower())
                out = ''
                if type(d["Output"]) is list:
                    for item in d["Output"]:
                        out += item
                    A_golds.append(out.replace(' ', ''))
                else:
                    A_golds.append(d["Output"].replace(' ', ''))
                        
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
                            output_list = output.strip().replace('\"', '').replace('[', '').replace(']', '').replace('《', '').replace('》', '').split(',') 
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
            print(preds[:10])
            print(refs[:10])
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
    # # file_path = "goal_full.json"
    # file_path = "TG-Redial_CHINESE-CHATGPT-GOAL-guiNone-shot1-42.json"
    # file_dir = 'result_goal'
    # file_path = 'goal2.json'
    # file_dir = 'result_COT'
    # file_path = 'TG-Redial_CHINESE-CHATGPT-CRS-guiNone-shot3-42.json'
    # # file_path = 'DuRecDial_ENGLISH-CHATGPT-CRS-COT-GOAL-shot3-42.json'
    # file_path = 'KG/DuRecDial_ENGLISH-CHATGPT-CRS-guiNone-shot3-42.json'
    # file_path = 'KG/DuRecDial_ENGLISH-CHATGPT-REC-guiNone-shot3-42.json'
    # file_dir = 'result_DG/KG_new'
    # file_path = 'DuRecDial_ENGLISH-CHATGPT-CRS-guiNone-shot3-42.json'
    # file_path = 'DuRecDial_ENGLISH-CHATGPT-REC-guiNone-shot3-42.json'
    # file_dir = 'result_KGG'
    # file_path = 'TG-Redial_CHINESE-CHATGPT-CRS-guiGOAL-shot3-42-quick_test1000.json'
    # file_dir = 'result_KGG'
    # file_path = 'DuRecDial_ENGLISH-CHATGPT-CRS-guiGOAL-shot3-42.json'
    # file_path = 'new_KG/DuRecDial_ENGLISH-CHATGPT-CRS-guiKNOWLEDGE-shot3-42.json'
    # file_path = 'KG/DuRecDial_ENGLISH-CHATGPT-REC-guiKNOWLEDGE-shot3-42.json'

    # file_dir = 'result_DG/KG'
    # file_path = 'TG-Redial_CHINESE-CHATGPT-REC-guiNone-shot3-42.json'

    file_dir = 'result_ChatCRS'
    file_path = 'TG-Redial_CHINESE-CHATGPT-CRS-guiBOTH-shot3-42.json'
    # file_path = 'TG-Redial_CHINESE-CHATGPT-REC-guiKNOWLEDGE-shot3-42.json'


    # file_dir = 'result_LLAMA'
    # # file_path =   'TG-Redial_CHINESE-LLAMA_C-CRS-guiNone-shot3-42.json'
    # file_path = 'KG-1/TG-Redial_CHINESE-LLAMA_C-REC-guiNone-shot3-42.json'
    file_name = os.path.join(file_dir, file_path)
    print(file_name)
    args = get_args()
    # add_guidance(file_name)

    # automatic_evaluation(args, file_name, json_format = False, special_format = True, special_task = 'GOAL') #, special_task = 'KNOWLEDGE'
    # automatic_evaluation(args, file_name, json_format = False, special_format = True, special_task = 'KNOWLEDGE') #, special_task = 'KNOWLEDGE'
    automatic_evaluation(args, file_name, json_format = False, special_format = True)