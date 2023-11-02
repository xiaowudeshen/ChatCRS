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

def goal_f1_score(pred_pt, gold_pt, args):
    # goal_dict = {}
    # with open('../data/{}/goal2id.txt'.format(args.dataset_name),'r',encoding='utf-8') as infile:
    #     for line in infile:
    #         items = line.strip().lower().split('\t')
    #         goal_dict[items[0]] = items[1]

    # def make_label(l, label_dict):
    #     length = len(label_dict)
    #     result = [0] * length
    #     for label in l:
    #         if label.strip().lower() == '':
    #             continue
    #         label = ''.join(label.strip().lower().split(' '))
    #         if label not in label_dict:
    #             continue
    #         result[int(label_dict[label])] = 1
    #     return result
    
    def get_metrics(y, y_pre, args):
        if args.dataset_name == 'DuRecDial_ENGLISH':
            macro_f1 = metrics.f1_score(y, y_pre, average='macro')
            macro_precision = metrics.precision_score(y, y_pre, average='macro')
            macro_recall = metrics.recall_score(y, y_pre, average='macro')
        # else:
        #     f1 = metrics.f1_score(y, y_pre, average=None).tolist()
        #     p = metrics.precision_score(y, y_pre, average=None).tolist()
        #     r = metrics.recall_score(y, y_pre, average=None).tolist()
        #     print(f1.count(0), p.count(0), r.count(0))
        #     macro_f1 = sum(f1)/(len(f1)-f1.count(0))
        #     macro_precision = sum(p)/(len(p)-p.count(0))
        #     macro_recall = sum(r)/(len(r)-r.count(0))
        # return macro_precision, macro_recall, macro_f1

    # if args.dataset_name == 'tgredial':
    #     reference = np.array([make_label(y, goal_dict) for y in gold_pt])
    #     candidate = np.array([make_label(y_pre, goal_dict) for y_pre in pred_pt])
    # else:
    reference = gold_pt
    candidate = pred_pt
    all_scores = list(get_metrics(reference, candidate, args.dataset_name))

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


def automatic_evaluation(args, path_to_data, json_format=True):
    task_dic = {
        "DuRecDial_ENGLISH": ["CRS", "CHAT", "REC", "TOPIC", "GOAL", "KNOWLEDGE"],
        "DuRecDial_CHINESE": ["CRS", "CHAT", "REC", "TOPIC", "GOAL"],
        "TG-Redial_CHINESE": ["CRS", "CHAT", "REC", "TOPIC", "GOAL"]
    }
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
    A_golds = []
    A_preds = []

    format_error = 0
    # json_format = True   # argument for json format
    for d in data["data"]:
        if type(d["Output"]) is list:
            A_golds.append([normalize_answer(item) for item in d["Output"]])
        else:
            A_golds.append(normalize_answer(d["Output"]))
        if type(d["output"]) is list:
            A_preds.append([normalize_answer(item) for item in d["output"]])
        else:
            
            if json_format:
                if is_json(d["output"]) and "System Resonse" in json.loads(d["output"]):
                    dj = json.loads(d["output"])
                    # if "system Resonse" not in dj:
                    #     print(dj)
                    #     breakpoint()
                    A_preds.append(dj["System Resonse"])  # process for json format
                else: 
                    format_error += 1
                    A_golds.pop()
            else:
                A_preds.append(normalize_answer(d["output"]))  # process for natural langauage
                    

            # dj = d["output"].replace("{System Resonse:", "").replace("}", "").replace("\n", "").strip()
            # A_preds.append(normalize_answer(dj))  # process for json format
        # print('$$$$$$$$$$$$$$$')
        # print(d["Output"])
        # print(dj)
        # breakpoint()
    print('format error: ', format_error)
    print(len(A_golds), len(A_preds))

    # A_preds = A_golds.copy()

    # json_format = True
    # if json_format:
    #     for d in A_preds:
    #         if is_json(d):
    #             dj = json.loads(d)[""]
    
    if args.dataset_name == 'DuRecDial_ENGLISH':
        if args.task in ['CRS', 'CHAT']:
            #running result for F1, blue, distinct
            #run bleu
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
        elif task == 'TOPIC':
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
        elif task == 'GOAL':
            if args.guidance is None:
                #take goal prediction as generation task same as topic prediction
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
            elif args.guidance == 'GOALLIST':
                refs = A_golds
                preds = A_preds
                f1_scores = goal_f1_score(preds, refs, args)
                print('Running P/R/F1 for ' + ' ' + args.dataset_name + '-----------------------------')
                print('P/R/F1: ', f1_scores)
                auto_scores = f1_scores
        elif task == 'KNOWLEDGE':
            #take knowledge prediction as generation task same as topic prediction
            refs = A_golds
            if A_preds is not list:
                preds = [pred for pred in A_preds] #if pred is one item
            else:
                preds = A_preds # if prd is a list
            hit_scores = topic_hit_score(preds, refs)
            # f1_scores = topic_f1_score(preds, refs)
            f1_scores = [f1_score(A_preds, A_golds)]
            print('Running P/R/F1 for ' + ' ' + args.dataset_name + '-----------------------------')
            print('P/R/F1/hits: ', f1_scores, hit_scores)
            
            # print('Running F1 for ' + ' ' + args.dataset_name + '-----------------------------')
            # print('F1: ', f1_scores)
            auto_scores = f1_scores + hit_scores
            
        elif task == 'REC':
            # breakpoint()
            if type(A_preds[0]) is not list: 
                preds = [pred for pred in A_preds]
            else:
                preds = A_preds.copy()
            if type(A_golds[0]) is not list: 
                refs = [ref for ref in A_golds]
            else:
                refs = A_golds.copy()
            ndcg_scores = ndcg_score(preds, refs)
            mrr_scores = mrr_score(preds, refs)
            print('Running NDCG and MRR for ' + ' ' + args.dataset_name + '-----------------------------')
            print('NDCG@10/NDCG@50/MRR@10/MRR@50: ', ndcg_scores, mrr_scores)
            auto_scores = ndcg_scores + mrr_scores

    return auto_scores
        
    


if __name__ == '__main__':
    file_dir = 'result_C'
    # file_path = 'DuRecDial_ENGLISH-CHATGPT-CHAT-None-shot1-42-quick_test5.json'
    # file_path = 'DuRecDial_ENGLISH-CHATGPT-TOPIC-None-shot1-42-quick_test5.json'
    # file_path = 'DuRecDial_ENGLISH-CHATGPT-GOAL-None-shot1-42-quick_test5.json'
    # # file_path = 'DuRecDial_ENGLISH-CHATGPT-KNOWLEDGE-None-shot1-42-quick_test5.json'
    # file_path = 'DuRecDial_ENGLISH-CHATGPT-REC-None-shot1-42-quick_test5.json'
    # file_path = 'DuRecDial_ENGLISH-LLAMA2-CRS-guiNone-shot3-42.json'
    # file_path = 'DuRecDial_ENGLISH-LLAMA2-CRS-guiREC-shot3-42.json'
    file_path = 'DuRecDial_ENGLISH-LLAMA2-CRS-guiGOAL-shot3-42-quick_test1000.json'
    # file_path = 'DuRecDial_ENGLISH-CHATGPT-CRS-guiNone-shot3-42-quick_test1000.json'
    # file_path = 'DuRecDial_ENGLISH-LLAMA2-CHAT-None-shot1-42.json'
    # print("Instructions: You are an excellent conversational recommender that helps the user achieve recommendation-related goals through conversations. Your task is to generate the next system reponse given the dialogue history and satisfy the conversation goal. Please output in json format and system response must satisfy the system dialogue goal provided.\n\nInput: {\nDialogue History: [user]:Hello! Do you know who starred in the movie Flying Dagger?\n[system]:Yes, of course I know that. It's Jimmy Lin.\n[user]:OK, thank you.\n[system]:He is an amazing all-rounder, and he has won Chinese Youth Leader in 2014.\n[user]:He is my favorite star.\n[system]:He has also won the TV Dramas Awards Made in China for Most Appealing Actor in 2011, and was awarded as American International Outstanding Youth in 2003.\n[user]:Impressive! I will always be proud of him.\n[system]:\nSystem Dialogue Goal: ['Movie recommendation']}\n\nOutput: {System Resonse: }Since you like him so much, I wanna recommend to you the movie To Miss with Love, which is starred by him. There are many classic lines in it. You can go to see it.\n\n\nInstructions: You are an excellent conversational recommender that helps the user achieve recommendation-related goals through conversations. Your task is to generate the next system reponse given the dialogue history and satisfy the conversation goal. Please output in json format and system response must satisfy the system dialogue goal provided.\n\nInput: {\nDialogue History: [user]:Hello! Do you know who starred in the movie Flying Dagger?\n[system]:Yes, of course I know that. It's Jimmy Lin.\n[user]:OK, thank you.\n[system]:He is an amazing all-rounder, and he has won Chinese Youth Leader in 2014.\n[user]:He is my favorite star.\n[system]:He has also won the TV Dramas Awards Made in China for Most Appealing Actor in 2011, and was awarded as American International Outstanding Youth in 2003.\n[user]:Impressive! I will always be proud of him.\n[system]:Since you like him so much, I wanna recommend to you the movie To Miss with Love, which is starred by him. There are many classic lines in it. You can go to see it.\n[user]:What kind of movie is it?\n[system]:\nSystem Dialogue Goal: ['Movie recommendation']}\n\nOutput: {System Resonse: }It's a comedy. Very funny!\n\n\nInstructions: You are an excellent conversational recommender that helps the user achieve recommendation-related goals through conversations. Your task is to generate the next system reponse given the dialogue history and satisfy the conversation goal. Please output in json format and system response must satisfy the system dialogue goal provided.\n\nInput: {\nDialogue History: [user]:What time is it now?\n[system]:It's 22 o'clock.\n[user]:Thank you.\n[system]:It's sunny with south wind today. The highest temperature is 12\u2103, and the lowest temperature is 1\u2103.\n[user]:No wonder I feel a little cold at home.\n[system]:\nSystem Dialogue Goal: ['Weather notification']}\n\nOutput: {System Resonse: }Yeah, it's getting colder. Please keep warm!\n\n\nInstructions: You are an excellent conversational recommender that helps the user achieve recommendation-related goals through conversations. Your task is to generate the next system reponse given the dialogue history and satisfy the conversation goal. Please output in json format and system response must satisfy the system dialogue goal provided.\n\nInput: {\nDialogue History: [user]:What is Xun Zhou's star sign?\n[system]:It's Libra.\n[user]:Good for you! You know so much.\n[system]:I also know that she has won the Asian Film Awards for Best Actress.\n[user]:She's my idol. Her acting skills are excellent.\n[system]:\nSystem Dialogue Goal: ['Chat about stars']}\n\nOutput: {System Resonse: }She was born for acting and has won the Chinese Film Media Awards for Best Actress.")
<<<<<<< HEAD
    print("Instructions: You are an excellent conversational recommender that helps the user achieve recommendation-related goals through conversations. Your task is to generate the next system reponse given the dialogue history and satisfy the conversation goal. Please output in json format and system response must satisfy the system dialogue goal provided.\n\nInput: {\nDialogue History: [user]:Hello! Do you know who starred in the movie Flying Dagger?\n[system]:Yes, of course I know that. It's Jimmy Lin.\n[user]:OK, thank you.\n[system]:He is an amazing all-rounder, and he has won Chinese Youth Leader in 2014.\n[user]:He is my favorite star.\n[system]:He has also won the TV Dramas Awards Made in China for Most Appealing Actor in 2011, and was awarded as American International Outstanding Youth in 2003.\n[user]:Impressive! I will always be proud of him.\n[system]:}\n\nOutput: {System Dialogue Goal: ['Movie recommendation']\nSystem Resonse: Since you like him so much, I wanna recommend to you the movie To Miss with Love, which is starred by him. There are many classic lines in it. You can go to see it.}\n\n\nInstructions: You are an excellent conversational recommender that helps the user achieve recommendation-related goals through conversations. Your task is to generate the next system reponse given the dialogue history and satisfy the conversation goal. Please output in json format and system response must satisfy the system dialogue goal provided.\n\nInput: {\nDialogue History: [user]:Hello! Do you know who starred in the movie Flying Dagger?\n[system]:Yes, of course I know that. It's Jimmy Lin.\n[user]:OK, thank you.\n[system]:He is an amazing all-rounder, and he has won Chinese Youth Leader in 2014.\n[user]:He is my favorite star.\n[system]:He has also won the TV Dramas Awards Made in China for Most Appealing Actor in 2011, and was awarded as American International Outstanding Youth in 2003.\n[user]:Impressive! I will always be proud of him.\n[system]:Since you like him so much, I wanna recommend to you the movie To Miss with Love, which is starred by him. There are many classic lines in it. You can go to see it.\n[user]:What kind of movie is it?\n[system]:}\n\nOutput: {System Dialogue Goal: ['Movie recommendation']\nSystem Resonse: It's a comedy. Very funny!}\n\n\nInstructions: You are an excellent conversational recommender that helps the user achieve recommendation-related goals through conversations. Your task is to generate the next system reponse given the dialogue history and satisfy the conversation goal. Please output in json format and system response must satisfy the system dialogue goal provided.\n\nInput: {\nDialogue History: [user]:What time is it now?\n[system]:It's 22 o'clock.\n[user]:Thank you.\n[system]:It's sunny with south wind today. The highest temperature is 12\u2103, and the lowest temperature is 1\u2103.\n[user]:No wonder I feel a little cold at home.\n[system]:}\n\nOutput: {System Dialogue Goal: ['Weather notification']\nSystem Resonse: Yeah, it's getting colder. Please keep warm!}\n\n\nInstructions: You are an excellent conversational recommender that helps the user achieve recommendation-related goals through conversations. Your task is to generate the next system reponse given the dialogue history and satisfy the conversation goal. Please output in json format and system response must satisfy the system dialogue goal provided.\n\nInput: {\nDialogue History: [user]:Hello! What's the weather like today?\n[system]:It's cloudy with no continuous wind direction. The highest temperature will be 24\u2103, and the lowest temperature will be 20\u2103.\n[user]:OK, thanks!\n[system]:}\n\nOutput:")
    
    # file_name = os.path.join(file_dir, file_path)
    # print(file_name)
    # args = get_args()
    # automatic_evaluation(args, file_name)
=======
    # print("Instructions: You are an excellent conversational recommender that helps the user achieve recommendation-related goals through conversations. Your task is to generate the next system reponse given the dialogue history and satisfy the conversation goal. Please output in json format and system response must satisfy the system dialogue goal provided.\n\nInput: {\nDialogue History: [user]:Hello! Do you know who starred in the movie Flying Dagger?\n[system]:Yes, of course I know that. It's Jimmy Lin.\n[user]:OK, thank you.\n[system]:He is an amazing all-rounder, and he has won Chinese Youth Leader in 2014.\n[user]:He is my favorite star.\n[system]:He has also won the TV Dramas Awards Made in China for Most Appealing Actor in 2011, and was awarded as American International Outstanding Youth in 2003.\n[user]:Impressive! I will always be proud of him.\n[system]:\nSystem Dialogue Goal: ['Movie recommendation']}\n\nOutput: {System Resonse: Since you like him so much, I wanna recommend to you the movie To Miss with Love, which is starred by him. There are many classic lines in it. You can go to see it.}\n\n\nInstructions: You are an excellent conversational recommender that helps the user achieve recommendation-related goals through conversations. Your task is to generate the next system reponse given the dialogue history and satisfy the conversation goal. Please output in json format and system response must satisfy the system dialogue goal provided.\n\nInput: {\nDialogue History: [user]:Hello! Do you know who starred in the movie Flying Dagger?\n[system]:Yes, of course I know that. It's Jimmy Lin.\n[user]:OK, thank you.\n[system]:He is an amazing all-rounder, and he has won Chinese Youth Leader in 2014.\n[user]:He is my favorite star.\n[system]:He has also won the TV Dramas Awards Made in China for Most Appealing Actor in 2011, and was awarded as American International Outstanding Youth in 2003.\n[user]:Impressive! I will always be proud of him.\n[system]:Since you like him so much, I wanna recommend to you the movie To Miss with Love, which is starred by him. There are many classic lines in it. You can go to see it.\n[user]:What kind of movie is it?\n[system]:\nSystem Dialogue Goal: ['Movie recommendation']}\n\nOutput: {System Resonse: It's a comedy. Very funny!}\n\n\nInstructions: You are an excellent conversational recommender that helps the user achieve recommendation-related goals through conversations. Your task is to generate the next system reponse given the dialogue history and satisfy the conversation goal. Please output in json format and system response must satisfy the system dialogue goal provided.\n\nInput: {\nDialogue History: [user]:What time is it now?\n[system]:It's 22 o'clock.\n[user]:Thank you.\n[system]:It's sunny with south wind today. The highest temperature is 12\u2103, and the lowest temperature is 1\u2103.\n[user]:No wonder I feel a little cold at home.\n[system]:\nSystem Dialogue Goal: ['Weather notification']}\n\nOutput: {System Resonse: Yeah, it's getting colder. Please keep warm!}\n\n\nInstructions: You are an excellent conversational recommender that helps the user achieve recommendation-related goals through conversations. Your task is to generate the next system reponse given the dialogue history and satisfy the conversation goal. Please output in json format and system response must satisfy the system dialogue goal provided.\n\nInput: {\nDialogue History: [user]:What's the height of Jay Chou, do you know?\n[system]:175cm.\n[user]:Good for you.\n[system]:Jay Chou is an entertainment and sports star\n[user]:Yes, he is the God of the whole nation.\n[system]:Yes, he is one of top singers in Taiwan.\n[user]:He has also won eight annual record sales Champions in Greater China area.\n[system]:I'd like to recommend one of his songs, Snake Dance. The whole song combines Egyptian and Middle Eastern folk music styles with hip-hop, as well as seductive singing, creating a unique and charming hip-hop duet song.\n[user]:It sounds very good.\n[system]:Yes, it's a good song.\n[user]:Then I want to hear it.\n[system]:Do you need me to play it for you?\n[user]:Yes.\n[system]:\nSystem Dialogue Goal: ['Play music']}\n\nOutput: {System Resonse: There you go.}")
    # print("Instructions: {INST}\n\nInput: {\n\"Dialogue History\": \"{Q}\"}\n\nOutput: {\"System Dialogue Goal\": \"{G}\",\"System Resonse\": \"{A}\"}")
    # A = "{\"System Dialogue Goal\": \"['Movie recommendation']\",\"System Resonse\": \"Children From The Distant Planet is very interesting. I heard it's about a story between a robot and a monster. You can go to see it.\"}\n"
 
    file_name = os.path.join(file_dir, file_path)
    print(file_name)
    args = get_args()
    automatic_evaluation(args, file_name)
>>>>>>> 3e7316fa41b14a7f0d93e06fb43d540b3d43af4e
