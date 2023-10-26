# Data: Oct, 2023
# Author: Victor Li
# This file is used to save the scripts for analysis
# including the followings:
# - dataset analysis



import json
import os




def Get_analysis_for_datasets():
    dataset_dic = {
        "durecdial": ["du_en_dev.json", "du_en_test.json", "du_en_train.json"],
        "opendialkg": ["opendialkg.json"],
        "redial": ["re_train.json", "re_test.json"],
        "tgredial": ["tg_train.json", "tg_test.json", "tg_valid.json"]
    }
    result_dic = {}
    goal_dic = {}
    topic_dic = {}
    item_dic = {}
    for dataset in dataset_dic:
        data_dir = os.path.join("../data_p", dataset)
        for file in dataset_dic[dataset]:
            data_file = os.path.join(data_dir, file)
            with open(data_file, "r") as f:
                data = json.load(f)
                result_dic[dataset + "_" + file] = {
                    "dialogue_num": len(data),
                    "dial_goal": 0,
                    "dial_goal_ratio": 0.0,
                    "dial_topic": 0,
                    "dial_topic_ratio": 0.0,
                    "dial_kg": 0,
                    "dial_kg_ratio": 0.0,
                    "turn_num": 0,
                    "turn_goal": 0,
                    "turn_goal_ratio": 0.0,
                    "turn_topic": 0,
                    "turn_topic_ratio": 0.0,
                    "turn_kg": 0,
                    "turn_kg_ratio": 0.0
                }
                for _, dialogue in data.items():
                    goal_flag = False
                    topic_flag = False
                    kg_flag = False
                    # breakpoint()
                    result_dic[dataset + "_" + file]["turn_num"] += len(dialogue["messages"])
                    for turn in dialogue["messages"]:
                        if turn["goal"]!=[]:
                            result_dic[dataset + "_" + file]["turn_goal"] += 1
                            if not goal_flag:
                                result_dic[dataset + "_" + file]["dial_goal"] += 1
                                goal_flag = True
                        if turn["topic"]!=[]:
                            result_dic[dataset + "_" + file]["turn_topic"] += 1
                            if not topic_flag:
                                result_dic[dataset + "_" + file]["dial_topic"] += 1
                                topic_flag = True
                        if turn["knowledge"]!=[]:
                            result_dic[dataset + "_" + file]["turn_kg"] += 1
                            if not kg_flag:
                                result_dic[dataset + "_" + file]["dial_kg"] += 1
                                kg_flag = True

                result_dic[dataset + "_" + file]["dial_goal_ratio"] = result_dic[dataset + "_" + file]["dial_goal"] / result_dic[dataset + "_" + file]["dialogue_num"]
                result_dic[dataset + "_" + file]["dial_topic_ratio"] = result_dic[dataset + "_" + file]["dial_topic"] / result_dic[dataset + "_" + file]["dialogue_num"]
                result_dic[dataset + "_" + file]["dial_kg_ratio"] = result_dic[dataset + "_" + file]["dial_kg"] / result_dic[dataset + "_" + file]["dialogue_num"]
                result_dic[dataset + "_" + file]["turn_goal_ratio"] = result_dic[dataset + "_" + file]["turn_goal"] / result_dic[dataset + "_" + file]["turn_num"]
                result_dic[dataset + "_" + file]["turn_topic_ratio"] = result_dic[dataset + "_" + file]["turn_topic"] / result_dic[dataset + "_" + file]["turn_num"]
                result_dic[dataset + "_" + file]["turn_kg_ratio"] = result_dic[dataset + "_" + file]["turn_kg"] / result_dic[dataset + "_" + file]["turn_num"]
    
    # out_file = os.path.join("data_p", "analysis.json")
    # with open(out_file, "w") as f:
    #     json.dump(result_dic, f, indent=4)
    return result_dic


                            
def Get_analysis_for_goal_topic_item():
    dataset_dic = {
        "durecdial": ["du_en_dev.json", "du_en_test.json"],
        "tgredial": ["tg_test.json", "tg_valid.json"]
    }
    
    for dataset in dataset_dic:
        goal_list = []
        topic_list = []
        item_list = []
        data_dir = os.path.join("../data_p", dataset)
        for file in dataset_dic[dataset]:
            data_file = os.path.join(data_dir, file)
            with open(data_file, "r", encoding='utf-8') as f:
                data = json.load(f)
                
                for _, dialogue in data.items():
                   
                    # breakpoint()
                    for turn in dialogue["messages"]:
                        if turn["goal"]!=[]:
                            if turn["goal"] not in goal_list:
                                goal_list.append(turn["goal"])
                        if turn["topic"]!=[]:
                            if turn["topic"] not in topic_list:
                                topic_list.append(turn["topic"])
                        if turn["mention"]!=[]:
                            for item in turn["mention"]:
                                if item not in item_list:
                                    item_list.append(item)
        if 'tgredial' in dataset:
            print(goal_list)
        goal_dict = {}
        topic_dict = {}
        item_dict = {}
        dict_names = [goal_dict, topic_dict, item_dict]
        for i, L in enumerate([goal_list, topic_list, item_list]):
            L_dict = dict_names[i]
            for idx, item in enumerate(L):
                if len(item) == 1:
                    L_dict[idx] = item[0]
                else:
                    L_dict[idx] = item

                    
        out_file = os.path.join("../instructions/GTR_dict", dataset)
        if not os.path.exists(out_file):
            os.makedirs(out_file)

        out_file1 = os.path.join(out_file, "goal_dict.json")
        with open(out_file1, "w", encoding='utf-8') as f:
            json.dump(goal_dict, f, indent=4)

        out_file2 = os.path.join(out_file, "topic_dict.json")
        with open(out_file2, "w", encoding='utf-8') as f:
            json.dump(topic_dict, f, indent=4) 

        out_file3 = os.path.join(out_file, "item_dict.json")
        with open(out_file3, "w", encoding='utf-8') as f:
            json.dump(item_dict, f, indent=4)

    












if __name__ == "__main__":
    Get_analysis_for_goal_topic_item()
    
    