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
    
    out_file = os.path.join("data_p", "analysis.json")
    with open(out_file, "w") as f:
        json.dump(result_dic, f, indent=4)
    return result_dic


                            













if __name__ == "__main__":
    result_dic = Get_analysis_for_datasets()
    
    