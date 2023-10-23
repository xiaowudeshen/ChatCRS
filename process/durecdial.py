#Date: 16/08/2023
#Author: Victor Li
#Email: lichuang@u.nus.edu


'''
This file is used to process the DuRecDial dataset into the standard format of MultiWoz dataset
Input file: 
DuRecDial/data/en_train.txt, DuRecDial/data/en_dev.txt, DuRecDial/data/en_test.txt
DuRecDial/data/train_knowledge.json, DuRecDial/data/dev_knowledge.json, DuRecDial/data/test_knowledge.json
Output file:
DuRecDial/data/train.json, DuRecDial/data/dev.json, DuRecDial/data/test.json
'''


import csv
import collections
import torch
import numpy as np
import os
import json
import ast
import re


class Durecdial():
    def __init__(self, durecdial_path):
        self.path = durecdial_path
        self.path1 = '../data_p/durecdial'
        if not os.path.exists(self.path1):
            os.makedirs(self.path1)
        self.train_dialogs = self.txt_to_json(os.path.join(self.path, "en_train.txt"), os.path.join(self.path1, "du_en_train.json"), os.path.join(self.path1, "du_en_train_knowledge.json"), action = "train")
        self.train_dialogs = self.txt_to_json(os.path.join(self.path, "zh_train.txt"), os.path.join(self.path1, "du_zh_train.json"), os.path.join(self.path1, "du_zh_train_knowledge.json"), action = "train")
        self.dev_dialogs = self.txt_to_json(os.path.join(self.path, "en_dev.txt"), os.path.join(self.path1, "du_en_dev.json"), os.path.join(self.path1, "du_en_dev_knowledge.json"), action = "dev")
        self.dev_dialogs = self.txt_to_json(os.path.join(self.path, "zh_dev.txt"), os.path.join(self.path1, "du_zh_dev.json"), os.path.join(self.path1, "du_zh_dev_knowledge.json"), action = "dev")
        self.test_dialogs = self.txt_to_json(os.path.join(self.path, "en_test.txt"), os.path.join(self.path1, "du_en_test.json"), os.path.join(self.path1, "du_en_test_knowledge.json"), action = "test")
        self.test_dialogs = self.txt_to_json(os.path.join(self.path, "zh_test.txt"), os.path.join(self.path1, "du_zh_test.json"), os.path.join(self.path1, "du_zh_test_knowledge.json"), action = "test")

    def txt_to_json(self, txt_path, json_file, knowledge_file, action = "train"):
        
        chitchat = [ 'Say goodbye','再见' ,  'Greetings','寒暄', "Ask about user's name",'问 User 姓名', "Ask about user's gender",  '问 User 性别', "Ask about user's age", '问 User 年龄',"Ask about user's hobbies", '问 User 爱好',  'Ask about date' ,'问 日期',  'Ask about time','问 时间',  'Ask about weather','问 天气',   'Music on demand','音乐 点播',  'Play music','播放 音乐',    'Weather notification','天气 信息 推送']
        possible_chitchat = ['Chat about stars','关于 明星 的 聊天', 'Food recommendation', '美食 推荐','POI recommendation','兴趣点 推荐', 'Q&A', '问答', 'Ask questions','提问',  'Music recommendation','音乐 推荐']
        rec_list = ['Food recommendation', '美食 推荐','POI recommendation','兴趣点 推荐',  'Music recommendation','音乐 推荐']
        utterance_count = 0
        info_utterance_count = 0
        topic_dict = {}
        file1 = open(txt_path, 'r')
        Lines = file1.readlines()
        dialog_dict = {}
        knowledge_dict = {}
        next_role = "Seeker"
        for idx, line in enumerate(Lines):
            single_dialog = json.loads(line)
            dialog_dict[idx] = {
                "situation": single_dialog["situation"],
                "messages": []
            }
            knowledge_dict[idx] = {
                "user_profile": single_dialog["user_profile"],  #dictionary of user profile like name, job and city
                "movie_kb": single_dialog["knowledge"] #list of knowledge for recommendation
            }
            turn_id = 0
            for  turn in single_dialog["conversation"]:
               
                if action == "train":
                    turn_dict = {
                        "role": next_role, #string, seeker/recommender
                        "message": turn[4:] if turn.startswith("[") else turn,  #string
                        "goal": [single_dialog["goal_type_list"][turn_id]], #list
                        "topic": [single_dialog["goal_topic_list"][turn_id]], #list
                        "knowledge": [],   #the knowledge for the training data is not assigned to each turn, pls refer to the knowledge file for better understanding
                        "mention": [] #item mentioned when goal is recommendation
                    }
                    
                else:
                    turn_dict = {
                        "role": next_role, #string, seeker/recommender
                        "message": turn[4:] if turn.startswith("[") else turn,  #string
                        "goal": [single_dialog["goal_type_list"][turn_id]], #list
                        "topic": [single_dialog["goal_topic_list"][turn_id]], #list
                        "knowledge": single_dialog["knowledge"][turn_id],   #list
                        "mention": [] #item mentioned when goal is recommendation
                    }
                # if turn_dict["goal"] == "Movie recommendation":
                #     breakpoint()
                if ("recommendation" in turn_dict["goal"][0] or "推荐" in turn_dict["goal"][0]) and turn_dict["topic"][0] in turn_dict["message"]:
                    # breakpoint()
                    turn_dict["mention"].append(turn_dict["topic"]) 

                turn_id += 1
                next_role = "Seeker" if next_role == "Recommender" else "Recommender"
                dialog_dict[idx]["messages"].append(turn_dict)


        with open(json_file, 'w') as outfile:
            json.dump(dialog_dict, outfile, indent=4, ensure_ascii=False)

        with open(knowledge_file, 'w') as outfile:
            json.dump(knowledge_dict, outfile, indent=4, ensure_ascii=False)
            

        return dialog_dict

if __name__ == "__main__":

    datafolder_path = "../../DuRecDial/data"
    tgredial = Durecdial(datafolder_path)
