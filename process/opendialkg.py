#Date: 16/08/2023
#Author: Victor Li
#Email: lichuang@u.nus.edu

# Modified from origianl code by: ISNPIRED PAPER->code->data-processsing


import csv
import collections
import torch
import numpy as np
import os
import json



class Opendialkg():
    def __init__(self, opendialkg_path):
        self.path = opendialkg_path
        # self.detect_semantic(os.path.join(self.path, "opendialkg.json"))
        self.path1 = '../data_p/opendialkg'
        if not os.path.exists(self.path1):
            os.makedirs(self.path1)
        self.train_dialogs = self.tsv_to_json(os.path.join(self.path, "opendialkg.csv"), os.path.join(self.path1, "opendialkg.json"), os.path.join(self.path1, "opendialkg_knowledge.json"))


    def tsv_to_json(self, tsv_path, json_file, knowledge_file):
        # breakpoint()

        # total_movie_list = []
        # with open("../Inspired/data/movie_database.tsv", "r") as f:
        #     csv_reader = csv.reader(f, delimiter = '\t')
        #     next(csv_reader)
        #     for row in csv_reader:
        #         total_movie_list.append(row[0])

        with open(tsv_path, "r") as file:
            csv_reader = csv.reader(file, delimiter=',')
            dialog_dict = {}
            count = 0
            total_conv = 0
            total_turn = 0
            info_turn = 0
            knowledge_dict = {}

            

            for i, row in enumerate(csv_reader):
                if i == 0:
                    continue
                total_conv += 1
                # breakpoint()
                content = json.loads(row[0])
                dialog_dict[i-1] = {
                "situation": [],  #data and time information
                "messages": []
                }
                knowledge_dict[i-1] = {
                    "user_rating": json.loads(row[1]),  #dictionary of user rating for recommendation
                    "assistant_rating": json.loads(row[2]) #dictionary of assistant rating for recommendation
                }

                turn_id = 0
                for _, item in enumerate(content):
                    # # breakpoint()
                    # if i == 758:
                    #     breakpoint()
                    if "message" in item:
                        total_turn += 1
                        turn_id += 1


                        turn_dict = {
                            "role": 'Recommender' if item["sender"] == "assistant" else "Seeker",
                            "message": item["message"],
                            "goal": item["type"],
                            "topic": [],
                            "knowledge": [],
                            "mention": []
                        }
                        dialog_dict[i-1]["messages"].append(turn_dict)
                    elif "metadata" in item:
                        # print(dialog_dict[i-1]["messages"])
                        if dialog_dict[i-1]["messages"] != []:
                            dialog_dict[i-1]["messages"][-1]["knowledge"].append(item["metadata"])
                            dialog_dict[i-1]["messages"][-1]["topic"].append(item["action_id"])
                            # print(dialog_dict[i-1]["messages"][-1])
                        else: 
                            continue
 

           
        with open(json_file, "w") as file:
            json.dump(dialog_dict, file, indent=4)  
        with open(knowledge_file, "w") as file:
            json.dump(knowledge_dict, file, indent=4)





if __name__ == "__main__":

    datafolder_path = "../opendialkg/data"
    tgredial = Opendialkg(datafolder_path)