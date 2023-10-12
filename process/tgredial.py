#Date: 16/08/2023
#Author: Victor Li
#Email: lichuang@u.nus.edu



import os
import sys
import json
import pickle
import tqdm
class Tgredial():
    def __init__(self, tgredial_path):
        self.path = tgredial_path

        item_history_file = os.path.join(self.path, "train_identity2history.pkl")
        user_history_file = os.path.join(self.path, "user2TopicSent.pkl")
        # with open(history_file, 'rb') as f:
        #     self.data = pickle.load(f)
        #     breakpoint()
        self.path1 = '../data_p/tgredial'
        if not os.path.exists(self.path1):
            os.makedirs(self.path1)
        train_dialogs = self.pickle_to_json(os.path.join(self.path, "train_data.pkl"), os.path.join(self.path1, "tg_train.json"), item_history_file, user_history_file, os.path.join(self.path1, "tg_train_knowledge.json"))
        valid_dialogs = self.pickle_to_json(os.path.join(self.path, "valid_data.pkl"), os.path.join(self.path1, "tg_valid.json"), item_history_file, user_history_file, os.path.join(self.path1, "tg_valid_knowledge.json"))
        test_dialogs = self.pickle_to_json(os.path.join(self.path, "test_data.pkl"), os.path.join(self.path1, "tg_test.json"), item_history_file, user_history_file, os.path.join(self.path1, "tg_test_knowledge.json"))
        # self.detect_semantic(os.path.join(self.path, "train.json"))




    def pickle_to_json(self, pickle_file, json_file, item_history_file, user_history_file, knowledge_file): 

        chitchat = ['谈论', '反馈', '反馈，结束']
        utt_count = 0
        utterance_count = 0
        info_utterance_count = 0
        with open(pickle_file, 'rb') as f:
            self.data = pickle.load(f)
        with open(item_history_file, 'rb') as f:
            self.item_history = pickle.load(f)
        with open(user_history_file, 'rb') as f:
            self.user_history = pickle.load(f)

            # breakpoint()
        dialog_dict = {}
        knowledge_dict = {}
        missing_knowledge = []
        for idx, dial in enumerate(self.data):

            dialog_dict[idx] = {
                "situation": {"user_id": dial["user_id"], "dial_id": dial["conv_id"]},
                "messages": []
            }
            knowledge_d = []

            knowledge_dict[idx] = {
                "user_profile": list(self.user_history[dial["user_id"]]),     #list of user profile in tages format, like" I enjouy watching movies"
                "item_history": []                                            #historical information of user-item interaction
            }

            # breakpoint()
            for turn in dial["messages"]:
                goals = []
                topics = []
                knowledge = []
                utterance_count += 1
                if turn["local_id"] in dial["goal_path"]:
                    goaltop = dial["goal_path"][turn["local_id"]]
                    for gid, goal in enumerate(goaltop[1:]):
                        if gid%2 == 1:
                            topics.append(goal)
                        elif gid%2 == 0:
                            goals.append(goal)
                            if goal == "推荐电影" and "《" in turn["content"]:
                                item_key = f"{dial['conv_id']}/{turn['local_id']}"
                                if item_key in self.item_history:
                                    items = self.item_history[item_key]
                                    knowledge_d = items.copy()
                                    knowledge = items.copy()
                                else:
                                    missing_knowledge.append(item_key)

                turn_dict = {
                        "role": turn['role'], #string, seeker/recommender
                        "message": turn["content"],  #string
                        "goal": goals, #string
                        "topic": topics, #string
                        "knowledge": knowledge,  #list, user_item_history
                        "mention": dial["mentionMovies"][turn["local_id"]] if turn["local_id"] in dial["mentionMovies"] else [] #list
                    }
                dialog_dict[idx]["messages"].append(turn_dict)
            knowledge_dict[idx]["item_history"] = knowledge_d
        with open(json_file, 'w', encoding='utf-8') as outfile:
            json.dump(dialog_dict, outfile, indent=4, ensure_ascii=False)
        with open(knowledge_file, 'w', encoding='utf-8') as outfile:
            json.dump(knowledge_dict, outfile, indent=4, ensure_ascii=False)

        # print(f"missing knowledge: {missing_knowledge}") 
        # Some knowledge is missing, check here if you want to print them out
        

       

  



if __name__ == "__main__":

    datafolder_path = "../TGReDial"
    tgredial = Tgredial(datafolder_path)