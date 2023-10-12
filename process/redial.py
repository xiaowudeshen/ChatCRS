#Date: 16/08/2023
#Author: Victor Li
#Email: lichuang@u.nus.edu


'''
This file is used to process the ReDial dataset into the standard format of MultiWoz dataset

'''


import os
import sys
import json
import re



class Redial():
    def __init__(self, redial_path):
        self.path = redial_path
        # self.detect_semantic(os.path.join(self.path, "train.json"))
        self.path1 = '../data_p/redial'
        if not os.path.exists(self.path1):
            os.makedirs(self.path1)
        train_dialogs = self.load_jsonl(os.path.join(self.path, "train_data.jsonl"), os.path.join(self.path1, "re_train.json"), os.path.join(self.path1, "re_train_knowledge.json"))
        test_dialogs = self.load_jsonl(os.path.join(self.path, "test_data.jsonl"), os.path.join(self.path1, "re_test.json"), os.path.join(self.path1, "re_test_knowledge.json"))
    

    def load_jsonl(self, data_path, data_out_path, knowledge_out_path):
    #create json file from jsonl files
    # return dialog list with each conversation as a dictionary
        utterance_count = 0
        info_utt = 0
        self.movie_count = {}
        with open(data_path, 'r') as json_file:
            json_list = list(json_file)
        dialog_dict = {}
        knowledge_dict = {}
        missing_knowledge = []
        for idx, json_str in enumerate(json_list):
            result = json.loads(json_str)
            dialog_dict[idx] = {
                "situation": result['conversationId'],  #conversation id
                "messages": []
            }
            item_candidates = result["movieMentions"]
            
            knowledge_dict[idx] = {
                "respondentWorkerID": result["respondentWorkerId"],  #user id
                "initiatorWorkerID": result["initiatorWorkerId"],    #agent id
                "respondentQuestions": result["respondentQuestions"],  #list of questions asked by user, marked by each item
                "initiatorQuestions": result["initiatorQuestions"],  #list of questions asked by agent, marked by each item
                "movieMentions": result["movieMentions"] #list of movie mentioned in the conversation
            }
            turn_id = 0
            message = {"local_id": "", "role": "", "content": ""}
            # dial['dialogue_id'] = result['conversationId']
            # dial['messages'] = {}
            initiator = result["initiatorWorkerId"]
            next_role = initiator
            pattern = r'@([^ ]+)'

            # breakpoint()
            for turn in result['messages']:
                
                if turn['senderWorkerId'] == next_role:
                    
                    turn_dict = {
                    "role": 'seeker' if turn['senderWorkerId'] == initiator else 'recommender', #string, seeker/recommender
                    "message": turn['text'],  #string
                    "goal": [], #Not applicable in ReDial
                    "topic": [], #Not applicable in ReDial
                    "knowledge": [],  #Not applicable in ReDial
                    "mention": [] #list of movie mentioned in the conversation turn
                    }
                    mentions = re.findall(pattern, turn['text'])
                    for movid_id in mentions:

                        

                        m_id =  re.sub('\D', '', movid_id)
                        if m_id == "":
                            continue
                        if m_id not in item_candidates:
                            missing_knowledge.append('conv_idx:'+ str(idx)+ 'turn_idx:'+str(turn_id)+ 'mention:'+ m_id)
                            continue
                        turn_dict["mention"].append(item_candidates[m_id])
                        turn_dict["message"] = turn_dict["message"].replace('@'+m_id, item_candidates[m_id])


                
                    turn_id += 1
                    next_role = result['respondentWorkerId'] if turn['senderWorkerId'] == initiator else result['initiatorWorkerId']
                
                else: 
                    turn_dict["message"] += ' ' + turn["text"]
                    mentions = re.findall(pattern, turn['text'])
                    for movid_id in mentions:
                        m_id =  re.sub('\D', '', movid_id)
                        if m_id == "":
                            continue
                        if m_id not in item_candidates:
                            missing_knowledge.append('conv_idx', idx, 'turn_idx', turn_id, 'mention', m_id)
                            continue
                        turn_dict["mention"].append(item_candidates[m_id])
                        turn_dict["message"] = turn_dict["message"].replace('@'+m_id, item_candidates[m_id])
                dialog_dict[idx]['messages'].append(turn_dict)
            
            


        with open(data_out_path, "w") as write_file:
            json.dump(dialog_dict, write_file, indent=4)
        with open(knowledge_out_path, "w") as write_file:
            json.dump(knowledge_dict, write_file, indent=4)


        return dialog_dict


if __name__ == "__main__":
    datafolder_path = "../redial"
    redial = Redial(datafolder_path)


