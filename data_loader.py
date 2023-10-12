#date ; 29 sep 2023
#author: Victor Li

import json
import os
import time


def task1(data_j, guidance = 'none'):
    #Name: CRS
    #convert the data into differen task format
    #task one: generate recommendation response
    #input: dialogue history
    #output: response
    data_jl = []
    for dial_id, dialogue in data_j.items():
        dial_history = ''
        for ti, turn in enumerate(dialogue['messages']):
            if ti == 0 and turn["role"] == "Recommender":
                mess = "[system]:" + turn["message"] + "\n"
                dial_history += mess
            elif turn["role"] == 'Seeker':
                mess = "[user]:" + turn["message"] + "\n"
                dial_history += mess
            elif turn["role"] == 'Recommender':
                if guidance == 'none':
                    out_mess = dial_history + "[system]:"
                elif guidance == 'goal':
                    out_mess = dial_history + f"[goal]:{turn['goal']}\n[system]:"
                elif guidance == 'topic':
                    out_mess = dial_history + f"[topic]:{turn['topic']}\n[system]:"
                elif guidance == 'knowledge':
                    out_mess = dial_history + f"[knowledge]:{turn['knowledge']}\n[system]:"
                elif guidance == "goal_topic":
                    out_mess = dial_history + f"[goal]:{turn['goal']} [topic]:{turn['topic']}\n[system]:"
                data_jl.append({'dial/turn_id': f"{dial_id}-{ti}",'task': 'CRS','dial_history': out_mess, 'response': turn["message"]})
                mess = "[system]:" + turn["message"] + "\n"
                dial_history += mess

    return data_jl

def task2(data_j):
    #Name: Chat
    #convert the data into differen task format
    #task two: generate chat response for chitchatting
    #input: dialogue history
    #output: response

    #some example
    CHAT_GOAL_LIST = ['chat', 'chatting', 'chitchat', 'chitchatting', 'talk', 'talking', 'free talk', 'free talking', 'free chat', 'free chatting']
    #for durecdial
    CHAT_GOAL_LIST = [ 'Say goodbye','再见' ,  'Greetings','寒暄', "Ask about user's name",'问 User 姓名', "Ask about user's gender",  '问 User 性别', "Ask about user's age", '问 User 年龄',"Ask about user's hobbies", '问 User 爱好',  'Ask about date' ,'问 日期',  'Ask about time','问 时间',  'Ask about weather','问 天气',   'Music on demand','音乐 点播',  'Play music','播放 音乐',    'Weather notification','天气 信息 推送']
    #for tgredial
    CHAT_GOAL_LIST = ['谈论', '反馈', '反馈，结束']
    data_jl = []
    for dial_id, dialogue in data_j.items():
        dial_history = ''
        for ti, turn in enumerate(dialogue['messages']):
            if ti == 0 and turn["role"] == "Recommender":
                mess = "[system]:" + turn["message"] + "\n"
                dial_history += mess
            elif turn["role"] == 'Seeker':
                mess = "[user]:" + turn["message"] + "\n"
                dial_history += mess
            elif turn["role"] == 'Recommender':
                out_mess = dial_history + "[system]:"
                if turn["goal"] in CHAT_GOAL_LIST:
                    data_jl.append({'dial/turn_id': f"{dial_id}-{ti}",'task': 'CHAT','dial_history': out_mess, 'response': turn["message"]})
                mess = "[system]:" + turn["message"] + "\n"
                dial_history += mess

    return data_jl


def check_recommendation(turn):
    if turn["mention"] == []:
        return False, None
    else:
        return True, None

def task3(data_j, guidance = 'none'):
    #Name: REC
    #convert the data into differen task format
    #task three: generate recommendation response
    #input: dialogue history
    #output: recommendation items

    data_jl = []
    for dial_id, dialogue in data_j.items():
        dial_history = ''
        for ti, turn in enumerate(dialogue['messages']):
            if ti == 0 and turn["role"] == "Recommender":
                mess = "[system]:" + turn["message"] + "\n"
                dial_history += mess
            elif turn["role"] == 'Seeker':
                mess = "[user]:" + turn["message"] + "\n"
                dial_history += mess
            elif turn["role"] == 'Recommender':
                Rec, Item = check_recommendation(turn)
                if Rec:
                    if guidance == 'none':
                        out_mess = dial_history + "[recommendation item]:"
                    elif guidance == 'goal':
                        out_mess = dial_history + f"[goal]:{turn['goal']}\n[recommendation item]:"
                    elif guidance == 'topic':
                        out_mess = dial_history + f"[topic]:{turn['topic']}\n[recommendation item]:"
                    elif guidance == 'knowledge':
                        out_mess = dial_history + f"[knowledge]:{turn['knowledge']}\n[recommendation item]:"
                    elif guidance == "goal_topic":
                        out_mess = dial_history + f"[goal]:{turn['goal']} [topic]:{turn['topic']}\n[recommendation item]:"
                    out_mess = dial_history + "[system]:"
                    if turn["goal"] in CHAT_GOAL_LIST:
                        data_jl.append({'dial/turn_id': f"{dial_id}-{ti}",'task': 'REC','dial_history': out_mess, 'response': turn["message"]})
                mess = "[system]:" + turn["message"] + "\n"
                dial_history += mess

    return data_jl

def task4(data_j):
    #Name: TOPIC
    #convert the data into differen task format
    #task four: generate topic response
    #input: dialogue history
    #output: topic
    data_jl = []
    for dial_id, dialogue in data_j.items():
        dial_history = ''
        for ti, turn in enumerate(dialogue['messages']):
            if ti == 0 and turn["role"] == "Recommender":
                mess = "[system]:" + turn["message"] + "\n"
                dial_history += mess
            elif turn["role"] == 'Seeker':
                mess = "[user]:" + turn["message"] + "\n"
                dial_history += mess
            elif turn["role"] == 'Recommender':
                if turn["topic"] != []:
                    out_mess = dial_history + "[topic]:"
                    data_jl.append({'dial/turn_id': f"{dial_id}-{ti}",'task': 'TOPIC','dial_history': out_mess, 'response': turn["message"]})
                mess = "[system]:" + turn["message"] + "\n"
                dial_history += mess

    return data_jl


def task5(data_j):
    #Name: GOAL
    #convert the data into differen task format
    #task five: generate goal response
    #input: dialogue history
    #output: goal
    data_jl = []
    for dial_id, dialogue in data_j.items():
        dial_history = ''
        for ti, turn in enumerate(dialogue['messages']):
            if ti == 0 and turn["role"] == "Recommender":
                mess = "[system]:" + turn["message"] + "\n"
                dial_history += mess
            elif turn["role"] == 'Seeker':
                mess = "[user]:" + turn["message"] + "\n"
                dial_history += mess
            elif turn["role"] == 'Recommender':
                if turn["goal"] != []:
                    out_mess = dial_history + "[goal]:"
                    data_jl.append({'dial/turn_id': f"{dial_id}-{ti}",'task': 'GOAL','dial_history': out_mess, 'response': turn["message"]})
                mess = "[system]:" + turn["message"] + "\n"
                dial_history += mess

    return data_jl


def knowledge_to_question(knowledge, KG_pool):
    #check if the knowledge is already in KG_pool
    # if not generate proper question to test for the knowledge grounding
    #input: knowledge and KG_pool
    #output: boolean and the question, and answers [all in list format]
    KG = False
    out_mess = "There is no knowledge in this turn"
    result = "NO results"

    return KG, out_mess, result


def task6(data_j):
    #Name: KNOWLEDGE
    #convert the data into differen task format
    #task six: generate knowledge response
    #input: question about the knowledge, node predicction or edge prediction
    #output: entity level node prediction or edge prediction
    data_jl = []
    KG_pool = {}
    for dial_id, dialogue in data_j.items():
        
        for ti, turn in enumerate(dialogue['messages']):
            if turn["knowledge"] != []:
                #check whehter it may contain multiple knowledge
                KG, out_mess, result = knowledge_to_question(turn["knowledge"], KG_pool)

                if KG:
                    data_jl.append({'dial/turn_id': f"{dial_id}-{ti}",'task': 'KNOWLEDGE','dial_history': out_mess, 'response': result})
    return data_jl

def json_to_jsonl(json_file, jsonl_file, purpose = 'task1'):
    with open(json_file, "r") as f:
        data_j = json.load(f)
    if purpose == 'CRS':  #task1
        data_jl = task1(data_j)
    elif purpose == 'CHAT': #task2
        data_jl = task2(data_j)
    elif purpose == 'REC': #task3
        data_jl = task3(data_j)
    elif purpose == 'TOPIC': #task4
        data_jl = task4(data_j)
    elif purpose == 'GOAL': #task5
        data_jl = task5(data_j)
    elif purpose == 'KNOWLEDGE': #task6
        data_jl = task6(data_j)
    else:
        print("You defined task is not in the provided list")
        raise Exception("Sorry, you have to check you task before running thi function")

    with open(jsonl_file, "w") as f:
        for line in data_jl:
            json.dump(line, f)
            f.write('\n')

    return data_jl


def prepare_test_data(task, data_d):
    data_output = []
    # detailed tasks for each dataset
    # need to revised later
    task_dic = {
        "DuRecDial_ENGLISH": ["CRS", "CHAT", "REC", "TOPIC", "GOAL", "KNOWLEDGE"],
        "DuRecDial_CHINESE": ["CRS", "CHAT", "REC", "TOPIC", "GOAL", "KNOWLEDGE"],
        "OpenKnowKG": ["CRS", "CHAT", "REC", "TOPIC", "GOAL", "KNOWLEDGE"],
        "ReDial": ["CRS", "CHAT", "REC", "TOPIC", "GOAL", "KNOWLEDGE"],
        "TG-Redial_CHINESE": ["CRS", "CHAT", "REC", "TOPIC", "GOAL", "KNOWLEDGE"]
    }
    if data_d == "DuRecDial_ENGLISH":
        data_path = os.path.join('data_p/durecdial', 'du_en_test.json')
    elif data_d == "DuRecDial_CHINESE":
        data_path = os.path.join('data_p/durecdial', 'du_zh_test.json')
    elif data_d == "TG-Redial_CHINESE":
        data_path = os.path.join('data_p/tgredial', 'tg_test.json')
    elif data_d == "OpenKnowKG":
        data_path = os.path.join('data_p/opendialkg', 'opendialkg.json')
    elif data_d == "ReDial":
        data_path = os.path.join('data_p/redial', 're_test.json')
    else: 
        print("+++++++++++++++++++++++++++++++++++++++++++++")
        print("You defined data is not in the provided list")
        print("+++++++++++++++++++++++++++++++++++++++++++++")
        raise Exception("Sorry, you have to check you data before running thi function")
    

    avail_task = list(set(task_dic[data_d]) & set(task))
    print("Step 1 for loading data and task:+++++++++++++++++++++++++++++++++")
    print("You proposed dataset is: ", data_d)
    print("You proposed task is: ", task)
    print("The available task for this dataset is: ", avail_task)
    if not os.path.exists('data_test'):
        os.makedirs('data_test')
    for task_i in avail_task:
        out_file = os.path.join('data_test', data_d + '_' + task_i + '.jsonl')
        data_jl = json_to_jsonl(data_path, out_file, task_i)

        data_output.append(data_jl)
    return data_output
        

if __name__ == "__main__":
    #data_d = "DuRecDial_ENGLISH"
    #data_d = "DuRecDial_CHINESE"
    #data_d = "TG-Redial_CHINESE"
    data_d = "OpenKnowKG"
    task = ["CRS", "CHAT", "REC", "TOPIC", "GOAL", "KNOWLEDGE"]
    data_output = prepare_test_data(task, data_d)