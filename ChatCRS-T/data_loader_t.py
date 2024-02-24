#date ; 29 sep 2023
#author: Victor Li
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
import json
import os
import time
import argparse
# from process.Knowledge import Question_for_relation

def task1(input_data, guidance = 'None'):
    #Name: CRS
    #convert the data into differen task format
    #task one: generate recommendation response
    #input: dialogue history
    #output: response
    
    output_dict = {}
    output_data = []
    for dial_id, dialogue in input_data.items():
        dial_history = ''
        for ti, turn in enumerate(dialogue['messages']):
            if ti == 0 and turn["role"] == "Recommender":
                mess = "[system]:" + turn["message"] + "\n"
                # mess = "[推荐系统]:" + turn["message"] + "\n"
                dial_history += mess
            elif turn["role"] == 'Seeker':
                mess = "[user]:" + turn["message"] + "\n"
                # mess = "[用户]:" + turn["message"] + "\n"
                dial_history += mess
            elif turn["role"] == 'Recommender':
                # out_mess = dial_history + "[推荐系统]:"
                out_mess = dial_history + "[system]:"
                if guidance == 'None':
                    output_data.append({'dial-turn_id': f"{dial_id}-{ti}",'task': 'CRS','Input': out_mess, 'Output': turn["message"]})
                    output_dict[f"{dial_id}-{ti}"] = {'dial-turn_id': f"{dial_id}-{ti}",'task': 'CRS','Input': out_mess, 'Output': turn["message"]} 
                else:
                    if guidance == 'GOAL':
                        output_data.append({'dial/turn_id': f"{dial_id}-{ti}",'task': 'CRS', 'guidance': 'GOAL','Input': out_mess, 'guide_message': f"{turn['goal']}", 'Output': turn["message"]})
                        output_dict[f"{dial_id}-{ti}"] = {'dial/turn_id': f"{dial_id}-{ti}",'task': 'CRS', 'guidance': 'GOAL','Input': out_mess, 'guide_message': f"{turn['goal']}", 'Output': turn["message"]}
                        # out_mess = dial_history + f"[goal]:{turn['goal']}\n[system]:"
                    elif guidance == 'TOPIC':
                        output_data.append({'dial/turn_id': f"{dial_id}-{ti}",'task': 'CRS', 'guidance': 'TOPIC','Input': out_mess, 'guide_message': f"{turn['topic']}", 'Output': turn["message"]})
                        output_dict[f"{dial_id}-{ti}"] = {'dial/turn_id': f"{dial_id}-{ti}",'task': 'CRS', 'guidance': 'TOPIC','Input': out_mess, 'guide_message': f"{turn['topic']}", 'Output': turn["message"]}
                        # out_mess = dial_history + f"[topic]:{turn['topic']}\n[system]:"
                    elif guidance == 'KNOWLEDGE':
                        # out_mess = dial_history + f"[knowledge]:{turn['knowledge']}\n[system]:"
                        output_data.append({'dial/turn_id': f"{dial_id}-{ti}",'task': 'CRS', 'guidance': 'KNOWLEDGE','Input': out_mess, 'guide_message': f"{turn['knowledge']}", 'Output': turn["message"]})
                        output_dict[f"{dial_id}-{ti}"] = {'dial/turn_id': f"{dial_id}-{ti}",'task': 'CRS', 'guidance': 'KNOWLEDGE','Input': out_mess, 'guide_message': f"{turn['knowledge']}", 'Output': turn["message"]}                    
                    elif guidance == "REC":
                        output_data.append({'dial/turn_id': f"{dial_id}-{ti}",'task': 'CRS', 'guidance': 'REC','Input': out_mess, 'guide_message': f"{turn['mention']}", 'Output': turn["message"]})
                        output_dict[f"{dial_id}-{ti}"] = {'dial/turn_id': f"{dial_id}-{ti}",'task': 'CRS', 'guidance': 'REC','Input': out_mess, 'guide_message': f"{turn['mention']}", 'Output': turn["message"]}
                    elif guidance == "BOTH":
                        # out_mess = dial_history + f"[goal]:{turn['goal']} [topic]:{turn['topic']}\n[system]:"
                        output_data.append({'dial/turn_id': f"{dial_id}-{ti}",'task': 'CRS', 'guidance': 'BOTH','Input': out_mess, 'guide_message': f"Conversation Goal:{turn['goal']}\n\nKnowledge Triples:{turn['knowledge']}", 'Output': turn["message"]})
                        
                mess = "[system]:" + turn["message"] + "\n"
                # mess = "[推荐系统]:" + turn["message"] + "\n"
                dial_history += mess

    # return  output_dict
    return output_data


def task2(input_data, CHAT_GOAL_LIST ):
    #Name: Chat
    #convert the data into differen task format
    #task two: generate chat response for chitchatting
    #input: dialogue history
    #output: response
    
    #some example
    # CHAT_GOAL_LIST = ['chat', 'chatting', 'chitchat', 'chitchatting', 'talk', 'talking', 'free talk', 'free talking', 'free chat', 'free chatting']
    # #for durecdial
    CHAT_GOAL_LIST = [ 'Say goodbye', '再见' ,  'Greetings','寒暄', "Ask about user's name",'问 User 姓名', "Ask about user's gender",  '问 User 性别', "Ask about user's age", '问 User 年龄',"Ask about user's hobbies", '问 User 爱好',    'Music on demand','音乐 点播',  'Play music','播放 音乐']
    # #for tgredial
    # CHAT_GOAL_LIST = ['谈论', '反馈', '反馈，结束']
    output_data = []
    for dial_id, dialogue in input_data.items():
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
                # breakpoint()
                if len(turn["goal"]) > 1:
                    breakpoint()
                if turn["goal"][0] in CHAT_GOAL_LIST:
                    output_data.append({'dial/turn_id': f"{dial_id}-{ti}",'task': 'CHAT','Input': out_mess, 'Output': turn["message"]})
                mess = "[system]:" + turn["message"] + "\n"
                dial_history += mess

    return output_data


def check_recommendation(turn, dataset):
    if turn["mention"] == []:
        return False, None
    else:
        if "DuRecDial" in dataset:
            if len(turn["mention"]) > 1:
                raise Exception("there are more than 1 recommendation in DuRecDial Conversation")
            else:
                return True, turn["mention"][0]
        elif "TG-Redial" in dataset:
            if len(turn["mention"]) > 2:
                raise Exception("there are more than 1 recommendation in TG-Redial Conversation")
            elif not "推荐电影" in turn["goal"]:
                print(turn["goal"])
                raise Exception("the goal is not recommendation in TG-Redial Conversation")
            else:
                return True, turn["mention"][1].split("(")[0]


def task3(input_data,  dataset, guidance = 'None'):
    #Name: REC
    #convert the data into differen task format
    #task three: generate recommendation response
    #input: dialogue history
    #output: recommendation items

    output_data = []
    for dial_id, dialogue in input_data.items():
        dial_history = ''
        for ti, turn in enumerate(dialogue['messages']):
            if ti == 0 and turn["role"] == "Recommender":
                mess = "[system]:" + turn["message"] + "\n"
                dial_history += mess
            elif turn["role"] == 'Seeker':
                mess = "[user]:" + turn["message"] + "\n"
                dial_history += mess
            elif turn["role"] == 'Recommender':
                Rec, Item = check_recommendation(turn, dataset)
                if Rec:
                    out_mess = dial_history
                    if guidance == 'None':
                        output_data.append({'dial-turn_id': f"{dial_id}-{ti}",'task': 'REC','Input': out_mess, 'Output': Item})
                    elif guidance == 'GOAL':
                        output_data.append({'dial/turn_id': f"{dial_id}-{ti}",'task': 'REC', 'guidance': 'GOAL','Input': out_mess, 'guide_message': f"[goal]:{turn['goal']}", 'Output': Item})
                    elif guidance == 'TOPIC':
                        output_data.append({'dial/turn_id': f"{dial_id}-{ti}",'task': 'REC', 'guidance': 'TOPIC','Input': out_mess, 'guide_message': turn['topic'], 'Output': Item})
                    elif guidance == 'KNOWLEDGE':
                        output_data.append({'dial/turn_id': f"{dial_id}-{ti}",'task': 'REC', 'guidance': 'KNOWLEDGE','Input': out_mess, 'guide_message': f"[knowledge]:{turn['knowledge']}", 'Output': Item})  
                    
                mess = "[system]:" + turn["message"] + "\n"
                dial_history += mess

    return output_data

def task4(input_data):
    #Name: TOPIC
    #convert the data into differen task format
    #task four: generate topic response
    #input: dialogue history
    #output: topic
    output_data = []
    for dial_id, dialogue in input_data.items():
        dial_history = ''
        for ti, turn in enumerate(dialogue['messages']):
            if ti == 0 and turn["role"] == "Recommender":
                mess = "[system]:" + turn["message"] + "\n"
                dial_history += mess
            elif turn["role"] == 'Seeker':
                mess = "[user]:" + turn["message"] + "\n"
                dial_history += mess
            elif turn["role"] == 'Recommender':
                if turn["topic"] != [] and turn["topic"] != ["None"]:
                    out_mess = dial_history
                    output_data.append({'dial/turn_id': f"{dial_id}-{ti}",'task': 'TOPIC','Input': out_mess, 'Output': turn["topic"]})
                mess = "[system]:" + turn["message"] + "\n"
                dial_history += mess

    return output_data


def task5(input_data):
    #Name: GOAL
    #convert the data into differen task format
    #task five: generate goal response
    #input: dialogue history
    #output: goal
    output_data = []
    for dial_id, dialogue in input_data.items():
        dial_history = ''
        for ti, turn in enumerate(dialogue['messages']):
            if ti == 0 and turn["role"] == "Recommender":
                # mess = "[推荐系统]:" + turn["message"] + "\n"
                mess = "[system]:" + turn["message"] + "\n"
                dial_history += mess
            elif turn["role"] == 'Seeker':
                mess = "[user]:" + turn["message"] + "\n"
                dial_history += mess
            elif turn["role"] == 'Recommender':
                if turn["goal"] != []:
                    out_mess = dial_history
                    if turn["goal"] == [ "谈论", "请求推荐"]:
                        goal1 = "谈论并请求推荐"
                    elif turn["goal"] == ["请求推荐", "推荐电影"]:
                        goal1 = "请求推荐并推荐电影"
                    else:
                        goal1 = turn["goal"][0]
                    output_data.append({'dial/turn_id': f"{dial_id}-{ti}",'task': 'GOAL','Input': out_mess, 'Output': goal1})
                # mess = "[推荐系统]:" + turn["message"] + "\n"
                mess = "[system]:" + turn["message"] + "\n"
                dial_history += mess

    return output_data



def task6(input_data):
    #Name: KNOWLEDGE
    #convert the data into differen task format
    #task six: generate knowledge response
    #input: question about the knowledge, node predicction or edge prediction
    #output: entity level node prediction or edge prediction
    output_data = []
    KG_pool = {}
    unique_knowledge = []
    for dial_id, dialogue in input_data.items():
        
        for ti, turn in enumerate(dialogue['messages']):
            if turn["knowledge"] != []:
                if turn["knowledge"] not in unique_knowledge:
                    unique_knowledge.append(turn["knowledge"])
                else:
                    continue
                #check whehter it may contain multiple knowledge
                #get knowledge triple from the data
                A = turn["knowledge"][0]
                R = turn["knowledge"][1]
                B = turn["knowledge"][2]
                # Question = Question_for_relation(A, R)
                Question = f"{A} {R} ?"
                output_data.append({'dial/turn_id': f"{dial_id}-{ti}",'task': 'KNOWLEDGE','Input': Question, 'Output': B})
    return output_data

def read_data(input_file, output_file, args):
    guidance_dic = {
        "DuRecDial_ENGLISH": ["REC", "TOPIC", "GOAL", "KNOWLEDGE", "BOTH"],
        "DuRecDial_CHINESE": ["REC", "TOPIC", "GOAL", "KNOWLEDGE", "ALL"],
        "TG-Redial_CHINESE": ["REC", "TOPIC", "GOAL", "ALL"]
    }
    chat_dic = {
        "DuRecDial_ENGLISH": ['chat', 'chatting', 'chitchat', 'chitchatting', 'talk', 'talking', 'free talk', 'free talking', 'free chat', 'free chatting'],
        "TG-Redial_CHINESE": [ 'Say goodbye','再见' ,  'Greetings','寒暄', "Ask about user's name",'问 User 姓名', "Ask about user's gender",  '问 User 性别', "Ask about user's age", '问 User 年龄',"Ask about user's hobbies", '问 User 爱好',  'Ask about date' ,'问 日期',  'Ask about time','问 时间',  'Ask about weather','问 天气',   'Music on demand','音乐 点播',  'Play music','播放 音乐',    'Weather notification','天气 信息 推送']
    }
    task_name = args.task
    with open(input_file, "r") as f:
        input_data = json.load(f)
    if task_name == 'CRS':  #task1
        if args.with_guidance and args.guidance in guidance_dic[args.dataset_name]:
            output_data = task1(input_data, args.guidance)
        else:
            output_data = task1(input_data)
    elif task_name == 'CHAT': #task2
        output_data = task2(input_data, chat_dic[args.dataset_name])
    elif task_name == 'REC': #task3
        if args.with_guidance and args.guidance in guidance_dic[args.dataset_name] and args.guidance != 'ALL' and args.guidance != 'REC':
            output_data = task3(input_data, args.dataset_name, args.guidance )
        else:
            output_data = task3(input_data, args.dataset_name)
    elif task_name == 'TOPIC': #task4
        output_data = task4(input_data)
    elif task_name == 'GOAL': #task5
        output_data = task5(input_data)
    elif task_name == 'KNOWLEDGE': #task6
        output_data = task6(input_data)
    else:
        print("You defined task is not in the provided list")
        raise Exception("Sorry, you have to check you task before running thi function")

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    return output_data


def prepare_train_data(args):
    data_output = []
    # detailed tasks for each dataset
    # need to revised later
    task_dic = {
        "DuRecDial_ENGLISH": ["CRS", "CHAT", "REC", "TOPIC", "GOAL", "KNOWLEDGE"],
        "DuRecDial_CHINESE": ["CRS", "CHAT", "REC", "TOPIC", "GOAL", "KNOWLEDGE"],
        "TG-Redial_CHINESE": ["CRS", "CHAT", "REC", "TOPIC", "GOAL"]
    }
    logger.info("Dataset is not previously save, creating data for evaluation...") 
        # "OpenKnowKG": ["CRS", "CHAT", "REC", "TOPIC", "GOAL", "KNOWLEDGE"],
        # "ReDial": ["CRS", "CHAT", "REC", "TOPIC", "GOAL", "KNOWLEDGE"],
    if args.dataset_name == "DuRecDial_ENGLISH":
        data_path = os.path.join('data_p/durecdial', 'du_en_train.json')
    elif args.dataset_name == "DuRecDial_CHINESE":
        data_path = os.path.join('data_p/durecdial', 'du_zh_train.json')
    elif args.dataset_name == "TG-Redial_CHINESE":
        data_path = os.path.join('data_p/tgredial', 'tg_train.json')
    # elif args.dataset_name == "OpenKnowKG":
    #     data_path = os.path.join('data_p/opendialkg', 'opendialkg.json')
    # elif args.dataset_name == "ReDial":
    #     data_path = os.path.join('data_p/redial', 're_test.json')
    else: 
        
        print("+++++++++++++++++++++++++++++++++++++++++++++")
        print("You defined data is not in the provided list")
        print("+++++++++++++++++++++++++++++++++++++++++++++")
        raise Exception("Sorry, you have to check you data before running this function")
    

    avail_task = task_dic[args.dataset_name]
    print("Step 1 for loading data and task:+++++++++++++++++++++++++++++++++")
    print("You proposed dataset is: ", args.dataset_name)
    print("The available task for this dataset is: ", avail_task)
    print("You proposed task is: ", args.task)
    if args.task not in avail_task:
        raise Exception("Sorry, you have to check you task in the provided task list")
    if not os.path.exists('data_train_T'):
        os.makedirs('data_train_T')
    if (args.task == 'CRS' or args.task == 'REC') and args.with_guidance:
        out_path = os.path.join('data_train_T', args.dataset_name + '_' + args.task + '_' + args.guidance + '.json') 
    else:  
        out_path = os.path.join('data_train_T', args.dataset_name + '_' + args.task + '.json')
    eval_data = read_data(data_path, out_path, args)

    return eval_data


def prepare_test_data(args):
    data_output = []
    # detailed tasks for each dataset
    # need to revised later
    task_dic = {
        "DuRecDial_ENGLISH": ["CRS", "CHAT", "REC", "TOPIC", "GOAL", "KNOWLEDGE"],
        "DuRecDial_CHINESE": ["CRS", "CHAT", "REC", "TOPIC", "GOAL", "KNOWLEDGE"],
        "TG-Redial_CHINESE": ["CRS", "CHAT", "REC", "TOPIC", "GOAL"]
    }
    logger.info("Dataset is not previously save, creating data for evaluation...") 
        # "OpenKnowKG": ["CRS", "CHAT", "REC", "TOPIC", "GOAL", "KNOWLEDGE"],
        # "ReDial": ["CRS", "CHAT", "REC", "TOPIC", "GOAL", "KNOWLEDGE"],
    if args.dataset_name == "DuRecDial_ENGLISH":
        data_path = os.path.join('data_p/durecdial', 'du_en_test.json')
    elif args.dataset_name == "DuRecDial_CHINESE":
        data_path = os.path.join('data_p/durecdial', 'du_zh_test.json')
    elif args.dataset_name == "TG-Redial_CHINESE":
        data_path = os.path.join('data_p/tgredial', 'tg_test.json')
    # elif args.dataset_name == "OpenKnowKG":
    #     data_path = os.path.join('data_p/opendialkg', 'opendialkg.json')
    # elif args.dataset_name == "ReDial":
    #     data_path = os.path.join('data_p/redial', 're_test.json')
    else: 
        
        print("+++++++++++++++++++++++++++++++++++++++++++++")
        print("You defined data is not in the provided list")
        print("+++++++++++++++++++++++++++++++++++++++++++++")
        raise Exception("Sorry, you have to check you data before running this function")
    


    avail_task = task_dic[args.dataset_name]
    print("Step 1 for loading data and task:+++++++++++++++++++++++++++++++++")
    print("You proposed dataset is: ", args.dataset_name)
    print("The available task for this dataset is: ", avail_task)
    print("You proposed task is: ", args.task)
    if args.task not in avail_task:
        raise Exception("Sorry, you have to check you task in the provided task list")
    if not os.path.exists('data_test_T'):
        os.makedirs('data_test_T')
    if (args.task == 'CRS' or args.task == 'REC') and args.with_guidance:
        out_path = os.path.join('data_test_T', args.dataset_name + '_' + args.task + '_' + args.guidance + '.json') 
    else:  
        out_path = os.path.join('data_test_T', args.dataset_name + '_' + args.task + '.json')
    eval_data = read_data(data_path, out_path, args)

    return eval_data

def prepare_test_data_dict(args):
    data_output = []
    # detailed tasks for each dataset
    # need to revised later
    task_dic = {
        "DuRecDial_ENGLISH": ["CRS", "CHAT", "REC", "TOPIC", "GOAL", "KNOWLEDGE"],
        "DuRecDial_CHINESE": ["CRS", "CHAT", "REC", "TOPIC", "GOAL", "KNOWLEDGE"],
        "TG-Redial_CHINESE": ["CRS", "CHAT", "REC", "TOPIC", "GOAL"]
    }
    logger.info("Dataset is not previously save, creating data for evaluation...") 
        # "OpenKnowKG": ["CRS", "CHAT", "REC", "TOPIC", "GOAL", "KNOWLEDGE"],
        # "ReDial": ["CRS", "CHAT", "REC", "TOPIC", "GOAL", "KNOWLEDGE"],
    if args.dataset_name == "DuRecDial_ENGLISH":
        data_path = os.path.join('data_p/durecdial', 'du_en_test.json')
    elif args.dataset_name == "DuRecDial_CHINESE":
        data_path = os.path.join('data_p/durecdial', 'du_zh_test.json')
    elif args.dataset_name == "TG-Redial_CHINESE":
        data_path = os.path.join('data_p/tgredial', 'tg_test.json')
    # elif args.dataset_name == "OpenKnowKG":
    #     data_path = os.path.join('data_p/opendialkg', 'opendialkg.json')
    # elif args.dataset_name == "ReDial":
    #     data_path = os.path.join('data_p/redial', 're_test.json')
    else: 
        
        print("+++++++++++++++++++++++++++++++++++++++++++++")
        print("You defined data is not in the provided list")
        print("+++++++++++++++++++++++++++++++++++++++++++++")
        raise Exception("Sorry, you have to check you data before running this function")
    


    avail_task = task_dic[args.dataset_name]
    print("Step 1 for loading data and task:+++++++++++++++++++++++++++++++++")
    print("You proposed dataset is: ", args.dataset_name)
    print("The available task for this dataset is: ", avail_task)
    print("You proposed task is: ", args.task)
    if args.task not in avail_task:
        raise Exception("Sorry, you have to check you task in the provided task list")
    if not os.path.exists('data_dict'):
        os.makedirs('data_dict')
    if (args.task == 'CRS' or args.task == 'REC') and args.with_guidance:
        out_path = os.path.join('data_dict', args.dataset_name + '_' + args.task + '_' + args.guidance + '.json') 
    else:  
        out_path = os.path.join('data_dict', args.dataset_name + '_' + args.task + '.json')
    eval_data = read_data(data_path, out_path, args)

    return eval_data

def prepare_dev_data(args):
    data_output = []
    # detailed tasks for each dataset
    # need to revised later
    task_dic = {
        "DuRecDial_ENGLISH": ["CRS", "CHAT", "REC", "TOPIC", "GOAL", "KNOWLEDGE"],
        "DuRecDial_CHINESE": ["CRS", "CHAT", "REC", "TOPIC", "GOAL", "KNOWLEDGE"],
        "TG-Redial_CHINESE": ["CRS", "CHAT", "REC", "TOPIC", "GOAL"]
    }
    logger.info("Dataset is not previously save, creating data for evaluation...") 
        # "OpenKnowKG": ["CRS", "CHAT", "REC", "TOPIC", "GOAL", "KNOWLEDGE"],
        # "ReDial": ["CRS", "CHAT", "REC", "TOPIC", "GOAL", "KNOWLEDGE"],
    if args.dataset_name == "DuRecDial_ENGLISH":
        data_path = os.path.join('data_p/durecdial', 'du_en_dev.json')
    elif args.dataset_name == "DuRecDial_CHINESE":
        data_path = os.path.join('data_p/durecdial', 'du_zh_dev.json')
    elif args.dataset_name == "TG-Redial_CHINESE":
        data_path = os.path.join('data_p/tgredial', 'tg_valid.json')
    # elif args.dataset_name == "OpenKnowKG":
    #     data_path = os.path.join('data_p/opendialkg', 'opendialkg.json')
    # elif args.dataset_name == "ReDial":
    #     data_path = os.path.join('data_p/redial', 're_dev.json')
    else: 
        
        print("+++++++++++++++++++++++++++++++++++++++++++++")
        print("You defined data is not in the provided list")
        print("+++++++++++++++++++++++++++++++++++++++++++++")
        raise Exception("Sorry, you have to check you data before running this function")
    


    avail_task = task_dic[args.dataset_name]
    print("Step 1 for loading data and task:+++++++++++++++++++++++++++++++++")
    print("You proposed dataset is: ", args.dataset_name)
    print("The available task for this dataset is: ", avail_task)
    print("You proposed task is: ", args.task)
    if args.task not in avail_task:
        raise Exception("Sorry, you have to check you task in the provided task list")
    if not os.path.exists('data_dev_T'):
        os.makedirs('data_dev_T')
    if (args.task == 'CRS' or args.task == 'REC') and args.with_guidance:
        out_path = os.path.join('data_dev_T', args.dataset_name + '_' + args.task + '_' + args.guidance + '.json') 
    else:  
        out_path = os.path.join('data_dev_T', args.dataset_name + '_' + args.task + '.json')
    eval_data = read_data(data_path, out_path, args)

    return eval_data


def prepare_demo_data(args):  
    data_output = []
    # detailed tasks for each dataset
    # need to revised later
    task_dic = {
        "DuRecDial_ENGLISH": ["CRS", "CHAT", "REC", "TOPIC", "GOAL"],
        "DuRecDial_CHINESE": ["CRS", "CHAT", "REC", "TOPIC", "GOAL"],
        "TG-Redial_CHINESE": ["CRS", "CHAT", "REC", "TOPIC", "GOAL"]
    }
    logger.info("Dataset is not previously save, creating data for evaluation...") 
        # "OpenKnowKG": ["CRS", "CHAT", "REC", "TOPIC", "GOAL", "KNOWLEDGE"],
        # "ReDial": ["CRS", "CHAT", "REC", "TOPIC", "GOAL", "KNOWLEDGE"],
    if args.dataset_name == "DuRecDial_ENGLISH":
        data_path = os.path.join('instructions', 'Du_en_demo.json')
    elif args.dataset_name == "DuRecDial_CHINESE":
        data_path = os.path.join('data_p/durecdial', 'du_zh_test.json')
    elif args.dataset_name == "TG-Redial_CHINESE":
        data_path = os.path.join('data_p/tgredial', 'tg_test.json')
    # elif args.dataset_name == "OpenKnowKG":
    #     data_path = os.path.join('data_p/opendialkg', 'opendialkg.json')
    # elif args.dataset_name == "ReDial":
    #     data_path = os.path.join('data_p/redial', 're_test.json')
    else: 
        
        print("+++++++++++++++++++++++++++++++++++++++++++++")
        print("You defined data is not in the provided list")
        print("+++++++++++++++++++++++++++++++++++++++++++++")
        raise Exception("Sorry, you have to check you data before running this function")
    

    avail_task = task_dic[args.dataset_name]
    print("Step 1 for loading data and task:+++++++++++++++++++++++++++++++++")
    print("You proposed dataset is: ", args.dataset_name)
    print("The available task for this dataset is: ", avail_task)
    print("You proposed task is: ", args.task)
    if args.task not in avail_task:
        raise Exception("Sorry, you have to check you task in the provided task list")
    if not os.path.exists('data_test'):
        os.makedirs('data_test')
    if (args.task == 'CRS' or args.task == 'REC') and args.with_guidance:
        out_path = os.path.join('instructions/demo', args.dataset_name + '_' + args.task + '_' + args.guidance + '.json') 
    else:  
        out_path = os.path.join('instructions/demo', args.dataset_name + '_' + args.task + '.json')
    eval_data = read_data(data_path, out_path, args)

    return eval_data    

if __name__ == "__main__":
    #data_d = "DuRecDial_ENGLISH"
    #data_d = "DuRecDial_CHINESE"
    #data_d = "TG-Redial_CHINESE"
    # data_d = "OpenKnowKG"
    # task = ["CRS", "CHAT", "REC", "TOPIC", "GOAL", "KNOWLEDGE"]
    # data_output = prepare_test_data(task, data_d)
    task_dic = {
        "DuRecDial_ENGLISH": ["CRS", "CHAT", "REC", "TOPIC", "GOAL", "KNOWLEDGE"],
        "DuRecDial_CHINESE": ["CRS", "CHAT", "REC", "TOPIC", "GOAL", "KNOWLEDGE"],
        "TG-Redial_CHINESE": ["CRS", "CHAT", "REC", "TOPIC", "GOAL"]
    }
    from config import get_args
    args = get_args()
    args.dataset_name = "TG-Redial_CHINESE"
    

    # for task in task_dic[args.dataset_name]:
    #     args.task = task
    #     # data_output = prepare_test_data(args)
    #     # data_output = prepare_train_data(args)
    #     data_output = prepare_dev_data(args)
    ##For testing data with guidance
    args.task = "GOAL"
    # args.task = "REC"
    args.with_guidance = True
    args.guidance = 'TOPIC'
    data_output = prepare_train_data(args)
    data_output = prepare_dev_data(args)
    data_output = prepare_test_data(args)
    # for gui in task_dic[args.dataset_name][3:]:
    #     args.guidance = gui
        # data_output = prepare_test_data_dict(args)
