# Data: Oct, 2023
# Author: Victor Li
# This file is used to save the scripts for analysis
# including the followings:
# - dataset analysis



import json
import os




def Get_analysis_for_datasets():
    dataset_dic = {
        "durecdial": ["du_en_dev_knowledge.json", "du_en_test_knowledge.json",  "du_zh_dev_knowledge.json", "du_zh_test_knowledge.json"],

    }
    unique_relation = []
    unique_dic = {
        'weather': [],
        'food': []
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
                    "KG_type": [],
                    "KG_type_dict": {},
                    "perfect":[],
                    "weather": [],
                    "weather_dict": {}
                }
                for _, dialogue in data.items():
                    goal_flag = False
                    topic_flag = False
                    kg_flag = False
                    # breakpoint()
                    for triple in dialogue["movie_kb"]:
                    
                        if triple == []:
                            continue
                        if 'zh' in file:
                            triple = triple[0]
                        if len(triple) != 3:
                            raise Exception("The triple is not in the correct format (A-R-B)")
                        R = triple[1]
                        A = triple[0]
                        B = triple[2]
                        
                        if '-' in R and '2018' in R:
                            if triple not in unique_dic['weather']:
                                unique_dic['weather'].append(triple)
                                result_dic[dataset + "_" + file]["weather"].append(R)
                            # if R not in result_dic[dataset + "_" + file]["weather_dict"]:
                                # result_dic[dataset + "_" + file]["weather_dict"][R] = 1
                            # else:
                                # result_dic[dataset + "_" + file]["weather_dict"][R] += 1
                        # elif R == "Type":
                        #     result_dic[dataset + "_" + file]["perfect"].append(triple)
                        else:
                            if R not in unique_relation:
                                unique_relation.append(R)
                            if R not in result_dic[dataset + "_" + file]["KG_type_dict"]:
                                result_dic[dataset + "_" + file]["KG_type_dict"][R] = 1
                                result_dic[dataset + "_" + file]["KG_type"].append(R)
                            else: 
                                result_dic[dataset + "_" + file]["KG_type_dict"][R] += 1
    
    print(unique_relation)
    out_file = os.path.join("../data_p", "knowledge_analysis.json")
    with open(out_file, "w") as f:
        json.dump(result_dic, f, indent=4)

    out_file = os.path.join("../data_p", "Unique_knowledge_dic.json")
    with open(out_file, "w") as f:
        json.dump(unique_dic, f, indent=4)
    return result_dic


                            
def Question_for_relation(A, R):
    # description_dic = {}
    # total_list = []
    # relation = ['Perfect for having', 'Type', 'Specials', 'Price per person', 'Rating', 'Address', 'Birthday', 'Achievement', 'Intro', 'Comments', 'Height', 'Prefers', 'Sings', 'Star sign', 'Awards', 'Time', 'Stars', 'Reputation', 'Ingredients', 'Date', 'Director', 'Blood Type', 'Likes', 'Perfect for listening to', 'Number of orders', 'Birthplace', 'Country', 'Weight', 'Chinese zodiac']
    # zh_relation = [ '适合吃', '类型', '特色菜', '人均价格', '评分', '地址', '生日', '成就', '简介', '演唱', '身高', '喜好', '评论', '主演', '星座', '时间', '获奖', '口碑', '成分', '日期', '导演', '血型', '喜欢', '适合听', '出生地', '订单量', 'Town店） 特色菜', 'Town店） 人均价格', '体重', '国家地区', '属相', '道道道 口碑', '30 40 评分', '30 40 导演', '嘿玛 类型']
    # for R in relation:
        # Generate question for each relation in the relation list
    if '-' in R and '2018' in R:
        Question = f'What is the weather in City "{A}" on the day of "{R}"" ?' 
    elif R == "Perfect for having":
        Question = f'What meal is good to have for the weather of "{A}"?'    
    elif R == "Type":
        Question = f'What is the type or category of "{A}"?'
    elif R == "Specials":
        Question = f'What is the special food for the restaurant named "{A}"?'
    elif R == "Price per person":
        Question = f'What is the price per person for "{A}"?'
    elif R == "Rating":
        Question = f'What is the rating for "{A}"?'
    elif R == "Address":
        Question = f'What is the address of "{A}"?'
    elif R == "Birthday":
        Question = f'What is the birthday of "{A}"?'
    elif R == "Achievement":
        Question = f'What is the achievement of the person named "{A}"?'
    elif R == "Intro":
        Question = f'What is the introduction of the person named"{A}"?'
    elif R == "Comments":
        Question = f'What are the comments for the movie named "{A}"?'
    elif R == "Height":
        Question = f'What is the height of the person named "{A}"?'
    elif R == "Prefers":
        Question = f'What is the preference of the person named "{A}"?'
    elif R == "Sings":
        Question = f'What is the song that the person named "{A}" sings?'
    elif R == "Star sign":
        Question = f'What is the star sign of the person named "{A}"?'
    elif R == "Awards":
        Question = f'What is the award of the person or movie named "{A}"?'
    elif R == "Time":
        Question = f'When is chatting time?'
    elif R == "Stars":
        Question = f'What movie did "{A}" stars in?'
    elif R == "Reputation":
        Question = f'What is the reputation of the person named "{A}"?'
    elif R == "Ingredients":
        Question = f'What are the ingredients for the food named "{A}"?'
    elif R == "Date":
        Question = f'What is the date for the event named "{A}"?'
    elif R == "Director":
        Question = f'Who is the director for the movie named "{A}"?'
    elif R == "Blood Type":
        Question = f'What is the blood type of the person named "{A}"?'
    elif R == "Likes":                  
        Question = f'What does the person named "{A}" like?'
    elif R == "Perfect for listening to":
        Question = f'What music is good to listen to for the weather of "{A}"?'
    elif R == "Number of orders":
        Question = f'What is the number of orders for the food named "{A}"?'
    elif R == "Birthplace":
        Question = f'What is the birthplace of the person named "{A}"?'
    elif R == "Country":
        Question = f'What is the country of the person named "{A}"?'
    elif R == "Weight":
        Question = f'What is the weight of the person named "{A}"?'
    elif R == "Chinese zodiac":
        Question = f'What is the Chinese zodiac of the person named "{A}"?'


    return Question


def Question_for_zh_relation(A, R):
    # description_dic = {}
    # total_list = []
    # relation = ['Perfect for having', 'Type', 'Specials', 'Price per person', 'Rating', 'Address', 'Birthday', 'Achievement', 'Intro', 'Comments', 'Height', 'Prefers', 'Sings', 'Star sign', 'Awards', 'Time', 'Stars', 'Reputation', 'Ingredients', 'Date', 'Director', 'Blood Type', 'Likes', 'Perfect for listening to', 'Number of orders', 'Birthplace', 'Country', 'Weight', 'Chinese zodiac']
    zh_relation = [ '适合吃', '类型', '特色菜', '人均价格', '评分', '地址', '生日', '成就', '简介', '演唱', '身高', '喜好', '评论', '主演', '星座', '时间', '获奖', '口碑', '成分', '日期', '导演', '血型', '喜欢', '适合听', '出生地', '订单量', 'Town店） 特色菜', 'Town店） 人均价格', '体重', '国家地区', '属相', '道道道 口碑', '30 40 评分', '30 40 导演', '嘿玛 类型']
    # for R in relation:
        # Generate question for each relation in the relation list
    if '-' in R and '2018' in R:
        Question = f'“{A}”城市在“{R}”这一天的天气是什么？' 
    elif R == '适合吃':
        questions[R] = '“{A}”这个天气适合吃什么食物？'
    elif R == '类型':
        questions[R] = '“{A}”的类型是什么？'
    elif R == '特色菜':
        questions[R] = '“{A}”这家店有什么特色菜？'
    elif R == '人均价格':
        questions[R] = '“{A}”的人均价格是多少？'
    elif R == '评分':
        questions[R] = '“{A}”的评分是多少？'
    elif R == '地址':
        questions[R] = '“{A}”的地址在哪里？'
    elif R == '生日':
        questions[R] = '“{A}”的生日是什么时候？'
    elif R == '成就':
        questions[R] = '“{A}”有哪些成就？'
    elif R == '简介':
        questions[R] = '能给我介绍一下“{A}”吗？'
    elif R == '演唱':
        questions[R] = '“{A}”演唱了哪些歌曲？'
    elif R == '身高':
        questions[R] = '“{A}”的身高是多少？'
    elif R == '喜好':
        questions[R] = '“{A}”有什么特别的喜好？'
    elif R == '评论':
        questions[R] = '人们对“{A}”有什么评论？'
    elif R == '主演':
        questions[R] = '“{A}”主演了哪些作品？'
    elif R == '星座':
        questions[R] = '“{A}”的星座是什么？'
    elif R == '时间':
        questions[R] = '“{A}”发生在什么时间？'
    elif R == '获奖':
        questions[R] = '“{A}”获得了哪些奖项？'
    elif R == '口碑':
        questions[R] = '“{A}”的口碑如何？'
    elif R == '成分':
        questions[R] = '“{A}”包含哪些成分？'
    elif R == '日期':
        questions[R] = '“{A}”的日期是什么时候？'
    elif R == '导演':
        questions[R] = '谁是“{A}”的导演？'
    elif R == '血型':
        questions[R] = '“{A}”的血型是什么？'
    elif R == '喜欢':
        questions[R] = '“{A}”喜欢什么？'
    elif R == '适合听':
        questions[R] = '在“{A}”的情况下适合听什么音乐？'
    elif R == '出生地':
        questions[R] = '“{A}”的出生地在哪里？'
    elif R == '订单量':
        questions[R] = '“{A}”的订单量有多少？'
    elif R == 'Town店） 特色菜':
        questions[R] = 'Town店的“{A}”有哪些特色菜？'
    elif R == 'Town店） 人均价格':
        questions[R] = 'Town店的“{A}”人均价格是多少？'
    elif R == '体重':
        questions[R] = '“{A}”的体重是多少？'
    elif R == '国家地区':
        questions[R] = '“{A}”属于哪个国家地区？'
    elif R == '属相':
        questions[R] = '“{A}”的属相是什么？'
    elif R == '道道道 口碑':
        questions[R] = '道道道的“{A}”口碑如何？'
    elif R == '30 40 评分':
        questions[R] = '30 40的“{A}”评分是多少？'
    elif R == '30 40 导演':
        questions[R] = '30 40的“{A}”有哪些导演？'
    elif R == '嘿玛 类型':
        questions[R] = '嘿玛的“{A}”是什么类型？'
    else:
        questions[R] = '关于“{A}”，有关于“{}”的信息吗？'.format(R)



    return Question
'''
# Code for genenrate json for the question and relation
        description_dic[R] = {
            "Question": Question,
            "Category": []
        }

        total_list.append(description_dic)
    out_file = os.path.join("../data_p", "relation_description.json")
    with open(out_file, "w") as f:
        json.dump(description_dic, f, indent=4)

'''





if __name__ == "__main__":
    result_dic = Get_analysis_for_datasets()
    Question_for_relation()
    