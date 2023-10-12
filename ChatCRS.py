import configparser
import json
from models import generate_inference
from data_loader import prepare_test_data
import time


def get_config():
    config = configparser.ConfigParser()
    config.read('config.ini')
    model = []
    task = []
    data = []
    evaluation = []

    # choose example configuration or personalised configuration
    # future add to argument parser
    # SETTING = config['EXAMPLE']
    SETTING = config['PERSONAL']
    ####################################
    
    for key in json.loads(SETTING['task']):
        new_key = 'task' + str(key)
        task.append(config['DEFAULT'][new_key])
    for key in json.loads(SETTING['model']):
        new_key = 'model' + str(key)
        model.append(config['DEFAULT'][new_key])
    for key in json.loads(SETTING['dataset']):
        new_key = 'dataset' + str(key)
        data.append(config['DEFAULT'][new_key])
    for key in json.loads(SETTING['evaluation']):
        new_key = 'evaluation' + str(key)
        evaluation.append(config['DEFAULT'][new_key])
    if 'api_key' in config['DEFAULT']:
        api_key = config['DEFAULT']['api_key']
    else:
        api_key = None
        print("No API key found in config.ini. Please add your API key to the config.ini file. Otherwise, you will not be able to use the ChatGPT")
    return model, task, data, evaluation, api_key
    # return model, task, data, evaluation

       

def task_distributor(model, task, data, evaluation, api_key):
    # breakpoint()


    for model_name in model:
        for data_d in data:
            print("Starting testing on models and datasets, total 3 steps")
            print("Model: ", model_name, "Dataset: ", data_d)
            test_loader = prepare_test_data(task, data_d)
            generate_inference(model_name, test_loader, api_key, evaluation, data_d, task)






if __name__ == "__main__":
    
    model, task, data, evaluation, api_key = get_config()
    task_distributor(model, task, data, evaluation, api_key)


''' 
Overall logic for ChatCRS Evaluation Process:

- Get the config file for the model, task, data, and evaluation (ChatCRS.py; config.ini; config.py)
    - check the format for any error case
    - save the config as variables

- Load the data (data_loader.py)
    - for each dataset, load the tasks (if task is contained in the dataset)
    - for each task, design sepecific instructions for the task
    - prepare necessary supporting task data and send to dataloader

- Load the model (models.py)
    - for each model, load the model
    - for each dataset, load dataloader for each specific task
    - run model on the data and save for evaluation
     
- Load the evaluation (evaluation.py)
    - for each save results (model, dataset, tasks), load the evaluation
    - for each task, run the evaluation
        - Metric-based evaluation
        - Plot the results and graphs
    - for response generation task (additional task):
        - evaluation on each supporting task
        - prepare scripts for human evaluation
'''
