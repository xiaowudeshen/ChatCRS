import json
import os 
import pickle
import logging

logger = logging.getLogger(__name__)

### Data Reader
def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def save_json(file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f)

def save_pickle(file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def load_and_cache_examples(args, evaluation = False):
    mode = 'test' if evaluation else 'train'
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format( 
        args.data_name, 
        mode,
        # list(filter(None, args.model_name.split("/"))).pop(),
        str(args.max_seq_length),
        str(args.max_target_length)))

    if os.path.exists(cached_features_file):
        print("Loading features from cached file %s", cached_features_file)
        features = load_pickle(cached_features_file)
        print("Loaded number of instances:", len(features['resp']['source_ids']) )
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        features = convert_to_features(args, mode)
        print("Loaded number of instance:", len(features['resp']['source_ids']))
    
        logger.info("Saving features into cached file %s", cached_features_file)
        write_pkl(features, cached_features_file)
    return features

def convert_to_features(args, mode):

    ath = os.path.join(args.data_dir, '{}/item2id.txt'.format(args.data_name))
    with open(path, 'r', encoding='utf-8') as infile:
        #item_dict = {0:'PAD'}
        item_dict = {}
        for line in infile:
            items = line.strip().split('\t')
            #item_dict[int(items[1])+1] = items[0]
            item_dict[int(items[1])] = items[0]
        item_dict[len(item_dict)] = '<PAD>'

    if args.data_name == 'durecdial':   
        path = os.path.join(args.data_dir, 'kb_{}.jsonl'.format(args.data_name))
        outfile = open(path, 'w', encoding='utf-8')
    path = os.path.join(args.data_dir, '{}/{}.jsonl'.format(args.data_name, mode))
    print('tokenizing {}'.format(path))
    #print(tokenizer.SPECIAL_TOKENS_ATTRIBUTES)
    data_dict = {'resp':{'source_ids':[], 'target_ids':[], 'item_ids':[]}, 'item':{'source_ids':[], 'target_ids':[], 'item_ids':[]}, 'goal':{'source_ids':[], 'target_ids':[], 'item_ids':[]}, 'know':{'source_ids':[], 'target_ids':[], 'item_ids':[]}}
    with open(path, 'r', encoding='utf-8') as infile:
        max_dia_len = 0
        avg_dia_len = []
        max_res_len = 0
        avg_res_len = []
        source_ids = []
        source_infos = []
        target_ids = []
        target_infos = []
        item_ids = []
        hist_ids = []
        item_infos = []
        hist_infos = []
        rec_index = []
        i = 0
        for line in infile:
            d = eval(line.strip())
            know = d['knowledge']
            conv = d['conversation']
            source_id = []
            source_know_id = []
            source_goal_id = []
            target_id = []
            source_info = ''
            source_know_info = ''
            source_goal_info = ''
            target_info = ''
            hist_id = know['item_history'] if len(know['item_history'])>0 else [len(item_dict)-1]
            #hist_id = tokenizer.encode('[history]' + '|'.join(['<'+str(x)+'>' for x in know['item_history']]))[1:]
            # profile_id = tokenizer.encode('[profile]' + '|'.join(know['user_profile']))[1:]
            profile_info = '[profile]' + '|'.join(know['user_profile'])

            first_utt = conv[0]
            if first_utt['role'] == 'user' and args.data_name == 'durecdial':
                pass
            else:
                if type(first_utt['goal']) is list:
                    first_utt['goal'] = '|'.join(first_utt['goal'])
                # source_goal_id += tokenizer.encode('[goal]' + first_utt['goal'])[1:]
                source_goal_info += '[goal]' + first_utt['goal']
                # source_know_id += tokenizer.encode('[knowledge]' + '|'.join(first_utt['knowledge']))[1:]
                source_know_info += '[knowledge]' + '|'.join(first_utt['knowledge'])
            # source_id += tokenizer.encode('[{}]'.format(first_utt['role']) + first_utt['utterance'])[1:]
            source_info += '{}'.format(first_utt['role']) + first_utt['utterance']
            # source_goal_id += tokenizer.encode('[{}]'.format(first_utt['role']) + first_utt['utterance'])[1:]
            source_goal_info += '{}'.format(first_utt['role']) + first_utt['utterance']
            # source_know_id += tokenizer.encode('[{}]'.format(first_utt['role']) + first_utt['utterance'])[1:]
            source_know_info += '{}'.format(first_utt['role']) + first_utt['utterance']

            for utt in conv[1:]:
                if utt['role'] == 'user':# and args.data_name == 'durecdial':
                    # source_id += tokenizer.encode('[user]' + utt['utterance'])[1:]
                    source_info += '[user]' + utt['utterance']
                    # if args.data_name == 'tgredial':
                    #     source_know_id += tokenizer.encode('[knowledge]' + '|'.join(utt['knowledge']))[1:]
                    #     source_goal_id += tokenizer.encode('[goal]' + '|'.join(utt['goal']))[1:]
                    # source_know_id += tokenizer.encode('[user]' + utt['utterance'])[1:]
                    source_know_info += '[user]' + utt['utterance']
                    # source_goal_id += tokenizer.encode('[user]' + utt['utterance'])[1:]
                    source_goal_info += '[user]' + utt['utterance']
                    continue
                if type(utt['goal']) is list:
                    utt['goal'] = '|'.join(utt['goal'])

                ### prepare response generation data 
                # target_id = tokenizer.encode(utt['utterance'])
                target_info = utt['utterance']
                know_len = int(args.max_seq_length/2)
                if args.data_name == 'tgredial':
                    # new_source_id = source_id + tokenizer.encode('[goal]' + utt['goal'])[1:] + tokenizer.encode('[knowledge]')[1:-1] + tokenizer.encode('|'.join(utt['knowledge']))[1:][-know_len:] + tokenizer.encode('[item]' + '|'.join(utt['item']))[1:] + tokenizer.encode('[{}]'.format(utt['role']))[1:]
                    new_source_info = source_info + '[goal]' + utt['goal'] + '[knowledge]' + '|'.join(utt['knowledge'])[-know_len:] + '[item]' + '|'.join(utt['item']) + '[{}]'.format(utt['role'])
                else:
                    # new_source_id = source_id + tokenizer.encode('[goal]' + utt['goal'])[1:] + tokenizer.encode('[knowledge]')[1:-1] + tokenizer.encode('|'.join(utt['know_text']))[1:][-know_len:] + tokenizer.encode('[item]' + '|'.join(utt['item']))[1:] + tokenizer.encode('[{}]'.format(utt['role']))[1:]
                    new_source_info = source_info + '[goal]' + utt['goal'] + '[knowledge]' + '|'.join(utt['know_text'])[-know_len:] + '[item]' + '|'.join(utt['item']) + '[{}]'.format(utt['role'])
                    if mode == 'test':
                        outfile.write(str(know['knowledge']) + '\n')


                # source_ids.append([101] + new_source_id[-args.max_seq_length+1:])
                source_infos.append(new_source_info[-args.max_seq_length:])
                # target_ids.append([101] + target_id[-args.max_target_length+1:])
                target_infos.append(target_info[-args.max_target_length:])
                item_ids.append([len(item_dict)-1])
                
                # data_dict['resp']['source_ids'].append(source_ids[-1])
                data_dict['resp']['source_infos'].append(source_infos[-1])
                # data_dict['resp']['target_ids'].append(target_ids[-1])
                data_dict['resp']['target_infos'].append(target_infos[-1])
                data_dict['resp']['item_ids'].append(item_ids[-1])
                

                avg_dia_len.append(len(new_source_id))
                max_dia_len = max(max_dia_len, len(new_source_id))
                avg_res_len.append(len(target_id))
                max_res_len = max(max_res_len, len(target_id))

                ### prepare goal selection data
                # target_id = tokenizer.encode(utt['goal'])
                target_info = utt['goal']
                # new_source_id = source_goal_id + tokenizer.encode('[goal]')[1:]
                new_source_info = source_goal_info + '[goal]'
                # source_goal_id += (tokenizer.encode('[goal]' + utt['goal'])[1:] + tokenizer.encode('[{}]'.format(utt['role']) + utt['utterance'])[1:])
                source_goal_info += ('[goal]' + utt['goal'] + '[{}]'.format(utt['role']) + utt['utterance'])
                # source_ids.append([101] + new_source_id[-args.max_seq_length+1:])
                source_infos.append(new_source_info[-args.max_seq_length:])
                # target_ids.append([101] + target_id[-args.max_target_length+1:])
                target_infos.append(target_info[-args.max_target_length:])
                item_ids.append([len(item_dict)-1])
                # data_dict['goal']['source_ids'].append(source_ids[-1])
                data_dict['goal']['source_infos'].append(source_infos[-1])
                # data_dict['goal']['target_ids'].append(target_ids[-1])
                data_dict['goal']['target_infos'].append(target_infos[-1])
                data_dict['goal']['item_ids'].append(item_ids[-1])

                ### prepare topic prediction data
                # target_id = tokenizer.encode('|'.join(utt['knowledge']))
                target_info = '|'.join(utt['knowledge'])
                # new_source_id = profile_id + source_know_id + tokenizer.encode('[goal]' + utt['goal'])[1:] + tokenizer.encode('[knowledge]')[1:]
                new_source_info = profile_info + '[goal]' + utt['goal'] + '[knowledge]'
                #new_source_id = profile_id + source_know_id + tokenizer.encode('[knowledge]')[1:]
                # source_know_id += (tokenizer.encode('[knowledge]' + '|'.join(utt['knowledge']))[1:] + tokenizer.encode('[{}]'.format(utt['role']) + utt['utterance'])[1:])
                source_know_info += ('[knowledge]' + '|'.join(utt['knowledge']) + '[{}]'.format(utt['role']) + utt['utterance'])
                # source_ids.append([101] + new_source_id[-args.max_seq_length+1:])
                source_infos.append(new_source_info[-args.max_seq_length:])
                # target_ids.append([101] + target_id[-args.max_target_length+1:])
                target_infos.append(target_info[-args.max_target_length:])
                item_ids.append([len(item_dict)-1])
                # data_dict['know']['source_ids'].append(source_ids[-1])
                data_dict['know']['source_infos'].append(source_infos[-1])
                # data_dict['know']['target_ids'].append(target_ids[-1])
                data_dict['know']['target_infos'].append(target_infos[-1])
                data_dict['know']['item_ids'].append(item_ids[-1])
                
                ### prepare item recommendation data
                if len(utt['item_id']) > 0:
                    target_text = []
                    for item, item_id in zip(utt['item'], utt['item_id']):
                        target_text.append('<'+str(item_id)+'>'+item)
                    # target_id = tokenizer.encode('|'.join(target_text))
                    target_info = '|'.join(target_text)
                    # new_source_id = profile_id + source_id + tokenizer.encode('[goal]' + utt['goal'])[1:] + tokenizer.encode('[knowledge]' + '|'.join(utt['knowledge']))[1:] + tokenizer.encode('[item]')[1:]#  
                    new_source_info = profile_info + '[goal]' + utt['goal'] + '[knowledge]' + '|'.join(utt['knowledge']) + '[item]'#
                    item_id = utt['item_id']
                    # source_ids.append([101] + new_source_id[-args.max_seq_length+1:])
                    source_infos.append(new_source_info[-args.max_seq_length:])
                    # target_ids.append([101] + target_id[-args.max_target_length+1:])
                    target_infos.append(target_info[-args.max_target_length:])
                    item_ids.append(item_id)
                    # data_dict['item']['source_ids'].append(source_ids[-1])
                    data_dict['item']['source_infos'].append(source_infos[-1])
                    # data_dict['item']['target_ids'].append(target_ids[-1])
                    data_dict['item']['target_infos'].append(target_infos[-1])
                    data_dict['item']['item_ids'].append(item_ids[-1])
                    rec_index.append(i)
                i += 1

                source_id += tokenizer.encode('[{}]'.format(utt['role']) + utt['utterance'])[1:]
                
                #hist_ids.append(hist_id)
                #hist_id.extend(item_id)

        print('{} set, max_res_len: {}, max_dia_len: {}, avg_res_len: {}, avg_dia_len: {}'.format(mode, max_res_len, max_dia_len, float(sum(avg_res_len))/len(avg_res_len), float(sum(avg_dia_len))/len(avg_dia_len)))
