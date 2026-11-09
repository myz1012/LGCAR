import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
import time
import random
from collections import defaultdict
import math
from tqdm import tqdm
from torch_geometric.nn import LGConv, GATConv
from torch_geometric.utils import dropout_edge
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer  

class MyLoader(object):
    def __init__(self, config, verbose=True):
        self.config = config
        self.verbose = verbose
        self.loadData()
        
    def loadData(self):
        path = './data/'+self.config['dataset']


        train_path = self.config.get('train_path') or f'{path}/train.txt'
        test_path = self.config.get('test_path') or f'{path}/test.txt'
        if not os.path.isfile(train_path):
            raise FileNotFoundError(f'Train file not found: {train_path}')
        if not os.path.isfile(test_path):
            raise FileNotFoundError(f'Test file not found: {test_path}')
        self.train_path = os.fspath(train_path)
        self.test_path = os.fspath(test_path)
        

        kg_file = f'{path}/kg_final_four.txt'
        
        
        self.train = self._load_ratings(self.train_path)
        self.test = self._load_ratings(self.test_path)
        self.kg = self._load_kg(kg_file, self.config['inverse_r'])
        

        df = pd.concat((self.train, self.test))
        user_num = df['userid'].max() + 1
        

        try:

            api_list_df = pd.read_csv(f'{path}/api_list.txt', sep='\t')
            if 'org_id' in api_list_df.columns:
                all_api_ids_orig = set(api_list_df['org_id'].values)
            else:

                all_api_ids_orig = set(api_list_df.iloc[:, 0].values)
        except Exception as e:

            all_api_ids_orig = set(df['itemid'].unique())
        
        item_num = len(all_api_ids_orig)  # 所有API的数量
        

        kg_all_entities = set(self.kg['head'].unique()) | set(self.kg['tail'].unique())
        api_in_kg = kg_all_entities & all_api_ids_orig
        attr_ids_orig = kg_all_entities - all_api_ids_orig 
        

        self.kg_interest = pd.read_csv(f'{path}/mashup_functional_intent.txt', sep=' ', names=['uid', 'interest'])
        interest_ids_orig = set(self.kg_interest['interest'].unique())
        interest_num = len(interest_ids_orig)
        

        attr_ids_orig = attr_ids_orig - interest_ids_orig
        attr_num = len(attr_ids_orig)
        
        
        api_id_map = {}
        for idx, api_id in enumerate(sorted(all_api_ids_orig)):
            api_id_map[api_id] = user_num + idx
        

        attr_id_map = {}
        for idx, attr_id in enumerate(sorted(attr_ids_orig)):
            attr_id_map[attr_id] = user_num + item_num + idx
        

        interest_id_map = {}
        for idx, interest_id in enumerate(sorted(interest_ids_orig)):
            interest_id_map[interest_id] = user_num + item_num + attr_num + idx
        

        self.train['itemid'] = self.train['itemid'].apply(lambda x: api_id_map.get(x, x + user_num))
        self.test['itemid'] = self.test['itemid'].apply(lambda x: api_id_map.get(x, x + user_num))
        

        self.kg_interest['interest'] = self.kg_interest['interest'].apply(lambda x: interest_id_map.get(x, x + user_num))
        

        self.api_id_map = api_id_map
        self.attr_id_map = attr_id_map
        self.interest_id_map = interest_id_map
        

        entity_num = item_num + attr_num + interest_num

        

        if self.config['dataset'] == 'HGA':

            recon_relation_ids = [2, 3, 4, 5]
            
            self.kg_api_attr_recon = self.kg[
                self.kg['relation'].isin(recon_relation_ids)
                & self.kg['head'].isin(all_api_ids_orig)
                & self.kg['tail'].isin(attr_ids_orig)
            ].copy()
            self.kg_api_attr_recon.columns = ['api_id', 'relation', 'attribute_entity_id']

            self.kg_api_attr_recon['api_id'] = self.kg_api_attr_recon['api_id'].apply(
                lambda x: api_id_map.get(x, attr_id_map.get(x, interest_id_map.get(x, x))))
            self.kg_api_attr_recon['attribute_entity_id'] = self.kg_api_attr_recon['attribute_entity_id'].apply(
                lambda x: api_id_map.get(x, attr_id_map.get(x, interest_id_map.get(x, x))))
        else:

            recon_relation_ids = [2, 3, 4, 6]
            
            self.kg_api_attr_recon = self.kg[self.kg['relation'].isin(recon_relation_ids)].copy()
            self.kg_api_attr_recon.columns = ['api_id', 'relation', 'attribute_entity_id']

            self.kg_api_attr_recon['api_id'] = self.kg_api_attr_recon['api_id'].apply(
                lambda x: api_id_map.get(x, attr_id_map.get(x, interest_id_map.get(x, x))))
            self.kg_api_attr_recon['attribute_entity_id'] = self.kg_api_attr_recon['attribute_entity_id'].apply(
                lambda x: api_id_map.get(x, attr_id_map.get(x, interest_id_map.get(x, x))))
        

        api_attr_num = len(self.kg_api_attr_recon['attribute_entity_id'].unique())
        

        def map_entity_id(entity_id):
            """统一的实体ID映射函数"""
            if entity_id in api_id_map:
                return api_id_map[entity_id]
            elif entity_id in attr_id_map:
                return attr_id_map[entity_id]
            elif entity_id in interest_id_map:
                return interest_id_map[entity_id]
            else:

                return entity_id
        
        self.kg['head'] = self.kg['head'].apply(map_entity_id)
        self.kg['tail'] = self.kg['tail'].apply(map_entity_id)
        self.kg_org['head'] = self.kg_org['head'].apply(map_entity_id)
        self.kg_org['tail'] = self.kg_org['tail'].apply(map_entity_id)

        kg_interactive = pd.DataFrame({'head':self.train['userid'].to_list(), 'relation':[0]*len(self.train), 'tail':self.train['itemid'].to_list()})
        kg_interactive_r = pd.DataFrame({'head':self.train['itemid'].to_list(), 'relation':[1]*len(self.train), 'tail':self.train['userid'].to_list()})
        self.kg = pd.concat([self.kg, kg_interactive, kg_interactive_r])

        self.kg_dict = defaultdict(list)
        for row in self.kg.values:
            h_id, r_id, t_id = row
            self.kg_dict[h_id].append((r_id, t_id))
        
        self.config['users'] = user_num
        self.config['items'] = item_num  
        self.config['entities'] = entity_num  
        self.config['relations'] = self.kg['relation'].max() + 1
        self.config['interests'] = interest_num
        self.config['api_attributes'] = api_attr_num
        
        if self.verbose:
            self.statistic(df, self.kg_org)        
        self.prepare_eval()
        

        self._validate_id_ranges()

    def _load_ratings(self, file_name):
        inter_mat = list()
    
        lines = open(file_name, 'r').readlines()
        for l in lines:
            if l == "\n":
                continue
            tmps = l.replace("\n", "").strip()
            inters = [int(i) for i in tmps.split(' ')]
    
            u_id, pos_ids = inters[0], inters[1:]
            pos_ids = set(pos_ids)
    
            for i_id in pos_ids:
                inter_mat.append([u_id, i_id])
    
        df = pd.DataFrame(inter_mat, columns=['userid', 'itemid'])
        return df

    def _load_kg(self, file_name, inverse_r=True):
        kg = list()    
        lines = open(file_name, 'r').readlines()
        for l in lines:
            if l == "\n":
                continue
            tmps = l.replace("\n", "").strip()
            tup = [int(i) for i in tmps.split(' ')]
            kg.append(tup)

        can_triplets_np = np.array(kg)
        can_triplets_np = np.unique(can_triplets_np, axis=0)
        if inverse_r:

            inv_triplets_np = can_triplets_np.copy()
            inv_triplets_np[:, 0] = can_triplets_np[:, 2]
            inv_triplets_np[:, 2] = can_triplets_np[:, 0]
            inv_triplets_np[:, 1] = can_triplets_np[:, 1] + max(can_triplets_np[:, 1]) + 1

            can_triplets_np[:, 1] = can_triplets_np[:, 1] + 2
            inv_triplets_np[:, 1] = inv_triplets_np[:, 1] + 2

            triplets = np.concatenate((can_triplets_np, inv_triplets_np), axis=0)
        else:
            if self.config['use_ckg']:

                can_triplets_np[:, 1] = can_triplets_np[:, 1] + 2
            triplets = can_triplets_np.copy()
            
        df = pd.DataFrame(triplets, columns=['head', 'relation', 'tail'])
        self.config['n_triplets'] = len(df)
        self.kg_org = df.copy()
        return df
    
    def get_cf_loader(self,seed, bs=1024):
        dataset = TrainDataset(self.train, self.rated_dict, self.config['users'], self.config['items'],seed=seed)  

        generator = torch.Generator()
        generator.manual_seed(seed)
        return torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=bs, pin_memory=True,generator=generator,num_workers=0), dataset
    
    def get_eval_data(self, dtype='valid'):
        if dtype == 'valid':
            return self.rated_dict, self.item_dict
        else:
            return self.rated_dict, self.item_dict_test
    
    def prepare_eval(self):
        self.rated_dict = {}
        rated_group = self.train.groupby('userid')
        for u, v in rated_group:
            self.rated_dict[u] = set(v['itemid'].to_list())

        eval_users = list(self.test['userid'].unique())
        eval_df = self.test
        self.item_dict_test = {}
        item_group = eval_df.groupby('userid')
        for u, v in item_group:
            self.item_dict_test[u] = v['itemid'].values.tolist()
        return

    def statistic(self, df, kg):
        user_num = self.config['users']
        item_num = self.config['items']
        sparse = round(len(df)/(user_num*item_num), 4)
        interaction = len(df)
        tups = len(kg)
        itemset = set(df['itemid'].to_list())
        entityset = set(kg['head'].to_list()) | set(kg['tail'].to_list())
        return

    def _validate_id_ranges(self):

        

        user_ids = set(self.train['userid'].unique()) | set(self.test['userid'].unique())
        train_item_ids = set(self.train['itemid'].unique())
        test_item_ids = set(self.test['itemid'].unique())
        all_item_ids = train_item_ids | test_item_ids
        kg_entity_ids = set(self.kg_org['head'].unique()) | set(self.kg_org['tail'].unique())
        interest_ids = set(self.kg_interest['interest'].unique())
        

        user_num = self.config['users']
        item_num = self.config['items']
        attr_num = len(self.attr_id_map)
        interest_num = self.config['interests']
        

        
        

        user_item_overlap = user_ids & all_item_ids
        if user_item_overlap:
            raise ValueError("User and Item ID ranges must not overlap!")
        

        api_interest_overlap = all_item_ids & interest_ids
        if api_interest_overlap:
            raise ValueError("API and Interest ID ranges must not overlap!")
        

        user_interest_overlap = user_ids & interest_ids
        if user_interest_overlap:
            raise ValueError("User and Interest ID ranges must not overlap!")
        

        train_interactions = set(zip(self.train['userid'], self.train['itemid']))
        test_interactions = set(zip(self.test['userid'], self.test['itemid']))
        leakage = train_interactions & test_interactions
        if leakage:
            raise ValueError("Train-Test data leakage detected!")
        

        train_users = set(self.train['userid'].unique())
        test_users = set(self.test['userid'].unique())
        unseen_users = test_users - train_users
        

        unseen_items = test_item_ids - train_item_ids
        

        expected_api_min = user_num
        expected_api_max = user_num + item_num - 1
        actual_api_min = min(all_item_ids)
        actual_api_max = max(all_item_ids)
        

        expected_interest_min = user_num + item_num + attr_num
        expected_interest_max = user_num + item_num + attr_num + interest_num - 1
        actual_interest_min = min(interest_ids)
        actual_interest_max = max(interest_ids)
        

        kg_relations = set(self.kg['relation'].unique())
        


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, df, rated_dict, user_num, item_num,seed):
        self.pos = df
        self.pos_rec = rated_dict 
        self.user_num = user_num
        self.item_num = item_num
        
        self.train_cf = df.values
        self.seed = seed  # 保存seed
        random.seed(seed)  
    
    def generate_cf_neg(self, user):
        poss = self.pos_rec[user]
        while True:
            neg = random.randint(self.user_num, self.user_num+self.item_num-1)
            if neg not in poss:
                break
        return neg
            
    def __len__(self):
        return len(self.pos)
    
    def __getitem__(self, index):
        user = self.train_cf[index][0]
        pos = self.train_cf[index][1]
        neg = self.generate_cf_neg(user)
        return user, pos, neg

def init_seed(seed, reproducibility=True):
    r"""init random seed for random functions in numpy, torch, cuda and cudnn

    Args:
        seed (int): random seed
        reproducibility (bool): Whether to require reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if reproducibility:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def sample_pos_triples_for_h(kg_dict, head, n_sample_pos_triples):
    pos_triples = kg_dict[head]
    n_pos_triples = len(pos_triples)

    sample_relations, sample_pos_tails = [], []
    while True:
        if len(sample_relations) == n_sample_pos_triples:
            break

        pos_triple_idx = np.random.randint(low=0, high=n_pos_triples, size=1)[0]
        tail = pos_triples[pos_triple_idx][1]
        relation = pos_triples[pos_triple_idx][0]

        if relation not in sample_relations and tail not in sample_pos_tails:
            sample_relations.append(relation)
            sample_pos_tails.append(tail)
    return sample_relations, sample_pos_tails


def sample_neg_triples_for_h(kg_dict, head, relation, n_sample_neg_triples, highest_neg_idx, lowest_neg_idx):
    pos_triples = kg_dict[head]

    sample_neg_tails = []
    while True:
        if len(sample_neg_tails) == n_sample_neg_triples:
            break


        tail = np.random.randint(low=lowest_neg_idx, high=highest_neg_idx, size=1)[0]
        if (relation, tail) not in pos_triples and tail not in sample_neg_tails:
            sample_neg_tails.append(tail)
    return sample_neg_tails


def generate_kg_batch(kg_dict, batch_size, highest_neg_idx, lowest_neg_idx):
    exist_heads = kg_dict.keys()
    if batch_size <= len(exist_heads):
        batch_head = random.sample(exist_heads, batch_size)
    else:
        batch_head = [random.choice(exist_heads) for _ in range(batch_size)]

    batch_relation, batch_pos_tail, batch_neg_tail = [], [], []
    for h in batch_head:
        relation, pos_tail = sample_pos_triples_for_h(kg_dict, h, 1)
        batch_relation += relation
        batch_pos_tail += pos_tail

        neg_tail = sample_neg_triples_for_h(kg_dict, h, relation[0], 1, highest_neg_idx, lowest_neg_idx)
        batch_neg_tail += neg_tail

    batch_head = torch.LongTensor(batch_head)
    batch_relation = torch.LongTensor(batch_relation)
    batch_pos_tail = torch.LongTensor(batch_pos_tail)
    batch_neg_tail = torch.LongTensor(batch_neg_tail)
    return batch_head, batch_relation, batch_pos_tail, batch_neg_tail


try:
    sw = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords', quiet=True)
    sw = set(stopwords.words('english'))


try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()

def clean_text(text):    
    text = text.lower()    
    text = re.sub(r"[^a-zA-Z?.!,¿]+", " ", text) # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")    
    punctuations = '@#!?+&*[]-%.:/();$=><|{}^' + "'`" + '_'
    for p in punctuations:
        text = text.replace(p,'') #Removing punctuations        
    text = [word.lower() for word in text.split() if word.lower() not in sw]    
    text = [lemmatizer.lemmatize(word) for word in text]    
    text = " ".join(text) #removing stopwords    
    return text


def clean_interest(s: str) -> str:

    if s is None:
        return ""
    s = str(s).strip()

    s = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff&\+\-/\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    if re.match(r'^[a-zA-Z0-9&\+\-/\s]+$', s):
        s = s.lower()
    return s
