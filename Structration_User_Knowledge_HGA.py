
import os
import time
import re
import sys
import jsonlines
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import clean_text
from utils import clean_interest
# 相似度计算相关包
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

if len(sys.argv) > 1:
    clustering_algorithm = sys.argv[1]
    print(f"[命令行] 使用聚类算法: {clustering_algorithm}")
else:
    clustering_algorithm = 'kmeans'

if len(sys.argv) > 2:
    n_clusters_kmeans = int(sys.argv[2])
    print(f"[命令行] KMeans聚类数: {n_clusters_kmeans}")
else:
    n_clusters_kmeans = 300  # 默认值

dataset = 'HGA'
field = 'apis'
    

r = ''


eps_dbscan = 0.2 
min_samples_dbscan = 2

max_his_num = 30


cluster_type = 'st'

data_path = f'./data/{dataset}'


item_list = list()    
entity_list = list()
lines = open(data_path+'/api_list.txt', 'r', encoding='utf-8').readlines()
for l in lines[1:]:
    if l == "\n":
        continue
    tmps = l.replace("\n", "").strip()
    elems = tmps.split('\t')
    if len(elems) < 3:
        continue
    org_id = elems[0]
    remap_id = elems[1]
    title = elems[2]    
    item_list.append([org_id, remap_id, title])
item_list = pd.DataFrame(item_list, columns=['org_id', 'remap_id', 'entity_name'])

lines = open(data_path+'/entity_list.txt', 'r', encoding='utf-8').readlines()
for l in lines[1:]:
    if l == "\n":
        continue
    tmps = l.replace("\n", "").strip()
    elems = tmps.split('\t')
    if len(elems) < 2:
        continue
    org_id = elems[0]
    remap_id = elems[1]
    entity_list.append([org_id, remap_id])
entity_list = pd.DataFrame(entity_list, columns=['org_id', 'remap_id'])

kg = pd.read_csv(data_path+'/kg_final_five.txt', sep=' ', names=['head', 'relation', 'tail'])
relation = pd.read_csv(data_path+'/relation_list.txt', sep=' ')
entity_list['remap_id'] = entity_list['remap_id'].astype(int)

relation_intent = kg['relation'].max() + 1
relation_sim = kg['relation'].max() + 2
items = item_list['remap_id'].to_list()


lower = 0
upper = 10
llm_answer = []
intent_triples = []
sim_triples = set()
intents_list = []
check_intent = []
waste = []

with open(output_path, mode='r') as f:
    for answer in jsonlines.Reader(f):
        row_id = answer['custom_id']
        raw_intents = answer['response']['body']['choices'][0]['message']['content']
        clean_intents = [clean_interest(it) for it in raw_intents.strip().split(',')]
        
        intents = []
        for it in clean_intents:
            if r == '':
                if len(it.split()) > lower and len(it.split()) <= upper:
                    intents.append(it)
                else:
                    print(f'waste:', it.split())
                    waste.append(it)
            elif r == '_r':
                if len(it) > lower and len(it) <= upper:
                    intents.append(it)
                else:
                    print(f'waste:', it)
                    waste.append(it)

        for it in intents:
            intent_triples.append([row_id, it])
        intents_list.extend(intents)


def encode_intents(cluster_type, intents_list):
    if cluster_type == 'st':

        device = 'cuda:0'

        local_model_path = './models/all-MiniLM-L6-v2'
        if os.path.exists(local_model_path):
            model = SentenceTransformer(local_model_path)
        else:
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        model.to(device)
        sentences = intents_list
        embeddings = model.encode(sentences, device=device)
        print("正在使用st")
        
    elif cluster_type == 'tfidf':

        sentences = intents_list

        vectorizer = TfidfVectorizer(max_df=0.8, ngram_range=(1, 2))
        embeddings = vectorizer.fit_transform(sentences)
        print("正在使用tfidf")
    
    return embeddings


embeddings = encode_intents(cluster_type, intents_list)


if clustering_algorithm == 'dbscan':

    from sklearn.cluster import DBSCAN
    
    print(f'使用DBSCAN自动聚类...')
    print(f'  eps={eps_dbscan}, min_samples={min_samples_dbscan}')
    
    dbscan = DBSCAN(
        eps=eps_dbscan,
        min_samples=min_samples_dbscan,
        metric='cosine'
    )
    cluster_ids = dbscan.fit_predict(embeddings)
    

    n_clusters = len(set(cluster_ids)) - (1 if -1 in cluster_ids else 0)
    n_noise = list(cluster_ids).count(-1)
    print(f'发现 {n_clusters} 个聚类')
    print(f'噪声点（无法聚类的词）: {n_noise} 个')
    

    print(f'过滤噪声点...')
    filtered_intents_list = []
    filtered_cluster_ids = []
    noise_intents = []
    
    for i in range(len(cluster_ids)):
        if cluster_ids[i] == -1:
            noise_intents.append(intents_list[i])
        else:
            filtered_intents_list.append(intents_list[i])
            filtered_cluster_ids.append(cluster_ids[i])
    
    print(f'过滤后保留兴趣数: {len(filtered_intents_list)}')
    print(f'过滤掉的噪声点: {len(noise_intents)}')
    
    # 打印噪声点详情
    if noise_intents:
        print("\n" + "="*80)
        print("噪声点详情 (无法聚类的兴趣):")
        print("="*80)
        print(f"总数: {len(noise_intents)} 个")
        print("\n前50个噪声点:")
        for idx, interest in enumerate(sorted(set(noise_intents))[:50], 1):
            print(f"  {idx}. {interest}")
        if len(set(noise_intents)) > 50:
            print(f"  ... 还有 {len(set(noise_intents)) - 50} 个不同的噪声点")
        print("="*80 + "\n")
    

    intents_list = filtered_intents_list
    cluster_ids = filtered_cluster_ids
    
    C = n_clusters 
    print(f'最终聚类数: {C}')

elif clustering_algorithm == 'kmeans':
    # 使用KMeans聚类
    print(f'使用KMeans聚类...')
    print(f'  n_clusters={n_clusters_kmeans}')
    
    kmeans = KMeans(
        n_clusters=n_clusters_kmeans,
        random_state=42,
        n_init=10,
        max_iter=300
    )
    cluster_ids = kmeans.fit_predict(embeddings)
    
    print(f'聚类完成，共 {n_clusters_kmeans} 个聚类')
    

    filtered_intents_list = intents_list
    filtered_cluster_ids = list(cluster_ids)
    
    C = n_clusters_kmeans
    print(f'最终聚类数: {C}')
    

    intents_list = filtered_intents_list
    cluster_ids = filtered_cluster_ids

else:
    raise ValueError(f"不支持的聚类算法: {clustering_algorithm}，请选择 'dbscan' 或 'kmeans'")


output_dict = dict(zip(intents_list, cluster_ids))


entity_add = []
entity_max = entity_list['remap_id'].max() + 1
for k, v in output_dict.items():
    entity_add.append([k, v + entity_max])
    output_dict[k] = v + entity_max
entity_add = pd.DataFrame(entity_add, columns=['org_id', 'remap_id'])

intent_triples_df = pd.DataFrame(intent_triples, columns=['uid', 'interest'])

intent_triples_df = intent_triples_df[intent_triples_df['interest'].isin(output_dict.keys())]
intent_triples_df['merged_interest'] = intent_triples_df['interest'].map(output_dict)
user_num = len(intent_triples_df['uid'].unique())
print(f'user num:{user_num}')
print(f'过滤噪声点后的边数: {len(intent_triples_df)}')

intent_triples_df_group = intent_triples_df.groupby('merged_interest').count()
intent_triples_df_sort = intent_triples_df_group.sort_values(by='uid', ascending=False)


del_list = []
del_list.extend(list(intent_triples_df_sort[intent_triples_df_sort['uid']>=int(user_num*0.4)].index))
print("clusters total:", C)
print("clusters deleted:", len(del_list))
print("clusters kept:", C - len(del_list))


if del_list:
    print("\n" + "="*80)
    print("被过滤的聚类详情 (用户数 >= 40%):")
    print("="*80)
    for idx, cluster_id in enumerate(del_list, 1):
        user_count = intent_triples_df_sort.loc[cluster_id, 'uid']

        cluster_interests = intent_triples_df[
            intent_triples_df['merged_interest'] == cluster_id
        ]['interest'].unique()
        
        print(f"\n过滤聚类 #{idx}:")
        print(f"  聚类ID: {cluster_id}")
        print(f"  用户数: {user_count} ({user_count/user_num*100:.1f}%)")
        print(f"  包含 {len(cluster_interests)} 个兴趣:")
        for interest in sorted(cluster_interests)[:20]:
            print(f"    - {interest}")
        if len(cluster_interests) > 20:
            print(f"    ... 还有 {len(cluster_interests) - 20} 个兴趣")
    print("="*80 + "\n")


intent_triples_df_filter = intent_triples_df[~intent_triples_df['merged_interest'].isin(del_list)]
print('interest graph edge num (before dedup):', len(intent_triples_df_filter))


intent_triples_df_filter = intent_triples_df_filter.drop_duplicates(subset=['uid', 'merged_interest'])
print('interest graph edge num (after dedup):', len(intent_triples_df_filter))

print('interest graph sparsity:', round(len(intent_triples_df_filter)/(user_num*C), 4)*100, '%')
kg_intent = intent_triples_df_filter.loc[:, ['uid', 'merged_interest']]
kg_intent.columns = ['uid', 'interest']
deg = intent_triples_df_filter.groupby("uid").size()
print("degree min/median/mean/max:", deg.min(), deg.median(), deg.mean(), deg.max())
print("users with degree<=1:", (deg<=1).sum(), "/", deg.shape[0])


kg_intent.to_csv(data_path+f'/mashup_functional_intent.txt', sep=' ', index=False, header=None)
