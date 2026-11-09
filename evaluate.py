import multiprocessing
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import time

cores = multiprocessing.cpu_count() // 2

Ks = [10, 20]
max_K = max(Ks)

score_matrix = None
rated_dict = None
candidate_dict = None
user_num = None
item_num = None

def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    return np.sum(r / np.log2(np.arange(2, r.size + 2)))


def ndcg_at_k(r, k, ground_truth):
    GT = set(ground_truth)
    if len(GT) > k :
        sent_list = [1.0] * k
    else:
        sent_list = [1.0]*len(GT) + [0.0]*(k-len(GT))
    dcg_max = dcg_at_k(sent_list, k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max


def recall_at_k(r, k, all_pos_num):
    r = r[:k]
    return np.sum(r) / all_pos_num


def hit_at_k(r, k):
    r = r[:k]
    if np.sum(r) > 0:
        return 1.
    else:
        return 0.

def test_one_user(uid):
    global score_matrix
    global rated_dict
    global candidate_dict
    global user_num
    global item_num
    score = score_matrix[uid]

    try:
        rated_items = np.array(list(rated_dict[uid]))

        score[rated_items-user_num] = -(1<<10)
    except:
        rated_items = np.array([])

    y_pred = score

    true_list = [x-user_num for x in candidate_dict[uid]]
    y_true = np.array([0.0]*item_num)
    y_true[true_list] = 1.0

    ground_true_num = len(true_list)
    recall, ndcg, hit_ratio = [], [], []

    topk_idx = np.argpartition(y_pred, -max_K)[-max_K:]

    order = topk_idx[np.argsort(y_pred[topk_idx])[::-1]]
    recall_sort = np.take(y_true, order)
    for k in Ks:
        recall.append(recall_at_k(recall_sort, k, ground_true_num))
        ndcg.append(ndcg_at_k(recall_sort, k, true_list))
        hit_ratio.append(hit_at_k(recall_sort, k))
    
    return {'recall': np.array(recall), 'ndcg': np.array(ndcg),
            'hit_ratio': np.array(hit_ratio)}

def eval_model(model, loader, dtype='test', edge_index_interest=None, edge_index_kg=None, edge_index_cf=None):
    global score_matrix
    global rated_dict
    global candidate_dict
    global user_num
    global item_num

    start = time.time()
    result = {'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks))}
    rated_dict, candidate_dict = loader.get_eval_data(dtype)
    user_num = model.users
    item_num = model.items
    ndcg = []
    recall = []
    model.eval()
    with torch.no_grad():
        score_matrix = model.get_score_matrix(edge_index_interest, edge_index_kg, edge_index_cf)
        score_matrix = score_matrix.detach().cpu().numpy()
        
    pred_true_uid = []
    n_test_users = len(candidate_dict.keys())
   
    batch_result = []
    for user_id in candidate_dict.keys():
        re = test_one_user(user_id)
        batch_result.append(re)
    for re in batch_result:
        result['recall'] += re['recall']/n_test_users
        result['ndcg'] += re['ndcg']/n_test_users
        result['hit_ratio'] += re['hit_ratio']/n_test_users

    end = time.time()

    return result
