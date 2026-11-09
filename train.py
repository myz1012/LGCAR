import numpy as np
import torch
import os
import re
import time
import json
from tqdm import tqdm
from torch_geometric.data import Data
import argparse
import pickle
import random
from datetime import datetime

from models.Model import LGCAR
from utils import MyLoader, init_seed, generate_kg_batch
from evaluate import eval_model
from prettytable import PrettyTable
from torch_geometric.utils import coalesce



def parse_args():
    parser = argparse.ArgumentParser(description="Run LGCAR.")
    parser.add_argument('--seed', type=int, default=2025,
                        help='Random seed.')
    parser.add_argument('--dataset', nargs='?', default='pwa',
                        help='Choose a dataset from {pwa,HGA}')
    parser.add_argument('--train_path', '--train-path', dest='train_path', default=None,
                        help='Optional interaction train file; defaults to data/{dataset}/train.txt')
    parser.add_argument('--test_path', '--test-path', dest='test_path', default=None,
                        help='Optional interaction test file; defaults to data/{dataset}/test.txt')
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")
    parser.add_argument('--dim', type=int, default=None,
                        help='Embedding dimension (overrides config file if specified)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs (overrides config file if specified)')
    args = parser.parse_args()
    return args

args = parse_args()
dataset = args.dataset
config = json.load(open(f'./config/{dataset}.json'))


if args.train_path is not None:
    config['train_path'] = args.train_path
if args.test_path is not None:
    config['test_path'] = args.test_path

# 命令行指定 dim 会覆盖配置文件
if args.dim is not None:
    config['dim'] = args.dim

# 命令行指定 epochs 会覆盖配置文件
if args.epochs is not None:
    config['num_epoch'] = args.epochs


seed = args.seed
Ks = config['Ks']
init_seed(seed, True)

loader = MyLoader(config)

# ========= 构图 =========
#1.协同过滤图(mashup-api交互)
src = loader.train.loc[:, 'userid'].to_list()+ loader.train.loc[:, 'itemid'].to_list()
tgt = loader.train.loc[:, 'itemid'].to_list() + loader.train.loc[:, 'userid'].to_list()
src_cf = loader.train.loc[:, 'userid'].to_list()+ loader.train.loc[:, 'itemid'].to_list()
tgt_cf = loader.train.loc[:, 'itemid'].to_list() + loader.train.loc[:, 'userid'].to_list()
edge_index = [src, tgt]
#2.知识图谱(实体-关系-实体)
src_k = loader.kg_org['head'].to_list() + loader.kg_org['tail'].to_list()
tgt_k = loader.kg_org['tail'].to_list() + loader.kg_org['head'].to_list()
edge_index[0].extend(src_k)
edge_index[1].extend(tgt_k)



#3.mashup功能意图图（mashup-意图）
src_in = loader.kg_interest['uid'].to_list() + loader.kg_interest['interest'].to_list()
tgt_in = loader.kg_interest['interest'].to_list() + loader.kg_interest['uid'].to_list()
edge_index[0].extend(src_in)
edge_index[1].extend(tgt_in)

api_attr_indices = torch.LongTensor(sorted(loader.kg_api_attr_recon['attribute_entity_id'].unique().tolist()))
api_attr_indices = api_attr_indices.to(config['device'])

#4.构建意图图
edge_index_ig = [src_in+src_cf, tgt_in+tgt_cf]
edge_index_ig = torch.LongTensor(edge_index_ig)
edge_index_ig = coalesce(edge_index_ig)
graph_ig = Data(edge_index=edge_index_ig.contiguous())
graph_ig = graph_ig.to(config['device'])

#构建知识图谱
edge_index_kg = [src_k+src_cf, tgt_k+tgt_cf]
edge_index_kg = torch.LongTensor(edge_index_kg)
edge_index_kg = coalesce(edge_index_kg)
graph_kg = Data(edge_index=edge_index_kg.contiguous())
graph_kg = graph_kg.to(config['device'])

#构建完整图
edge_index = torch.LongTensor(edge_index)
edge_index = coalesce(edge_index)
graph = Data(edge_index=edge_index.contiguous())
graph = graph.to(config['device'])




num_edges = graph.edge_index.shape[1]



#构建纯协同过滤图
edge_index_cf = torch.LongTensor([src_cf, tgt_cf])
edge_index_cf = coalesce(edge_index_cf)
graph_cf = Data(edge_index=edge_index_cf.contiguous())
graph_cf = graph_cf.to(config['device'])



model = LGCAR(config, graph.edge_index)
model = model.to(config['device'])


optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

if config['use_kge']:
    kg_optimizer = torch.optim.Adam(model.parameters(), lr=config['lr_kg'])
train_loader, _ = loader.get_cf_loader(args.seed,bs=config['batch_size'])

best_res = None
best_loss = 1e10
best_score = 0
patience = config['patience']
best_performance = []

# 创建结果保存目录
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
result_dir = f"results/{dataset}_{timestamp}"
os.makedirs(result_dir, exist_ok=True)

# 保存超参数（转换 numpy 类型为 Python 原生类型）
def convert_to_serializable(obj):
    """将 numpy 类型转换为 Python 原生类型"""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj

hyperparams = {
    'dataset': dataset,
    'seed': seed,
    'lr': config['lr'],
    'batch_size': config['batch_size'],
    'num_epoch': config['num_epoch'],
    'timestamp': timestamp
}
hyperparams.update(config)
hyperparams = convert_to_serializable(hyperparams)

with open(f"{result_dir}/hyperparameters.json", 'w') as f:
    json.dump(hyperparams, f, indent=4)

# 记录训练过程
results_history = []
best_epoch = 0


for epoch in range(config['num_epoch']):
    
    loss_list = []
    print(f'epoch {epoch+1} start!')
    model.train()
    start = time.time() 
    model.gmae_p = model.calc_mask_rate(epoch)

    for data in train_loader:
        user, pos, neg = data
        user, pos, neg = user.to(config['device']), pos.to(config['device']), neg.to(config['device'])

        optimizer.zero_grad()

        edge_index_interest = graph_ig.edge_index
        loss_rec = model(user, pos, neg, edge_index_interest, graph_kg.edge_index, graph_cf.edge_index)
        

        loss = loss_rec
        
        # 跨域对比学习损失
        loss_cross_domain_contrastive = model.cross_domain_contrastive_loss(user, pos, graph_ig.edge_index, graph_kg.edge_index, graph_cf.edge_index)
        loss = loss + loss_cross_domain_contrastive

        # Mashup意图重建损失
        loss_interest_recon = model.interest_recon_loss(graph.edge_index)
        loss = loss + loss_interest_recon

        # API服务语义重建损失
        loss_api_attr_recon = model.api_attr_recon_loss(graph.edge_index, api_attr_indices)
        loss = loss + loss_api_attr_recon

        loss.backward()
        optimizer.step()
        loss_list.append(loss.detach().cpu().numpy())
        
    sum_loss = np.sum(loss_list)/len(loss_list)
    

    
    if config['use_kge']:

        kg_total_loss = 0
        n_kg_batch = config['n_triplets'] // 4096
        for iter in range(1, n_kg_batch + 1):

            kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail = generate_kg_batch(loader.kg_dict, 4096, config['entities']-config['interests'], 0)
            kg_batch_head = kg_batch_head.to(config['device'])
            kg_batch_relation = kg_batch_relation.to(config['device'])
            kg_batch_pos_tail = kg_batch_pos_tail.to(config['device'])
            kg_batch_neg_tail = kg_batch_neg_tail.to(config['device'])
    
            kg_batch_loss = model.get_kg_loss(kg_batch_head, kg_batch_pos_tail, kg_batch_neg_tail, kg_batch_relation)
    
            kg_optimizer.zero_grad()
            kg_batch_loss.backward()
            kg_optimizer.step()
            kg_total_loss += kg_batch_loss.item()
    if (epoch+1) % config['eval_interval'] == 0:            
        result = eval_model(
            model,
            loader,
            'test',
            graph_ig.edge_index,
            graph_kg.edge_index,
            graph_cf.edge_index,
        )

        score = result['recall'][1] + result['ndcg'][1]
        
        # 记录评估结果
        epoch_result = {'epoch': epoch + 1}
        for i, k in enumerate(Ks):
            epoch_result[f'recall@{k}'] = round(result['recall'][i], 4)
            epoch_result[f'ndcg@{k}'] = round(result['ndcg'][i], 4)
            epoch_result[f'hit_ratio@{k}'] = round(result['hit_ratio'][i], 4)
        results_history.append(epoch_result)


        print(f"Evaluation at Epoch {epoch + 1}:")
        evaluation_tables = []
        metric_names = ('recall', 'ndcg', 'hit_ratio')
        for k in Ks:
            table = PrettyTable([f'{metric}@{k}' for metric in metric_names])
            table.add_row([epoch_result[f'{metric}@{k}'] for metric in metric_names])
            print(table)
            evaluation_tables.append(table)
        
        if best_epoch == 0 or score > best_score:
            best_performance = evaluation_tables
            best_score = score
            best_epoch = epoch + 1
            patience = config['patience']
            
            # 保存最佳模型
            model_save_path = f"{result_dir}/best_model.pth"
            torch.save(model.state_dict(), model_save_path)
            


    if config['use_kge']:
        print(f'kg loss:{round(kg_total_loss / (n_kg_batch+1), 4)}')    
    print('epoch loss: ', round(sum_loss, 4))
    print('current mask rate:', round(model.gmae_p, 4))
    print('current learning rate:', round(optimizer.param_groups[0]['lr'], 6))

    end = time.time()
    print(f'train time {round(end-start)}s')
    print('-'*90)

print('Best testset performance:')
for table in best_performance:
    print(table)

# 保存最佳结果的详细指标
if best_epoch > 0:
    best_results = {
        'best_epoch': best_epoch,
        'best_score': float(best_score),
        'seed': seed,
    }

    best_row = next(row for row in results_history if row['epoch'] == best_epoch)
    for k in Ks:
        best_results[f'recall@{k}'] = float(best_row[f'recall@{k}'])
        best_results[f'ndcg@{k}'] = float(best_row[f'ndcg@{k}'])
        best_results[f'hit_ratio@{k}'] = float(best_row[f'hit_ratio@{k}'])
    
    with open(f"{result_dir}/best_results.json", 'w') as f:
        json.dump(best_results, f, indent=4)

print(f"最佳模型出现在 Epoch {best_epoch}，得分: {best_score:.4f}")
