import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
import time
import random
from collections import defaultdict
from functools import partial
import math
from tqdm import tqdm
from torch_geometric.nn import LGConv
from torch_geometric.utils import dropout_edge, add_self_loops
from .loss_func import _L2_loss_mean, BPRLoss, InfoNCELoss, sce_loss

class LGCAR(torch.nn.Module):
    def __init__(self, config, edge_index):
        super(LGCAR, self).__init__()
        self.config = config
        self.users = config['users']
        self.items = config['items']
        self.entities = config['entities']
        self.relations = config['relations']
        self.interests = config['interests']
        self.api_attributes = config.get('api_attributes', 0)  
        self.layer = config['layer']
        self.emb_dim = config['dim']
        self.weight_decay = config['l2_reg']
        self.l2_reg_kge = config['l2_reg_kge']
        self.cf_weight = config['cf_weight']
        self.edge_drop = config['edge_dropout']
        self.message_drop_rate = config['message_dropout']
        self.message_drop = nn.Dropout(p=self.message_drop_rate)


        self.gcl_weight = config['gcl_weight']
        self.gcl_temp = config['gcl_temp']
        self.eps = config['eps']


        self.user_entity_emb = nn.Embedding(self.users+self.entities, self.emb_dim)

        nn.init.xavier_uniform_(self.user_entity_emb.weight)
        self.rel_emb = nn.Embedding(self.relations, self.emb_dim)
        nn.init.xavier_uniform_(self.rel_emb.weight)


        self.encoder_layer_emb_mask = config['encoder_layer_emb_mask']
        self.decoder_layer_emb_mask = config['decoder_layer_emb_mask']
        self._replace_rate = config['replace_rate']
        self._mask_token_rate = 1.0 - self._replace_rate
        self._drop_edge_rate = config['edge_mask_drop_edge_rate']
        self.interest_recon_weight = config['interest_recon_weight']
        self.criterion_emb_mask = self.setup_loss_fn(config['emb_mask_loss'], config['emb_mask_loss_alpha'])

        self.enc_mask_token = nn.Parameter(torch.zeros(1, self.emb_dim))
        self.encoder_to_decoder = nn.Linear(self.emb_dim, self.emb_dim)
        nn.init.xavier_uniform_(self.encoder_to_decoder.weight)
        #动态mask参数
        self.gmae_p = None
        self.total_epoch = config['total_epoch']
        self.increase_type = config['increase_type']
        self.min_mask_rate = config['min_mask_rate']
        self.max_mask_rate = config['max_mask_rate']
        
        # API服务语义重建模块
        self.api_attr_recon_weight = config.get('api_attr_recon_weight', 0.03) 

        self.propagate = LGConv(normalize=True)


        if config['add_self_loops']:

            edge_index, _ = add_self_loops(edge_index)
        self.edge_index = edge_index
        

        self.bpr = BPRLoss()

    def setup_loss_fn(self, loss_fn, alpha_l=2):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion

    def compute(self, x=None, edge_index=None, perturbed=False, mess_drop=False, layer_num=None):
        
        if x == None:
            emb = self.user_entity_emb.weight

        else:
            emb = x
        if self.message_drop_rate > 0.0 and mess_drop:
            emb = self.message_drop(emb)
        all_layer = [emb]
        if layer_num != None:
            layers = layer_num
        else:
            layers = self.layer
        for layer in range(layers):
            if edge_index == None:
                emb = self.propagate(emb, self.edge_index)
            else:
                emb = self.propagate(emb, edge_index)
            if perturbed:
                random_noise = torch.rand_like(emb).to(self.config['device'])
                emb += torch.sign(emb) * F.normalize(random_noise, dim=-1) * self.eps
            all_layer.append(emb)
        all_layer = torch.stack(all_layer, dim=1)
        all_layer = torch.mean(all_layer, dim=1)
        return all_layer                                 
   

    def forward(self, user_idx, pos_item, neg_item, edge_index_interest, edge_index_kg, edge_index_cf):
        """
        前向传播，计算BPR损失
        """
        # 使用增强表示计算BPR损失（与对比学习保持一致）
        all_layer_cf = self.compute(edge_index=edge_index_cf, perturbed=False, mess_drop=True)
        all_layer_kg = self.compute(edge_index=edge_index_kg, perturbed=False, mess_drop=False)
        all_layer_interest = self.compute(edge_index=edge_index_interest, perturbed=False, mess_drop=False)
        
        if self.edge_drop > 0.0:
            use_edge, _ = dropout_edge(edge_index=self.edge_index, force_undirected=True, 
                                      p=self.edge_drop, training=self.training)
            all_layer_lgcar = self.compute(edge_index=use_edge, mess_drop=True)
        else:
            all_layer_lgcar = self.compute(mess_drop=True)
        
        # 归一化
        all_layer_cf = F.normalize(all_layer_cf, dim=1)
        all_layer_kg = F.normalize(all_layer_kg, dim=1)
        all_layer_lgcar = F.normalize(all_layer_lgcar, dim=1)
        all_layer_interest = F.normalize(all_layer_interest, dim=1)
        
        # mashup意图增强
        mashup_cf = all_layer_cf[user_idx]
        mashup_interest = all_layer_interest[user_idx]
        enhanced_mashup_1 = F.normalize(mashup_interest + mashup_cf, dim=1)
        
        # API服务语义增强
        api_cf_pos = all_layer_cf[pos_item]
        api_kg_pos = all_layer_kg[pos_item]
        enhanced_api_pos_1 = F.normalize(api_cf_pos + api_kg_pos, dim=1)
        
        api_cf_neg = all_layer_cf[neg_item]
        api_kg_neg = all_layer_kg[neg_item]
        enhanced_api_neg_1 = F.normalize(api_cf_neg + api_kg_neg, dim=1)
        
        # 与LGCAR整图融合
        mashup_lgcar = all_layer_lgcar[user_idx]
        api_lgcar_pos = all_layer_lgcar[pos_item]
        api_lgcar_neg = all_layer_lgcar[neg_item]
        
        
        user_emb = F.normalize(enhanced_mashup_1 + mashup_lgcar, dim=1)
        
        pos_emb = F.normalize(enhanced_api_pos_1 + api_lgcar_pos, dim=1)
        neg_emb = F.normalize(enhanced_api_neg_1 + api_lgcar_neg, dim=1)
        

        users_emb_ego = self.user_entity_emb(user_idx)
        pos_emb_ego = self.user_entity_emb(pos_item)
        neg_emb_ego = self.user_entity_emb(neg_item)
        
        # 计算BPR损失
        pos_score = (user_emb * pos_emb).squeeze()
        neg_score = (user_emb * neg_emb).squeeze()
        
        # 计算BPR损失
        pos_score_sum = torch.sum(pos_score, dim=-1)
        neg_score_sum = torch.sum(neg_score, dim=-1)
        cf_loss = -torch.log(self.bpr.gamma + torch.sigmoid(pos_score_sum - neg_score_sum)).mean()
        
        # 正则化损失
        reg_loss = (1/2)*(users_emb_ego.norm(p=2).pow(2)+pos_emb_ego.norm(p=2).pow(2)+neg_emb_ego.norm(p=2).pow(2))/float(len(user_idx))
        loss = self.cf_weight*cf_loss + reg_loss*self.weight_decay

        return loss



    def cross_domain_contrastive_loss(self, user_idx, pos_item, edge_index_interest, edge_index_kg, edge_index_cf):

        # ========== 编码四个图的表示 ==========
        all_layer_cf = self.compute(edge_index=edge_index_cf, perturbed=True, mess_drop=False)  # 交互图
        all_layer_interest = self.compute(edge_index=edge_index_interest, perturbed=False, mess_drop=False)  # mashup意图图
        all_layer_kg = self.compute(edge_index=edge_index_kg, perturbed=False, mess_drop=False)  # API服务语义图
        all_layer_lgcar = self.compute(edge_index=self.edge_index, perturbed=False, mess_drop=False)  # LGCAR整图
        
        # 归一化
        all_layer_cf = F.normalize(all_layer_cf, dim=1)
        all_layer_interest = F.normalize(all_layer_interest, dim=1)
        all_layer_kg = F.normalize(all_layer_kg, dim=1)
        all_layer_lgcar = F.normalize(all_layer_lgcar, dim=1)
        

        mashup_cf = all_layer_cf[user_idx]
        mashup_interest = all_layer_interest[user_idx]
        api_cf = all_layer_cf[pos_item]
        api_kg = all_layer_kg[pos_item]
        mashup_lgcar = all_layer_lgcar[user_idx]
        api_lgcar = all_layer_lgcar[pos_item]
        
        # ========== 第一阶段：mashup意图图 vs 交互图 ==========

        pos_mashup_1 = torch.sum(mashup_interest * mashup_cf, dim=-1)
        tot_mashup_1 = torch.matmul(mashup_cf, torch.transpose(all_layer_interest[:self.users], 0, 1))
        logits_mashup_1 = tot_mashup_1 - pos_mashup_1[:, None]
        loss_mashup_1 = torch.mean(torch.logsumexp(logits_mashup_1 / self.gcl_temp, dim=1))
        

        enhanced_mashup_1 = F.normalize(mashup_interest + mashup_cf, dim=1)
        
        # ========== 第二阶段：api交互图 vs 服务语义图 ==========

        pos_api_1 = torch.sum(api_cf * api_kg, dim=-1)
        tot_api_1 = torch.matmul(api_cf, torch.transpose(all_layer_kg[self.users:self.users+self.items], 0, 1))
        logits_api_1 = tot_api_1 - pos_api_1[:, None]
        loss_api_1 = torch.mean(torch.logsumexp(logits_api_1 / self.gcl_temp, dim=1))
        

        enhanced_api_1 = F.normalize(api_cf + api_kg, dim=1)
        
        # ========== 第三阶段：enhanced_mashup vs LGCAR整图mashup ==========

        pos_mashup_2 = torch.sum(enhanced_mashup_1 * mashup_lgcar, dim=-1)
        tot_mashup_2 = torch.matmul(enhanced_mashup_1, torch.transpose(all_layer_lgcar[:self.users], 0, 1))
        logits_mashup_2 = tot_mashup_2 - pos_mashup_2[:, None]
        loss_mashup_2 = torch.mean(torch.logsumexp(logits_mashup_2 / self.gcl_temp, dim=1))

        final_mashup = F.normalize(enhanced_mashup_1 + mashup_lgcar, dim=1)
        
        # ========== 第四阶段：enhanced_api vs LGCAR整图api ==========

        pos_api_2 = torch.sum(enhanced_api_1 * api_lgcar, dim=-1)
        tot_api_2 = torch.matmul(enhanced_api_1, torch.transpose(all_layer_lgcar[self.users:self.users+self.items], 0, 1))
        logits_api_2 = tot_api_2 - pos_api_2[:, None]
        loss_api_2 = torch.mean(torch.logsumexp(logits_api_2 / self.gcl_temp, dim=1))
        

        final_api = F.normalize(enhanced_api_1 + api_lgcar, dim=1)
        
        # ========== 总损失 ==========
        total_loss = sum([loss_mashup_1, loss_api_1, loss_mashup_2, loss_api_2])
        return self.gcl_weight * total_loss


    def encoding_mask_noise(self, x, mask_rate=0.3):
        num_nodes = x.shape[0]
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes: ]

        if self._replace_rate > 0:
            num_noise_nodes = int(self._replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(self._mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self._replace_rate * num_mask_nodes):]]
            if num_noise_nodes > 0:
                noise_to_be_chosen = torch.randint(0, num_nodes, (num_noise_nodes,), device=x.device)
            else:
                noise_to_be_chosen = torch.tensor([], dtype=torch.long, device=x.device)
            out_x = x.clone()
            if len(noise_nodes) > 0 and len(noise_to_be_chosen) > 0:
                out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
        out_x[token_nodes] = 0.0
        out_x[token_nodes] += self.enc_mask_token
        return out_x, (mask_nodes, keep_nodes)

    def interest_recon_loss(self, edge_index):
        x = self.user_entity_emb.weight
        #mask user interest nodes
        user_emb, entity_emb, interest_emb = torch.split(x, [self.users, self.entities-self.interests, self.interests])
        use_interest, (mask_nodes, keep_nodes) = self.encoding_mask_noise(interest_emb, self.gmae_p)
        mask_nodes = mask_nodes + (self.users + self.entities-self.interests)
        keep_nodes = keep_nodes + (self.users + self.entities-self.interests)
        use_x = torch.cat((user_emb, entity_emb, use_interest))

        if self._drop_edge_rate > 0.0:
            use_edge_index, masked_edges = dropout_edge(edge_index, force_undirected=True, p=self._drop_edge_rate, training=self.training)
            use_edge_index = add_self_loops(use_edge_index)[0]
        else:
            use_edge_index = edge_index

        enc_rep = self.compute(x=use_x, edge_index=use_edge_index, layer_num=self.encoder_layer_emb_mask)
        rep = self.encoder_to_decoder(enc_rep)

        rep[mask_nodes] = 0.0
        recon = self.compute(x=rep, edge_index=use_edge_index, layer_num=self.decoder_layer_emb_mask)

        x_init = x[mask_nodes]
        x_rec = recon[mask_nodes]
        

        recon_loss = self.criterion_emb_mask(x_rec, x_init)
        
        return self.interest_recon_weight * recon_loss
    
    def api_attr_recon_loss(self, edge_index, api_attr_indices):

        if api_attr_indices is None or len(api_attr_indices) == 0:
            return torch.tensor(0.0, device=self.config['device'])
        

        x = self.user_entity_emb.weight
        api_attr_emb = x[api_attr_indices]
        use_api_attr, (mask_nodes, keep_nodes) = self.encoding_mask_noise(api_attr_emb, self.gmae_p)
        mask_nodes = api_attr_indices[mask_nodes]
        keep_nodes = api_attr_indices[keep_nodes]
        use_x = x.clone()
        use_x[api_attr_indices] = use_api_attr
        

        if self._drop_edge_rate > 0.0:
            use_edge_index, masked_edges = dropout_edge(edge_index, force_undirected=True, 
                                                        p=self._drop_edge_rate, training=self.training)
            use_edge_index = add_self_loops(use_edge_index)[0]
        else:
            use_edge_index = edge_index
        

        enc_rep = self.compute(x=use_x, edge_index=use_edge_index, layer_num=self.encoder_layer_emb_mask)
        

        rep = self.encoder_to_decoder(enc_rep)
        

        rep[mask_nodes] = 0.0
        

        recon = self.compute(x=rep, edge_index=use_edge_index, layer_num=self.decoder_layer_emb_mask)
        

        x_init = x[mask_nodes]
        x_rec = recon[mask_nodes]
        

        recon_loss = self.criterion_emb_mask(x_rec, x_init)
        

        return self.api_attr_recon_weight * recon_loss


    def calc_mask_rate(self, epoch):
        total_round, increase_type, init_rate, max_rate = self.total_epoch, self.increase_type, self.min_mask_rate, self.max_mask_rate
        if increase_type == "lin":
            increase_rate = (max_rate - init_rate) / total_round
            mask_rate = init_rate + epoch * increase_rate
        elif increase_type == "exp":
            alpha = (max_rate / init_rate) ** (1 / total_round)
            mask_rate = init_rate * alpha ** epoch
        mask_rate = min(mask_rate, max_rate)    
        return mask_rate


    def get_score_matrix(self, edge_index_interest, edge_index_kg, edge_index_cf):

        U_e, V_e = self.get_enhanced_embeddings(edge_index_interest, edge_index_kg, edge_index_cf)
        U_e = U_e.detach().cpu()
        V_e = V_e.detach().cpu()

        score_matrix = torch.matmul(U_e, V_e.t())   
        return score_matrix



    def get_enhanced_embeddings(self, edge_index_interest, edge_index_kg, edge_index_cf):

        all_layer_cf = self.compute(edge_index=edge_index_cf, perturbed=False, mess_drop=False)
        all_layer_kg = self.compute(edge_index=edge_index_kg, perturbed=False, mess_drop=False)
        all_layer_lgcar = self.compute(edge_index=self.edge_index, perturbed=False, mess_drop=False)
        all_layer_interest = self.compute(edge_index=edge_index_interest, perturbed=False, mess_drop=False)
        
        # 归一化
        all_layer_cf = F.normalize(all_layer_cf, dim=1)
        all_layer_kg = F.normalize(all_layer_kg, dim=1)
        all_layer_lgcar = F.normalize(all_layer_lgcar, dim=1)
        all_layer_interest = F.normalize(all_layer_interest, dim=1)
        
        # mashup意图增强
        mashup_cf = all_layer_cf[:self.users]
        mashup_interest = all_layer_interest[:self.users]
        enhanced_mashup_1 = F.normalize(mashup_interest + mashup_cf, dim=1)
        
        # API服务语义增强
        api_cf = all_layer_cf[self.users:self.users+self.items]
        api_kg = all_layer_kg[self.users:self.users+self.items]
        enhanced_api_1 = F.normalize(api_cf + api_kg, dim=1)
        

        mashup_lgcar = all_layer_lgcar[:self.users]
        api_lgcar = all_layer_lgcar[self.users:self.users+self.items]
        
        
        final_mashup = F.normalize(enhanced_mashup_1 + mashup_lgcar, dim=1)
        
        final_api = F.normalize(enhanced_api_1 + api_lgcar, dim=1)
        
        return final_mashup, final_api

    def get_kg_loss(self, batch_h, batch_t_pos, batch_t_neg, batch_r):
        h = self.user_entity_emb(batch_h)
        t_pos = self.user_entity_emb(batch_t_pos)
        t_neg = self.user_entity_emb(batch_t_neg)
        r = self.rel_emb(batch_r)
        pos_score = torch.sum(torch.pow(h + r - t_pos, 2), dim=1)     
        neg_score = torch.sum(torch.pow(h + r - t_neg, 2), dim=1)     
        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)
        l2_loss = _L2_loss_mean(h) + _L2_loss_mean(r) + _L2_loss_mean(t_pos) + _L2_loss_mean(t_neg)
        loss = kg_loss + self.l2_reg_kge * l2_loss
        return loss
