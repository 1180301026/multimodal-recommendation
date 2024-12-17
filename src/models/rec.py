# coding: utf-8
# @email: enoche.chow@gmail.com
r"""

################################################
"""
import os
import copy
import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity
import torch.nn.functional as F
from common.abstract_recommender import GeneralRecommender
from common.loss import EmbLoss

def build_knn_neighbourhood(adj, topk):
    knn_val, knn_ind = torch.topk(adj, topk, dim=-1)
    weighted_adjacency_matrix = (torch.zeros_like(adj)).scatter_(-1, knn_ind, knn_val)
    return weighted_adjacency_matrix
def compute_normalized_laplacian(adj):
    rowsum = torch.sum(adj, -1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt)
    L_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    return L_norm
def build_sim(context):
    context_norm = context.div(torch.norm(context, p=2, dim=-1, keepdim=True))
    sim = torch.mm(context_norm, context_norm.transpose(1, 0))
    return sim

class BM3(GeneralRecommender):
    def __init__(self, config, dataset):
        super(BM3, self).__init__(config, dataset)
        self.device = config['device']
        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['embedding_size']
        self.n_layers = config['n_layers']
        self.reg_weight = config['reg_weight']
        self.cl_weight = config['cl_weight']
        self.dropout = config['dropout']
        self.gru = nn.GRU(self.embedding_dim, self.embedding_dim, 1)
        initializer = nn.init.xavier_uniform_
        self.w_q=nn.Parameter(initializer(torch.empty([config['embedding_size'], config['embedding_size']])))
        self.w_k=nn.Parameter(initializer(torch.empty([config['embedding_size'], config['embedding_size']])))
        self.w_self_attention_cat= nn.Parameter(initializer(torch.empty([config['head_num']*config['embedding_size'], config['embedding_size']])))
        self.n_nodes = self.n_users + self.n_items
        self.topkk= config['topkk']
        self.head_num=config['head_num']
        self.model_cat_rate=config['model_cat_rate']
        # load dataset info
        self.inter=dataset.inter_matrix(form='coo')[1]
        self.norm_adj = self.get_norm_adj_mat(dataset.inter_matrix(form='coo')[0].astype(np.float32)).to(self.device)
        self.user_embedding_t = nn.Embedding(self.n_users, self.embedding_dim)
        self.user_embedding_v = nn.Embedding(self.n_users, self.embedding_dim)
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)
        #self.transformer =nn.Transformer(d_model=self.embedding_dim)
        self.predictor = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.reg_loss = EmbLoss()
        
        nn.init.xavier_normal_(self.predictor.weight)

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
            nn.init.xavier_normal_(self.image_trs.weight)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)
            nn.init.xavier_normal_(self.text_trs.weight)
    def get_norm_adj_mat(self, interaction_matrix):
        A = sp.dok_matrix((self.n_users + self.n_items,
                           self.n_users + self.n_items), dtype=np.float32)
        inter_M = interaction_matrix
        print("self.n_users:")
        inter_M_t = interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users),
                             [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col),
                                  [1] * inter_M_t.nnz)))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid Devide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        return torch.sparse.FloatTensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))
    def multi_head_self_attention(self, embedding_t_1, embedding_t):  
       
        q = embedding_t
        v = k = embedding_t_1
        beh, N, d_h = q.shape[0], q.shape[1],  self.embedding_dim/self.head_num

        Q = torch.matmul(q, self.w_q)  
        K = torch.matmul(k, self.w_k)
        V = v

        Q = Q.reshape(beh, N, self.head_num, int(d_h)).permute(2, 0, 1, 3)  
        K = Q.reshape(beh, N,self.head_num, int(d_h)).permute(2, 0, 1, 3)

        Q = torch.unsqueeze(Q, 2) 
        K = torch.unsqueeze(K, 1)  
        V = torch.unsqueeze(V, 1)  

        att = torch.mul(Q, K) / torch.sqrt(torch.tensor(d_h))  
        att = torch.sum(att, dim=-1) 
        att = torch.unsqueeze(att, dim=-1)  
        att = F.softmax(att, dim=2)  

        Z = torch.mul(att, V)  
        Z = torch.sum(Z, dim=2)  
        Z_list = [value for value in Z]
        Z = torch.cat(Z_list, -1)
        Z = torch.matmul(Z, self.w_self_attention_cat)
        Z=self.model_cat_rate*F.normalize(Z.mean(0), p=2, dim=1)
        return Z, att.detach()

    def forward(self, interactions):
        h = self.item_id_embedding.weight

        t_h_i=self.text_trs(self.text_embedding.weight)
        v_h_i=self.image_trs(self.image_embedding.weight)
        t_image_adj = build_sim(t_h_i)
        t_image_adj = build_knn_neighbourhood(t_image_adj,self.topkk)
        t_image_adj = compute_normalized_laplacian(t_image_adj)
        v_image_adj = build_sim(v_h_i)
        v_image_adj = build_knn_neighbourhood(v_image_adj, self.topkk)
        v_image_adj = compute_normalized_laplacian(v_image_adj)
        t_h_i=torch.matmul(t_image_adj,t_h_i)
        v_h_i=torch.matmul(v_image_adj,v_h_i)
        u_seq=None

        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        abc=torch.cat([u_g_embeddings,torch.zeros([1,self.embedding_dim]).to(self.device)],0)    
        
        numb=interactions[0].shape[0] 
        matrix=[]
        for i in range(numb):
            matrix.append(self.inter[interactions[0][i]])
        max_len = max((len(l) for l in matrix))
        new_matrix = list(map(lambda l:l + [-1]*(max_len - len(l)), matrix))
        embs=abc[torch.tensor(new_matrix)] 
        embs=embs.permute(1,0,2)
        hidden=torch.zeros((1, numb, self.embedding_dim), requires_grad=True).to(self.device) 
        gru_out, hidden=self.gru(embs, hidden)
        ht = hidden[-1]
        gru_out = gru_out.permute(1, 0, 2)
        return u_g_embeddings, i_g_embeddings + h,ht

    def calculate_loss(self, interactions):
        # online network
        u_online_ori, i_online_ori,a = self.forward(interactions)
        t_feat_online, v_feat_online = None, None
        if self.t_feat is not None:
            t_feat_online = self.text_trs(self.text_embedding.weight)
        if self.v_feat is not None:
            v_feat_online = self.image_trs(self.image_embedding.weight)

        with torch.no_grad():
            u_target, i_target = u_online_ori.clone(), i_online_ori.clone()
            u_target.detach()
            i_target.detach()
            u_target = F.dropout(u_target, self.dropout)
            i_target = F.dropout(i_target, self.dropout)

            if self.t_feat is not None:
                t_feat_target = t_feat_online.clone()
                t_feat_target = F.dropout(t_feat_target, self.dropout)

            if self.v_feat is not None:
                v_feat_target = v_feat_online.clone()
                v_feat_target = F.dropout(v_feat_target, self.dropout)

        u_online, i_online = self.predictor(u_online_ori), self.predictor(i_online_ori)

        users, items = interactions[0], interactions[1]
        u_online = u_online[users, :]
        i_online = i_online[items, :]
        u_target = u_target[users, :]
        i_target = i_target[items, :]

        loss_t, loss_v, loss_tv, loss_vt = 0.0, 0.0, 0.0, 0.0
        if self.t_feat is not None:
            t_feat_online = self.predictor(t_feat_online)
            t_feat_online = t_feat_online[items, :]
            t_feat_target = t_feat_target[items, :]
            loss_t = 1 - cosine_similarity(t_feat_online, i_target.detach(), dim=-1).mean()
            loss_tv = 1 - cosine_similarity(t_feat_online, t_feat_target.detach(), dim=-1).mean()
        if self.v_feat is not None:
            v_feat_online = self.predictor(v_feat_online)
            v_feat_online = v_feat_online[items, :]
            v_feat_target = v_feat_target[items, :]
            loss_v = 1 - cosine_similarity(v_feat_online, i_target.detach(), dim=-1).mean()
            loss_vt = 1 - cosine_similarity(v_feat_online, v_feat_target.detach(), dim=-1).mean()
        loss1 = 1 - cosine_similarity(a, i_target.detach(), dim=-1).mean()
        loss_ui = 1 - cosine_similarity(u_online, i_target.detach(), dim=-1).mean()
        loss_iu = 1 - cosine_similarity(i_online, u_target.detach(), dim=-1).mean()

        return (loss_ui + loss_iu+loss1).mean() + self.reg_weight * self.reg_loss(u_online_ori, i_online_ori) + \
               self.cl_weight * (loss_t + loss_v + loss_tv + loss_vt).mean()

    def full_sort_predict(self, interaction):

        user = interaction[0]
        u_online, i_online,a = self.forward(interaction)
        for i in range(interaction[0].shape[0]):
            u_online[interaction[0][i]]=0.4*a[i]+u_online[interaction[0][i]]

        u_online, i_online = self.predictor(u_online), self.predictor(i_online)
        score_mat_ui = torch.matmul(u_online[user], i_online.transpose(0, 1))
        return score_mat_ui

