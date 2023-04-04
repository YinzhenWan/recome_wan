#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as D
from torch.nn.init import xavier_normal_,constant_
import dgl.function as fn
from dgl import AddSelfLoop
from dgl import DGLGraph,graph
from dgl.nn import GraphConv
import dgl
from torch.utils.data import Dataset
from tqdm import tqdm
import faiss
from sklearn.preprocessing import normalize
import multiprocessing

import pandas as pd
import math
import numpy as np
from recome_wan.datasets.base_dataset import BaseDataset
class GeneralGraphDataset(BaseDataset):
    def __init__(self, df, num_user, num_item, phase='train'):
        self.df = df
        self.n_item = self.df['item_id'].nunique()
        self.phase = phase
        self.num_user = num_user
        self.num_item = num_item
        self.generate_test_gd()
        if self.phase == 'train':
            self.encode_data()

    def encode_data(self):
        '''
        模型训练的时候，读数据和模型传播是串行的。

        对于“小模型”而言，模型传播的耗时很小，这个时候数据读取变成了主导训练效率的主要原因，
        所以这里通过提前提前将数据转换成tensor来加速数据读取
        '''
        self.data = dict()
        self.data['user_id'] = torch.Tensor(np.array(self.df['user_id'])).long()
        self.data['item_id'] = torch.Tensor(np.array(self.df['item_id'])).long()

    def generate_test_gd(self):
        self.test_gd = self.df.groupby('user_id')['item_id'].apply(list).to_dict()
        self.user_list = list(self.test_gd.keys())

    def generate_graph(self):
        '''
        左闭右开
        user_id 范围: 0~num_user
        item_id 范围: num_user~num_user+num_item

        LightGCN构图没有添加自环边
        '''
        src_node_list = torch.cat([self.data['user_id'], self.data['item_id'] + self.num_user], axis=0)
        dst_node_list = torch.cat([self.data['item_id'] + self.num_user, self.data['user_id']], axis=0)
        g = graph((src_node_list, dst_node_list))

        src_degree = g.out_degrees().float()  # 代表着所有user和item的邻居个数:[num_user+num_item,1]
        norm = torch.pow(src_degree, -0.5).unsqueeze(1)  # compute norm
        g.ndata['norm'] = norm  # 节点粒度的norm

        edge_weight = norm[src_node_list] * norm[dst_node_list]  # 边标准化系数
        g.edata['edge_weight'] = edge_weight  # 边粒度norm
        return g

    def __getitem__(self, index):
        if self.phase == 'train':
            random_index = np.random.randint(0, self.num_item)
            while random_index in self.test_gd[self.df['user_id'].iloc[index]]:
                random_index = np.random.randint(0, self.num_item)
            neg_item_id = torch.Tensor([random_index]).squeeze().long()

            data = {
                'user_id': self.data['user_id'][index],
                'pos_item_id': self.data['item_id'][index],
                'neg_item_id': neg_item_id
            }
        return data

    def __len__(self):
        if self.phase == 'train':
            return len(self.df)
        else:
            return self.df['user_id'].nunique()


class LightGCNConv(nn.Module):
    def __init__(self):
        super(LightGCNConv, self).__init__()

    def message_fun(self, edges):
        return {'m': edges.src['h'] * edges.src['norm'] * edges.dst['norm']}

    def forward(self, g, ego_embedding):
        '''
        input ego_embedding : [num_user+num_item,emb_dim]
        output h : [num_user+num_item,hidden_dim]
        '''
        g.ndata['h'] = ego_embedding
        # 尽量使用DGL自带的message fun和reduce fun！！！
        g.update_all(message_func=fn.u_mul_e('h', 'edge_weight', 'm'),
                     reduce_func=fn.sum(msg="m", out="h"))
        #         g.update_all(message_func=self.message_fun,
        #                      reduce_func=fn.sum(msg="m",out="h"))

        h = F.normalize(g.ndata['h'], dim=1, p=2)

        return h