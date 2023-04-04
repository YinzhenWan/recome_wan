import torch
from torch.utils.data import Dataset
import torch.utils.data as D
from torch import nn
import pandas as pd
import numpy as np
import copy
import os
from sklearn.metrics import roc_auc_score,log_loss
from tqdm import tqdm

from models.layers.interaction import DINAttentionLayer
from recome_wan.models.layers import FM, LR_Layer, MLP_Layer, EmbeddingLayer
from recome_wan.models.utils import get_linear_input, get_dnn_input_dim, get_features_num


# DIN模型
class DIN(nn.Module):
    def __init__(self,
                 hidden_units=[64, 32, 16], # 第一个隐藏层有 64 个神经元，第二个隐藏层有 32 个神经元，第三个隐藏层有 16 个神经元
                 attention_units=[32], # 用于构建自注意力层中的MLP层，该层将注意力输入的高维度特征映射到一维空间。因此，在作者实现的模型中，attention_units的单元数是 [32]。
                 embedding_dim=10,
                 loss_fun='torch.nn.BCELoss()',
                 enc_dict=None,
                 count_map=None):
        super(DIN, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_units = hidden_units
        self.attention_units = attention_units
        self.loss_fun = eval(loss_fun) # 可以将其转换为实际的损失函数对象。
        self.enc_dict = enc_dict
        self.count_map = count_map  # count_map是一个记录每种用户行为发生次数的映射。

        self.embedding_layer = EmbeddingLayer(enc_dict=self.enc_dict, embedding_dim=self.embedding_dim)

        self.num_user, self.num_item = get_features_num(enc_dict)

        self.din_att = DINAttentionLayer(embedding_dim=self.embedding_dim * self.num_item,
                                         attention_units=self.attention_units)

        self.mlp = MLP_Layer(input_dim=self.embedding_dim * (2 * self.num_item + self.num_user),
                             hidden_units=self.hidden_units, output_dim=1)

    def MBA_Reg(self, data, col):
        feature_id = torch.unique(data[col])  # 获取去重之后的ID
        feature_emb_list = []
        for f in feature_id:
            feature_emb_list.append(
                self.embedding_layer.emb_layer[col](f) ** 2 / self.count_map[col][int(f.detach().cpu().numpy())]) #numpy()则是将tensor转换为numpy数组
        feature_emb = torch.cat(feature_emb_list, dim=0).mean(0) #拼起来求均值
        return feature_emb.mean()  # 1

    def l2_reg(self, data):
        user_mba_l2 = self.MBA_Reg(data, 'user_id')
        item_mba_l2 = self.MBA_Reg(data, 'target_item_id')
        cate_mba_l2 = self.MBA_Reg(data, 'target_categories')

        mba_l2 = user_mba_l2 + item_mba_l2 + cate_mba_l2
        return mba_l2

    def forward(self, data):
        user_emb, query_emb, his_item_emb = self.embedding_layer(data)

        din_out = self.din_att(query_emb, his_item_emb)  # [batch,emb*2]

        mlp_input = torch.cat([user_emb, din_out, query_emb], dim=-1)
        mlp_out = self.mlp(mlp_input)
        y_pred = torch.sigmoid(mlp_out)

        loss = self.loss_fun(y_pred.squeeze(-1), data['label']) + 0.2 * self.l2_reg(data)
        output_dict = {'pred': y_pred, 'loss': loss}

        return output_dict