import os
import torch
from torch import nn

def set_device(gpu=-1):
    if gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu}")
    else:
        device = torch.device("cpu")
    return device


def get_feature_num(enc_dict):
    num_sparse = 0
    num_dense = 0
    for col in enc_dict.keys():
        if 'min' in enc_dict[col].keys():
            num_dense += 1
        elif 'vocab_size' in enc_dict[col].keys():
            num_sparse += 1
    return num_sparse, num_dense


def set_activation(activation):
    if isinstance(activation, str):
        if activation.lower() == "relu":
            return nn.ReLU()
        elif activation.lower() == "sigmoid":
            return nn.Sigmoid()
        elif activation.lower() == "tanh":
            return nn.Tanh()
        else:
            return getattr(nn, activation)()  # 获取对象的属性值或方法。

    else:
        return activation


def get_dnn_input_dim(enc_dict, embedding_dim):
    num_sparse = 0
    num_dense = 0
    for col in enc_dict.keys():
        if 'min' in enc_dict[col].keys():
            num_dense += 1
        elif 'vocab_size' in enc_dict[col].keys():
            num_sparse += 1
    return num_sparse * embedding_dim + num_dense
# 函数返回num_sparse * embedding_dim + num_dense，这个值就是神经网络的输入维度。
# 其中，num_sparse * embedding_dim表示离散型特征编码后的总维度，num_dense表示连续型特征的维度


def get_linear_input(enc_dict, data):
    res_data = []
    for col in enc_dict.keys():
        if 'min' in enc_dict[col].keys():
            res_data.append(data[col])
    res_data = torch.stack(res_data, axis=1)
    return res_data
# 函数的功能是将输入数据中的所有列按照最小值进行堆叠
# 函数返回处理后的数据，是一个二维的tensor矩阵，其中每一行表示输入数据中的一条记录，每一列表示一个特征的编码结果。

def get_feature_num(enc_dict): # 计算的是离散特征和连续特征的数量
    num_sparse = 0
    num_dense = 0
    for col in enc_dict.keys():
        if 'min' in enc_dict[col].keys():
            num_dense+=1
        elif 'vocab_size' in enc_dict[col].keys():
            num_sparse+=1
    return num_sparse,num_dense
def get_features_num(enc_dict):   #din 里面需要 计算的是物品特征和用户特征的数量
    num_user = 0
    num_item = 0
    for col in enc_dict.keys():
        if 'vocab_size' in enc_dict[col].keys():
            if enc_dict[col]['type']=='item':
                num_item +=1
            else:
                num_user +=1
    return num_user,num_item