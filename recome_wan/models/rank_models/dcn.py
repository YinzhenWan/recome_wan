from typing import Dict, List
from torch import nn
import torch
from recome_wan.models.layers import CrossNet, EmbeddingLayer, LR_Layer, MLP_Layer
from recome_wan.models.utils import get_linear_input, get_dnn_input_dim


class DCN(nn.Module):
    def __init__(self,
                 embedding_dim: int = 10,
                 hidden_units: list = [64, 64, 64],
                 num_layers: int = 4,
                 loss_fun: str = 'torch.nn.BCELoss()',
                 enc_dict: dict = None) -> None:
        '''
        Deep & Cross Network model implementation.

        Args:
        - embedding_dim (int): embedding dimension
        - hidden_units (list): sizes of hidden layers in neural network
        - num_layers (int): number of layers in CrossNet
        - loss_fun (str): name of loss function in string form
        - enc_dict (dict): dictionary containing feature names and their input size

        '''

        super(DCN, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.loss_fun = eval(loss_fun)  # evalutaing loss function string to get actual function
        self.enc_dict = enc_dict

        # creating layer objects
        self.embedding_layer = EmbeddingLayer(enc_dict=self.enc_dict, embedding_dim=self.embedding_dim)

        self.lr = LR_Layer(enc_dict=enc_dict)  # 一阶

        self.dnn_input_dim = get_dnn_input_dim(self.enc_dict, self.embedding_dim)  # sparse_num * emb_dim + dense_num

        self.dnn = MLP_Layer(input_dim=self.dnn_input_dim, hidden_units=self.hidden_units,
                             hidden_activations='relu', dropout_rates=0)
        self.cross_net = CrossNet(input_dim=self.dnn_input_dim, num_layers=self.num_layers)

        self.fc = nn.Linear(self.dnn_input_dim + self.hidden_units[-1], 1)

    def forward(self, data: dict, is_training: bool = True) -> dict:
        '''
        Forward pass of DCN model.

        Args:
        - data (dict): dictionary containing features and their values

        Returns:
        - output_dict (dict): dictionary containing predicted output and loss

        '''
        sparse_embedding = self.embedding_layer(data)
        emb_flatten = torch.stack(sparse_embedding, dim=1).flatten(start_dim=1)
        dense_input = get_linear_input(self.enc_dict, data)

        lr_logit = self.lr(data)  # 一阶交叉

        x0 = torch.cat((emb_flatten, dense_input), dim=1)  # x0
        # CrossNet
        cross_out = self.cross_net(x0)
        # DNN
        dnn_out = self.dnn(x0)
        # fc
        nn_logit = self.fc(torch.cat([cross_out, dnn_out], axis=-1))

        # 输出
        y_pred = torch.sigmoid(lr_logit + nn_logit)
        if is_training:
            loss = self.loss_fun(y_pred.squeeze(-1), data['label'])
            output_dict = {'pred': y_pred, 'loss': loss}
        else:
            output_dict = {'pred': y_pred}
        return output_dict
