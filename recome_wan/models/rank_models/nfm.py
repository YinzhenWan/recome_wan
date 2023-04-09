from torch import nn
import torch
from typing import List
from recome_wan.models.layers import EmbeddingLayer, MLP_Layer,InnerProductLayer,LR_Layer
from recome_wan.models.utils import get_linear_input, get_dnn_input_dim


class NFM(nn.Module):
    def __init__(self, feature_map, embedding_dim: int = 10, hidden_units: List[int] = [64, 64, 64],
                 product_type="inner",
                 loss_fun: str = 'torch.nn.BCELoss()', enc_dict: dict = None):
        super(NFM, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_units = hidden_units
        self.loss_fun = eval(loss_fun)
        self.enc_dict = enc_dict

        self.embedding_layer = EmbeddingLayer(enc_dict=self.enc_dict,
                                              embedding_dim=self.embedding_dim)  # Embedding layer
        self.lr = LR_Layer(enc_dict=enc_dict)  # Linear-Layer layer

        self.bi_pooling_layer = InnerProductLayer(output="Bi_interaction_pooling")

        self.dnn_input_dim = get_dnn_input_dim(self.enc_dict, self.embedding_dim)
        # sparse_num*emb_dim + dense_num

        self.dnn = MLP_Layer(input_dim=self.dnn_input_dim, output_dim=1, hidden_units=self.hidden_units,
                             hidden_activations='relu', dropout_rates=0)  # MLP layer

    def forward(self, data: dict, is_training: bool = True) -> dict:


        # lr
        data += self.lr(data)  # linear contribution
        fea_emb = self.embedding_layer(data)


        # DNN
        bi_pooling_vec = self.bi_pooling_layer(fea_emb)
        dnn_output = self.dnn(bi_pooling_vec)  # MLP output


        # Output
        y_pred = torch.sigmoid(data + dnn_output)  # Final prediction
        if is_training:
            loss = self.loss_fun(y_pred.squeeze(-1), data['label'])  # Compute loss
            output_dict = {'pred': y_pred, 'loss': loss}
        else:
            output_dict = {'pred': y_pred}
        return output_dict