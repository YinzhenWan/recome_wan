from torch import nn
import torch
from typing import List
from recome_wan.models.layers import EmbeddingLayer, MLP_Layer,InnerProductLayer

class PNN(nn.Module):
    # 内积操作指的是将两个向量逐位相乘，然后将结果相加得到一个标量。
    # 在PNN模型中，内积操作被用来计算两个商品向量之间的相似度。
    # 外积操作指的是将两个向量进行外积，得到一个矩阵。在PNN模型中，
    # 外积操作被用来捕捉两个商品之间的交叉特征。具体来说，
    # PNN模型将每个商品的embedding与其他商品的embedding进行外积，
    # 得到一个矩阵，然后对这个矩阵进行池化操作，得到一个表示交叉特征的向量。
    # 内积操作用于计算相似度，外积操作用于捕捉交叉特征。
    def __init__(self, feature_map,embedding_dim: int = 10, hidden_units: List[int] = [64, 64, 64], product_type="inner",
                 loss_fun: str = 'torch.nn.BCELoss()', enc_dict: dict = None):
        super(PNN, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_units = hidden_units
        self.loss_fun = eval(loss_fun)
        self.enc_dict = enc_dict

        self.embedding_layer = EmbeddingLayer(enc_dict=self.enc_dict,
                                              embedding_dim=self.embedding_dim)  # Embedding layer

        if product_type != "inner":
            raise NotImplementedError("product_type={} has not been implemented.".format(product_type))
        self.inner_product_layer = InnerProductLayer(feature_map.num_fields, output="inner_product")
        # PNN模型中外积操作的输出维度，表示所有的交叉特征向量拼接起来的长度
        input_dim = int(feature_map.num_fields * (feature_map.num_fields - 1) / 2) \
                    + feature_map.num_fields * embedding_dim
        self.dnn = MLP_Layer(input_dim=input_dim, output_dim=1, hidden_units=self.hidden_units,
                             hidden_activations='relu', dropout_rates=0)  # MLP layer

    def forward(self, data: dict, is_training: bool = True) -> dict:
        """
        Forward pass of the PNN model.

        Args:
        - data (dict): a dictionary containing the input data. Example format: {"sparse_feature1": tensor_of_categories1, "sparse_feature2": tensor_of_categories2, "dense_feature1": tensor_of_floats1, ...}
        - is_training (bool): set to True when training, False when evaluating the model

        Returns:
        - output_dict (dict): a dictionary containing the predicted values and optionally the value of the loss function. Example format: {"pred": prediction_tensor, "loss": loss_value}
        """
        fea_embedding = self.embedding_layer(data)  # Embedding layer output
        inner_product_vec = self.inner_product_layer(fea_embedding)
        dnn_input = torch.cat([fea_embedding.flatten(start_dim=1), inner_product_vec], dim=1)


        # DNN
        dnn_input = torch.cat((dnn_input), dim=1)  # Concatenate embeddings and dense inputs
        dnn_output = self.dnn(dnn_input)  # MLP output

        # Output
        y_pred = torch.sigmoid(dnn_output)  # Final prediction
        if is_training:
            loss = self.loss_fun(y_pred.squeeze(-1), data['label'])  # Compute loss
            output_dict = {'pred': y_pred, 'loss': loss}
        else:
            output_dict = {'pred': y_pred}
        return output_dict