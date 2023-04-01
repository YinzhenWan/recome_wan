from torch import nn
import torch
from typing import Dict, Union, List
import numpy as np
from recome_wan.models.layers import LR_Layer, MLP_Layer, EmbeddingLayer, SENET_Layer, BilinearInteractionLayer
from recome_wan.models.utils import get_linear_input, get_feature_num


class FiBiNET(nn.Module):
    def __init__(self,
                 embedding_dim: int = 32,
                 hidden_units: List[int] = [64, 64, 64],
                 reduction_ratio: int = 3,
                 bilinear_type: str = 'field_interaction',
                 loss_fun: str = 'torch.nn.BCELoss()',
                 enc_dict: dict = None) -> None:
        """
        This class implements the FiBiNET model (Feature Interaction Bidirectional Encoder Network).
        Args:
            embedding_dim (int): Dimensionality of the embedding space.
            hidden_units (List[int]): A list of integers representing the number of units in each hidden layer of the MLP.
            reduction_ratio (int): A reduction ratio used in the SENET layer.
            bilinear_type (str): Type of bilinear interaction, i.e., 'field_interaction' or 'fieldwise_interaction'.
            loss_fun (str): A string representation of the loss function to be used.
            enc_dict (dict): A dictionary containing information about the training data such as the number of unique values for each categorical feature.
        """
        super(FiBiNET, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_units = hidden_units
        self.reduction_ratio = reduction_ratio
        self.bilinear_type = bilinear_type
        self.loss_fun = eval(loss_fun)
        self.enc_dict = enc_dict

        self.embedding_layer = EmbeddingLayer(enc_dict=self.enc_dict, embedding_dim=self.embedding_dim)
        num_sparse, num_dense = get_feature_num(self.enc_dict)

        self.senet = SENET_Layer(num_fields=num_sparse, reduction_ratio=self.reduction_ratio)
        self.bilinear = BilinearInteractionLayer(num_fields=num_sparse,
                                                 embedding_dim=self.embedding_dim,
                                                 bilinear_type=self.bilinear_type)
        self.lr = LR_Layer(enc_dict=enc_dict)  # 一阶

        # (n-1)*n /2 * embedding_dim + (n-1)*n /2 * embedding_dim + num_dense
        self.dnn_input_dim = num_sparse * (num_sparse - 1) * self.embedding_dim + num_dense
        # sparse_num * emb_dim + dense_num

        self.dnn = MLP_Layer(input_dim=self.dnn_input_dim, output_dim=1, hidden_units=self.hidden_units,
                             hidden_activations='relu', dropout_rates=0)

    def get_senet_weights(self, data: Dict[str, torch.Tensor]) -> np.ndarray:
        """
        Compute the weights from the SENET layer for a given batch of data.
        Args:
            data (Dict[str, torch.Tensor]): A dictionary containing the input data.
        Returns:
            np.ndarray: An array of the weights computed by the SENET layer.
        """
        sparse_embedding = self.embedding_layer(data)
        sparse_embedding = torch.stack(sparse_embedding, 1).squeeze(2)

        _, A = self.senet(sparse_embedding)
        return A.detach().cpu().numpy()

    def forward(self, data: Dict[str, torch.Tensor], is_training: bool = True) -> Dict[str, Union[torch.Tensor, float]]:
        """
        Forward pass through the FiBiNET model.
        Args:
            data (Dict[str, torch.Tensor]): A dictionary containing the input data.
            is_training (bool): A boolean flag indicating whether the model is currently training.
        Returns:
            Dict[str, Union[torch.Tensor, float]]: A dictionary containing the model's output.
        """
        sparse_embedding = self.embedding_layer(data)
        sparse_embedding = torch.stack(sparse_embedding, 1).squeeze(2)  # [batch,num_sparse,embedding_dim] 将稀疏矩阵的所有嵌入向量沿垂直方向叠加形成batch_size个矩阵。
        dense_input = get_linear_input(self.enc_dict, data)

        lr_logit = self.lr(data)

        # SENET
        senet_embedding, _ = self.senet(sparse_embedding)

        # Bilinear-Interaction
        p = self.bilinear(sparse_embedding)  # [batch, (n-1)*n/2 ,embedding_dim]
        q = self.bilinear(senet_embedding)  # [batch, (n-1)*n/2 ,embedding_dim]

        # Combination Layer
        c = torch.flatten(torch.cat([p, q], dim=1),
                          start_dim=1)  # [batch, (n-1)*n/2 ,embedding_dim] -> [batch, (n-1)*n/2 * embedding_dim]
        dnn_input = torch.cat((c, dense_input), dim=1)  # [batch, dnn_input_dim]
        # DNN
        dnn_logit = self.dnn(dnn_input)

        # 输出
        y_pred = torch.sigmoid(lr_logit + dnn_logit)
        if is_training:
            loss = self.loss_fun(y_pred.squeeze(-1), data['label'])  # Compute loss
            output_dict = {'pred': y_pred, 'loss': loss}
        else:
            output_dict = {'pred': y_pred}
        return output_dict

