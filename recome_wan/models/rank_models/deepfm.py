from torch import nn
import torch
from typing import List
from recome_wan.models.layers import FM, LR_Layer, MLP_Layer, EmbeddingLayer
from recome_wan.models.utils import get_linear_input, get_dnn_input_dim


class DeepFM(nn.Module):
    """
    A PyTorch module implementing the DeepFM model.
    """

    def __init__(self, embedding_dim: int = 10, hidden_units: List[int] = [64, 64, 64],
                 loss_fun: str = 'torch.nn.BCELoss()', enc_dict: dict = None):
        """
        Initializes the DeepFM model.

        Args:
        - embedding_dim (int): dimensionality of the feature embeddings
        - hidden_units (List[int]): sizes of the hidden units in the MLP layer
        - loss_fun (str): name of the loss function used to optimize the model
        - enc_dict (dict): a dictionary of the form {"sparse_feature1": num_categories1, "sparse_feature2": num_categories2, "dense_feature1": None, ...}
        """
        super(DeepFM, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_units = hidden_units
        self.loss_fun = eval(loss_fun)
        self.enc_dict = enc_dict

        self.embedding_layer = EmbeddingLayer(enc_dict=self.enc_dict,
                                              embedding_dim=self.embedding_dim)  # Embedding layer

        self.fm = FM()  # FM layer
        self.lr = LR_Layer(enc_dict=enc_dict)  # Linear-Layer layer

        self.dnn_input_dim = get_dnn_input_dim(self.enc_dict, self.embedding_dim)
        # sparse_num*emb_dim + dense_num

        self.dnn = MLP_Layer(input_dim=self.dnn_input_dim, output_dim=1, hidden_units=self.hidden_units,
                             hidden_activations='relu', dropout_rates=0)  # MLP layer

    def forward(self, data: dict, is_training: bool = True) -> dict:
        """
        Forward pass of the DeepFM model.

        Args:
        - data (dict): a dictionary containing the input data. Example format: {"sparse_feature1": tensor_of_categories1, "sparse_feature2": tensor_of_categories2, "dense_feature1": tensor_of_floats1, ...}
        - is_training (bool): set to True when training, False when evaluating the model

        Returns:
        - output_dict (dict): a dictionary containing the predicted values and optionally the value of the loss function. Example format: {"pred": prediction_tensor, "loss": loss_value}
        """
        sparse_embedding = self.embedding_layer(data)  # Embedding layer output
        emb_flatten = torch.stack(sparse_embedding, dim=1).flatten(start_dim=1)  # Flatten the embeddings
        dense_input = get_linear_input(self.enc_dict, data)  # Linear input

        # FM
        lr_logit = self.lr(data)  # linear contribution
        fm_out = self.fm(sparse_embedding)  # FM contribution
        fm_out += lr_logit  # Final FM output

        # DNN
        dnn_input = torch.cat((emb_flatten, dense_input), dim=1)  # Concatenate embeddings and dense inputs
        dnn_output = self.dnn(dnn_input)  # MLP output

        # Output
        y_pred = torch.sigmoid(fm_out + dnn_output)  # Final prediction
        if is_training:
            loss = self.loss_fun(y_pred.squeeze(-1), data['label'])  # Compute loss
            output_dict = {'pred': y_pred, 'loss': loss}
        else:
            output_dict = {'pred': y_pred}
        return output_dict