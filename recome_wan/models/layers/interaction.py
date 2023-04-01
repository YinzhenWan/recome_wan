from itertools import combinations
from typing import List, Tuple

import torch
from torch import nn
from recome_wan.models.layers.embedding import EmbeddingLayer
from recome_wan.models.utils import get_linear_input, get_dnn_input_dim


class FM(nn.Module):
    """
    FM (Factorization Machine) implementation for PyTorch.

    Args:
        None

    Example:
        fm = FM()
        fm_out = fm(feature_emb)

    References:
        - Zhou, Shuochao, et al. "DeepFM: A Factorization-Machine based Neural Network for CTR Prediction."
          arXiv preprint arXiv:1703.04247 (2017).
    """

    def __init__(self):
        super(FM, self).__init__()

    def forward(self, feature_emb: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward propagation of the FM model.

        Args:
            feature_emb (List[torch.Tensor]): List of feature embeddings. Each feature embedding is of shape
            [Batch, num_feature, embedding_dim].

        Returns:
            torch.Tensor: FM output of shape [Batch, 1].
        """
        # stack and squeeze feature embeddings to shape [Batch, num_feature, embedding_dim]
        feature_emb = torch.stack(feature_emb, dim=1).squeeze(2)

        # sum then square: formula first term
        sum_of_square = torch.sum(feature_emb, dim=1) ** 2

        # square then sum: formula second term
        square_of_sum = torch.sum(feature_emb ** 2, dim=1)

        # FM output
        fm_out = (sum_of_square - square_of_sum) * 0.5

        # sum over feature dimension and reshape to [Batch, 1]
        return torch.sum(fm_out, dim=-1).view(-1, 1)

class LR_Layer(nn.Module):
    def __init__(self, enc_dict: dict):
        """
        Initialize the LR_Layer object.

        Args:
        - enc_dict: a dictionary mapping feature names to their one-hot encoding sizes.
        """
        super(LR_Layer, self).__init__()
        self.enc_dict = enc_dict
        self.emb_layer = EmbeddingLayer(enc_dict=self.enc_dict, embedding_dim=1)
        self.dnn_input_dim = get_dnn_input_dim(self.enc_dict, 1)
        self.fc = nn.Linear(self.dnn_input_dim, 1)

    def forward(self, data: dict):
        """
        Perform forward pass through the LR_Layer.

        Args:
        - data: a dictionary mapping feature names to input tensors.

        Returns:
        - out: an output tensor produced by the LR model.
        """
        sparse_emb = self.emb_layer(data)
        sparse_emb = torch.stack(sparse_emb, dim=1).flatten(1)  # [batch,num_sparse*emb] torch.cat对比torch.stack
        # flatten(start_dim=1) 表示从第一个维度（也就是第二维）开始打平， (a, b, c, ...)，那么通过 flatten(1) 操作之后，
        # 它将被展平成一个形状为 (a, b * c * ...) 的一维张量，其中后面的维度都被拉成了一维。


        # 这行代码用于将多个稀疏张量（sparse tensor）沿着指定的维度进行堆叠，并将结果打平为一个二维张量（2D tensor）。
        # 具体来说，首先将多个稀疏张量通过 torch.stack(sparse_emb, dim=1) 沿着第一个轴进行堆叠。这里 sparse_emb 是一个张量列表，
        # 表示不同的稀疏张量。dim=1 表示在第一个轴进行堆叠。例如，如果 sparse_emb 包含两个大小为 (B, E1) 和 (B, E2) 的稀疏张量，
        # 则 torch.stack(sparse_emb, dim=1) 的结果是一个大小为 (B, 2, E1 or E2) 的三维张量，其中第一个轴大小为 2，
        # 表示堆叠的两个张量。接着，通过 .flatten(1) 将第一维和第二维合并成一个维度。这里 1 表示在第二维上进行打平操作，
        # 即将第二维的大小 2 和第三维的大小 E1 or E2 合并成一个大小为 2*E1 or 2*E2 的新维度。最终的结果是一个二维张量，
        # 大小为 (B, 2*E1 or 2*E2)，表示堆叠后的多个稀疏张量。
        # reshape(-1, x.shape[-1]) 表示将最后一维大小保持不变，将其他维度的元素按照最少的方式重排，在重排后的张量中，
        # 最后一维大小为 x.shape[-1]，而其他维度的大小自动计算。
        # 所以，reshape(-1, x[1]) 表示将张量重排成一个二维张量，第一维大小为 -1，也就是自动计算大小以满足其他维度的要求；
        # 第二维大小为 x[1]，也就是保持不变。这里的 x[1] 表示张量 x 的第二维大小。

        dense_input = get_linear_input(self.enc_dict, data)  # [batch,num_dense]
        dnn_input = torch.cat((sparse_emb, dense_input), dim=1)  # [batch,num_sparse*emb + num_dense]
        out = self.fc(dnn_input)
        return out
        # 这段代码的作用是将稀疏特征和稠密特征合并成一个新的“混合”特征向量，以便于后续的神经网络中进行处理和输出。
        # 具体来说，我们有两个输入特征，一个是由稀疏嵌入层编码得到的稀疏特征 sparse_emb，
        # 另一个是由 get_linear_input 函数转换得到的稠密特征 dense_input。我们需要将它们合并起来。
        # 其中，sparse_emb 是一个 (batch_size, num_sparse * embedding_dim) 的展平稀疏特征向量，
        # dense_input 是一个 (batch_size, num_dense) 的稠密特征向量，num_sparse 和 num_dense
        # 分别表示稀疏特征和稠密特征的维度大小。
        # 为了将这些特征向量合并起来，这段代码首先通过 torch.stack(sparse_emb, dim=1).flatten(1) 将稀疏特征向量展开成一个
        # (batch_size, num_sparse * embedding_dim) 的矩阵。
        # 然后，dense_input 和展平后的稀疏特征向量 sparse_emb 使用 torch.cat 沿着列的方向进行拼接，得到一个新的
        # (batch_size, num_sparse * embedding_dim + num_dense) 的特征矩阵 dnn_input。
        # 最终，dnn_input 将作为神经网络的输入，可以接入后续的神经网络模块进行分类、回归等任务。
        # 这种将稀疏特征和稠密特征拼接成一个新的特征向量的方法在工业界和学术界都得到了广泛的应用。


class CrossLayer(nn.Module):
    """
    This class implements the cross-layer computation of neural network.

    Args:
        input_dim (int): The dimension of input feature.

    Attributes:
        W (nn.Parameter): Weight parameters.
        b (nn.Parameter): Bias parameters.

    """
    def __init__(self, input_dim: int):
        super(CrossLayer, self).__init__()
        self.input_dim = input_dim
        self.W = nn.Parameter(torch.rand(self.input_dim))
        self.b = nn.Parameter(torch.rand(self.input_dim))

    def forward(self, x0: torch.Tensor, xl: torch.Tensor) -> torch.Tensor:
        """
        Computes the output of a cross-layer.

        Args:
            x0 (torch.Tensor): The original input features.
            xl (torch.Tensor): The output of last cross-layer.

        Returns:
            torch.Tensor: The output of the current cross-layer.

        """
        out = torch.bmm(x0.unsqueeze(-1), xl.unsqueeze(1))
        out = torch.matmul(out, self.W)
        out = out + self.b + xl
        return out


class CrossNet(nn.Module):
    """
    This class implements the cross network, which is composed of multiple cross-layers.

    Args:
        input_dim (int): The dimension of input feature.
        num_layers (int): The number of cross-net layers.

    Attributes:
        cross_net_list (nn.ModuleList): A list of cross-layers in the cross network.

    """
    def __init__(self, input_dim: int, num_layers: int):
        super(CrossNet, self).__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.cross_net_list = nn.ModuleList()
        for i in range(self.num_layers):
            self.cross_net_list.append(CrossLayer(input_dim))

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        """
        Computes the output of the cross network.

        Args:
            x0 (torch.Tensor): The original input features.

        Returns:
            torch.Tensor: The output of the cross network.

        """
        xl = x0
        for l in range(self.num_layers):
            xl = self.cross_net_list[l](x0, xl)
        return xl

class SENET_Layer(nn.Module):
    def __init__(self, num_fields: int, reduction_ratio: int = 3):
        """Create a SE block layer.

        Args:
            num_fields (int): The num of fields, which is the dim of input.
            reduction_ratio (int, optional): The reduction ratio for SE block. Defaults to 3.
        """
        super(SENET_Layer, self).__init__()
        reduced_size = max(1, int(num_fields / reduction_ratio))
        self.excitation = nn.Sequential(nn.Linear(num_fields, reduced_size, bias=False),
                                        nn.ReLU(),
                                        nn.Linear(reduced_size, num_fields, bias=False),
                                        nn.Sigmoid())

    def forward(self, feature_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward propagation of input through SE block layer.

        Args:
            feature_emb (torch.Tensor): The input tensor feature embedding in shape of [batch, f, embedding_dim].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple that contains feature_emb with channel attention and channel attention.
        """
        assert len(feature_emb.shape) == 3, "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(feature_emb.shape))
        Z = torch.mean(feature_emb, dim=-1, out=None) # [batch, f, embedding_dim] -> [batch, f]
        A = self.excitation(Z) # Z:[batch, f] -> 中间变量:[batch, reduced_size] -> A:[batch, f]
        V = feature_emb * A.unsqueeze(-1) # feature_emb:[batch, f, embedding_dim] A.unsqueeze(-1):[batch, f, 1]
        return V, A


class BilinearInteractionLayer(nn.Module):
    def __init__(self, num_fields: int, embedding_dim: int, bilinear_type: str = "field_interaction") -> None:
        """
        Define a BilinearInteractionLayer neural network module.

        Args:
            num_fields: An integer indicating the number of input fields.
            embedding_dim: An integer indicating the embedding dimension of input features.
            bilinear_type: A string indicating the bilinear type of the module, which can be either "field_all", "field_each",
            or "field_interaction". Default value is "field_interaction".
        """
        super(BilinearInteractionLayer, self).__init__()
        self.bilinear_type = bilinear_type

        assert self.bilinear_type in ['field_all', 'field_each',
                                      'field_interaction'], f"Unexpected bilinear_type {self.bilinear_type}, expect to be in ['field_all','field_each','field_interaction'] "

        if self.bilinear_type == "field_all":
            # field_all means using a single bilinear layer to interact all feature fields.
            # 使用一个双线性层来交叉所有的特征，将所有特征都看作一个整体进行交叉
            self.bilinear_layer = nn.Linear(embedding_dim, embedding_dim, bias=False)
        elif self.bilinear_type == "field_each":
            # field_each means using a bilinear layer to interact each pair of feature fields.
            # A list of bilinear layers will be used, with a different one for each field.
            # 对于每一个特征，使用一个双线性层来和其他特征交叉。
            self.bilinear_layer = nn.ModuleList([nn.Linear(embedding_dim, embedding_dim, bias=False)
                                                 for i in range(num_fields)])
        elif self.bilinear_type == "field_interaction":
            # field_interaction means using a bilinear layer to interact every possible pair of feature fields.
            # A list of bilinear layers will be used, with a different one for each pair of fields.
            # 使用多个双线性层来交叉每对不同的特征，对于每一对特征，使用一个双线性层来进行交叉。
            self.bilinear_layer = nn.ModuleList([nn.Linear(embedding_dim, embedding_dim, bias=False)
                                                 for i, j in combinations(range(num_fields), 2)])

    def forward(self, feature_emb: torch.Tensor) -> torch.Tensor:
        """
        Apply the bilinear interaction layer to input features.

        Args:
            feature_emb: A float tensor with shape [batch, num_fields, embedding_dim], where batch is the batch size,
            num_fields is the number of feature fields, and embedding_dim is the embedding dimension of the features.

        Returns:
            A float tensor with shape [batch, num_interaction_pairs * embedding_dim], where num_interaction_pairs is the number
            of pairs of feature fields that could be interacted with.
        """
        # feature_emb : [batch, num_fileds, embedding_dim]
        assert len(feature_emb.shape) == 3, "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (
            len(feature_emb.shape))
        feature_emb_list = torch.split(feature_emb, 1, dim=1)
        if self.bilinear_type == "field_all":
            # field_all case.
            # Feature interaction is performed by a single bilinear layer between all pairs of feature fields.
            bilinear_list = [self.bilinear_layer(v_i) * v_j
                             for v_i, v_j in combinations(feature_emb_list, 2)]
        elif self.bilinear_type == "field_each":
            # field_each case.
            # Feature interaction is performed by a different bilinear layer for each pair of feature fields.
            bilinear_list = [self.bilinear_layer[i](feature_emb_list[i]) * feature_emb_list[j]
                             for i, j in combinations(range(len(feature_emb_list)), 2)]
        elif self.bilinear_type == "field_interaction":
            # field_interaction case.
            # A different bilinear layer will be used for each pair of feature fields.
            bilinear_list = [self.bilinear_layer[i](v[0]) * v[1]
                             for i, v in enumerate(combinations(feature_emb_list, 2))]
        return torch.cat(bilinear_list, dim=1)