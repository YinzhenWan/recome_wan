import torch.nn as nn


class EmbeddingLayer(nn.Module):
    def __init__(self,
                 enc_dict: dict,
                 embedding_dim: int):
        """
        Initialize EmbeddingLayer.

        Args:
        - enc_dict (dict): A dictionary containing the encoding information
          of input features (e.g., vocabulary size).
        - embedding_dim (int): The dimension of output embeddings.
        """
        super(EmbeddingLayer, self).__init__()
        self.enc_dict = enc_dict
        self.embedding_dim = embedding_dim
        self.embedding_layer = nn.ModuleDict()

        self.emb_feature = []

        # Create an Embedding module for each sparse input feature
        # that has the key 'vocab_size' in enc_dict.
        for col in self.enc_dict.keys():
            if 'vocab_size' in self.enc_dict[col].keys():
                self.emb_feature.append(col)
                self.embedding_layer.update({col : nn.Embedding(
                    self.enc_dict[col]['vocab_size'],
                    self.embedding_dim,
                )})

    def forward(self, X: dict) -> list:
        """
        Pass the inputs through the embedding layer.

        Args:
        - X (dict): A dictionary containing the input features. Each value
          is a tensor whose size is (batch_size, feature_dim).

        Returns:
        - feature_emb_list (list): A list of feature embeddings. Each
          element of the list is a tensor whose size is
          (batch_size, feature_dim, embedding_dim).
        """
        # Embed each sparse input feature separately.
        feature_emb_list = []
        for col in self.emb_feature:
            inp = X[col].long().view(-1, 1)
            feature_emb_list.append(self.embedding_layer[col](inp))
        return feature_emb_list