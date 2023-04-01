import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict
from collections import defaultdict


class BaseDataset(Dataset):
    """
    The dataset class for deep learning models

    Args:
        config: A dictionary containing the configuration parameters
        df: A Pandas DataFrame consists of the dataset
        enc_dict: A dictionary containing the encoding maps

    Attributes:
        config: A dictionary containing the configuration parameters
        df: A Pandas DataFrame consists of the dataset
        enc_dict: A dictionary containing the encoding maps
        dense_cols: A list of dense feature columns
        sparse_cols: A list of sparse feature columns
        feature_name: A list of feature names
        data_dict: A dictionary containing the encoded data

    Methods:
        get_enc_dict: Builds the encoding maps for categorical features
        enc_dense_data: Encodes numeric data
        enc_sparse_data: Encodes categorical data
        enc_data: Encodes all feature columns in the dataset
        __len__: Returns the length of the dataset
    """

    def __init__(self, config: dict, df: pd.DataFrame, enc_dict: Dict[str, dict] = None):
        self.config = config
        self.df = df
        self.enc_dict = enc_dict
        self.dense_cols = list(set(self.config['dense_cols']))
        self.sparse_cols = list(set(self.config['sparse_cols']))
        self.feature_name = self.dense_cols + self.sparse_cols

        if self.enc_dict is None:
            self.get_enc_dict()

        self.enc_data()

    def get_enc_dict(self) -> Dict[str, dict]:
        """
        Builds the encoding maps for categorical features

        Returns:
            A dictionary containing the encoding maps for all categorical features
        """
        self.enc_dict = dict(zip(
            list(self.dense_cols + self.sparse_cols), [dict() for _ in range(len(self.dense_cols + self.sparse_cols))]))

        for f in self.sparse_cols:
            self.df[f] = self.df[f].astype('str')
            map_dict = dict(zip(sorted(self.df[f].unique()), range(1, 1 + self.df[f].nunique())))
            self.enc_dict[f] = map_dict
            self.enc_dict[f]['vocab_size'] = self.df[f].nunique() + 1  #为了将未出现过的特征值映射为0

        for f in self.dense_cols:
            self.enc_dict[f]['min'] = self.df[f].min()
            self.enc_dict[f]['max'] = self.df[f].max()

        return self.enc_dict

    def enc_dense_data(self, col: str) -> torch.Tensor:
        """
        Encodes numeric data

        Args:
            col: A string of dense feature column

        Returns:
            A torch.Tensor of encoded numeric data
        """
        return (self.df[col] - self.enc_dict[col]['min']) / (
                self.enc_dict[col]['max'] - self.enc_dict[col]['min'] + 1e-5)

    def enc_sparse_data(self, col: str) -> torch.Tensor:
        """
        Encodes categorical data

        Args:
            col: A string of sparse feature column

        Returns:
            A torch.Tensor of encoded categorical data
        """
        return self.df[col].apply(lambda x: self.enc_dict[col].get(x, 0))

    def enc_data(self):
        """
        Encodes all feature columns in the dataset
        """
        self.data_dict = defaultdict(np.array)

        for col in self.dense_cols:
            self.data_dict[col] = torch.Tensor(np.array(self.enc_dense_data(col)))
        for col in self.sparse_cols:
            self.data_dict[col] = torch.Tensor(np.array(self.enc_sparse_data(col))).long()

    def __len__(self) -> int:
        """
        Returns the length of the dataset

        Returns:
            An integer of the length of the dataset
        """
        return len(self.df)