import torch
from torch.utils.data import Dataset
import torch.utils.data as D
import numpy as np
from collections import defaultdict
from recome_wan.datasets.base_dataset import BaseDataset
class DINDataset(BaseDataset):
    def __init__(self, config, df, enc_dict=None):
        self.config = config
        self.df = df
        self.enc_dict = enc_dict
        self.sparse_cols = ['user_id', 'target_item_id', 'target_categories']
        self.seq_cols = ['hist_item_id', 'hist_categories']
        self.feature_name = self.sparse_cols + self.seq_cols + ['label']

        self.enc_data()

    def enc_data(self):
        # 使用enc_dict对数据进行编码
        self.enc_data_dict = defaultdict(np.array)

        for col in self.sparse_cols:
            self.enc_data_dict[col] = torch.Tensor(np.array(self.df[col].values)).long()

        for col in self.seq_cols:
            self.enc_data_dict[col] = torch.Tensor(np.array(self.df[col].values.tolist())).long()

    def __getitem__(self, index):
        data = defaultdict(np.array)
        for col in self.seq_cols:
            data[col] = self.enc_data_dict[col][index]
        for col in self.sparse_cols:
            data[col] = self.enc_data_dict[col][index]
        if 'label' in self.df.columns:
            data['label'] = torch.Tensor([self.df['label'].iloc[index]]).squeeze(-1)
        return data

    def __len__(self):
        return len(self.df)