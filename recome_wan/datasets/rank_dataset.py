import pandas as pd
import torch
from typing import Dict
from recome_wan.datasets.base_dataset import BaseDataset


class RankDataset(BaseDataset):

    def __init__(self, config: dict, df: pd.DataFrame, enc_dict: Dict[str, dict] = None):
        super(RankDataset, self).__init__(config, df, enc_dict)
        self.df = self.df.rename(columns={self.config['label_col']: 'label'})

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        data = {}

        for col in self.dense_cols:
            data[col] = self.data_dict[col][index]
        for col in self.sparse_cols:
            data[col] = self.data_dict[col][index]
        if 'label' in self.df.columns:
            data['label'] = torch.Tensor([self.df['label'].iloc[index]]).squeeze(-1)

        return data