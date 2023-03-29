from .rank_dataset import RankDataset
from .sequence_dataset import SequenceDataset
import torch.utils.data as D


def get_rank_dataloader(train_df, valid_df, test_df, schema, batch_size = 512*3):

    train_dataset = RankDataset(schema,train_df)
    enc_dict = train_dataset.get_enc_dict()
    valid_dataset = RankDataset(schema, valid_df,enc_dict=enc_dict)
    test_dataset = RankDataset(schema, test_df,enc_dict=enc_dict)

    train_loader = D.DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=0, pin_memory=True)
    valid_loader = D.DataLoader(valid_dataset,batch_size=batch_size,shuffle=False,num_workers=0, pin_memory=True)
    test_loader = D.DataLoader(test_dataset,batch_size=batch_size,shuffle=False,num_workers=0, pin_memory=True)

    return train_loader, valid_loader, test_loader, enc_dict

def get_seq_dataloader(train_df, valid_df, test_df, schema, batch_size = 512*3):

    train_dataset = SequenceDataset(schema,train_df)
    enc_dict = train_dataset.get_enc_dict()
    valid_dataset = SequenceDataset(schema, valid_df,enc_dict=enc_dict,phase='test')
    test_dataset = SequenceDataset(schema, test_df,enc_dict=enc_dict,phase='test')

    train_loader = D.DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=0, pin_memory=True)
    valid_loader = D.DataLoader(valid_dataset,batch_size=batch_size,shuffle=False,num_workers=0, pin_memory=True)
    test_loader = D.DataLoader(test_dataset,batch_size=batch_size,shuffle=False,num_workers=0, pin_memory=True)

    return train_loader, valid_loader, test_loader, enc_dict