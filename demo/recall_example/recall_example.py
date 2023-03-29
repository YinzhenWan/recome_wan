# -*- ecoding: utf-8 -*-
# @ModuleName: run_sequence_example
# @Author: wyatt
# @Email: 670973122@qq.com
# @Time: 2023/3/5 18:00
import sys
sys.path.append('../../')
import torch
from recome_wan.datasets import get_seq_dataloader
from recome_wan.models.recall_models.sequence_models import ComirecSA, ComirecDR, GRU4Rec, MIND, YotubeDNN
from recome_wan.trainer import SequenceTrainer
import pandas as pd

if __name__=='__main__':
    #声明数据schema
    schema = {
        'user_col': 'user_id',
        'item_col': 'item_id',
        'cate_cols': ['genre'],
        'max_length': 20,
        'time_col': 'timestamp',
        'task_type':'sequence'
    }
    # 模型配置
    config = {
        'embedding_dim': 64,
        'lr': 0.001,
        'K': 1,
        'device':torch.device('cpu'),
    }
    config.update(schema)

    #样例数据
    train_df = pd.read_csv('./sample_data/sample_train.csv')
    valid_df = pd.read_csv('./sample_data/sample_valid.csv')
    test_df = pd.read_csv('./sample_data/sample_test.csv')

    #声明使用的device
    device = torch.device('cpu')
    #获取dataloader
    train_loader, valid_loader, test_loader, enc_dict = get_seq_dataloader(train_df, valid_df, test_df, schema, batch_size=50)
    #声明模型,序列召回模型模型目前支持： ComirecSA,ComirecDR,MIND,CMI,Re4,NARM,YotubeDNN,SRGNN
    model = GRU4Rec(enc_dict=enc_dict,config=config)
    #声明Trainer
    trainer = SequenceTrainer(model_ckpt_dir='./model_ckpt')
    #训练模型
    trainer.fit(model, train_loader, valid_loader, epoch=500, lr=1e-3, device=device,log_rounds=10)
    #保存模型权重和enc_dict
    trainer.save_all(model, enc_dict, './model_ckpt')
    #模型验证
    test_metric = trainer.evaluate_model(model, test_loader, device=device)