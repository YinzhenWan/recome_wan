import sys
sys.path.append('../../')
import torch
from recome_wan.datasets import get_rank_dataloader
from recome_wan.models.rank_models import DeepFM, DCN, FiBiNET
from recome_wan.trainer import RankTrainer
import pandas as pd


if __name__=='__main__':
    df = pd.read_csv('sample_data/ranking_sample_data.csv')
    #声明数据schema
    schema={
        "sparse_cols":['user_id','item_id','item_type','dayofweek','is_workday','city','county',
                      'town','village','lbs_city','lbs_district','hardware_platform','hardware_ischarging',
                      'os_type','network_type','position'],
        "dense_cols" : ['item_expo_1d','item_expo_7d','item_expo_14d','item_expo_30d','item_clk_1d',
                       'item_clk_7d','item_clk_14d','item_clk_30d','use_duration'],
        "label_col":'click',
        'task_type': 'ranking'
    }
    #准备数据,这里只选择了100条数据,所以没有切分数据集
    train_df = df[:80]
    valid_df = df[:90]
    test_df = df[:95]

    #声明使用的device
    device = torch.device('cpu')
    #获取dataloader
    train_loader, valid_loader, test_loader, enc_dict = get_rank_dataloader(train_df, valid_df, test_df, schema, batch_size=512)
    #声明模型
    model = FiBiNET(enc_dict=enc_dict)
    #声明Trainer
    trainer = RankTrainer(model_ckpt_dir='./model_ckpt')
    #训练模型
    trainer.fit(model, train_loader, valid_loader, epoch=500, lr=1e-3, device=device)
    #保存模型权重和enc_dict
    trainer.save_all(model, enc_dict, './model_ckpt')
    #模型验证
    test_metric = trainer.evaluate_model(model, test_loader, device=device)
    # #测试 predict_dataframe
    y_pre_dataftame = trainer.predict_dataframe(model, test_df, enc_dict, schema)
