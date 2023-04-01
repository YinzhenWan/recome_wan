import torch
from torch.utils.data import Dataset
import torch.utils.data as D
from torch import nn
import pandas as pd
import numpy as np
import copy
import os
from tqdm import tqdm
from typing import Optional
import torch
from loguru import logger
from recome_wan.datasets import RankDataset
import torch.utils.data as D
from sklearn.metrics import roc_auc_score, log_loss

def train_model(model, train_loader, optimizer, device, metric_list=['roc_auc_score','log_loss'], num_task =1):
    model.train()
    if num_task == 1:
        pred_list = []
        label_list = []
        pbar = tqdm(train_loader)
        for data in pbar:

            for key in data.keys():
                data[key] = data[key].to(device)

            output = model(data)
            pred = output['pred']
            loss = output['loss']

            loss.backward()
            optimizer.step()
            model.zero_grad()

            pred_list.extend(pred.squeeze(-1).cpu().detach().numpy())
            label_list.extend(data['label'].squeeze(-1).cpu().detach().numpy())
            pbar.set_description("Loss {}".format(loss))

        res_dict = dict()
        for metric in metric_list:
            if metric =='log_loss':
                res_dict[f'train_{metric}'] = log_loss(label_list,pred_list, eps=1e-7)
            else:
                res_dict[f'train_{metric}'] = eval(metric)(label_list,pred_list)

        return
    else:
        multi_task_pred_list = [[] for _ in range(num_task)]
        multi_task_label_list = [[] for _ in range(num_task)]
        pbar = tqdm(train_loader)
        for data in pbar:

            for key in data.keys():
                data[key] = data[key].to(device)

            output = model(data)
            loss = output['loss']

            loss.backward()
            optimizer.step()
            model.zero_grad()
            for i in range(num_task):
                multi_task_pred_list[i].extend(list(output[f'task{i + 1}_pred'].squeeze(-1).cpu().detach().numpy()))
                multi_task_label_list[i].extend(list(data[f'task{i + 1}_label'].squeeze(-1).cpu().detach().numpy()))
            pbar.set_description("Loss {}".format(loss))

        res_dict = dict()
        for i in range(num_task):
            for metric in metric_list:
                if metric == 'log_loss':
                    res_dict[f'train_task{i+1}_{metric}'] = log_loss(multi_task_label_list[i], multi_task_pred_list[i], eps=1e-7)
                else:
                    res_dict[f'train_task{i+1}_{metric}'] = eval(metric)(multi_task_label_list[i], multi_task_pred_list[i])
        return res_dict

def valid_model(model, valid_loader, device, metric_list=['roc_auc_score','log_loss'],num_task =1):
    model.eval()
    if num_task == 1:
        pred_list = []
        label_list = []
        for data in tqdm(valid_loader):

            for key in data.keys():
                data[key] = data[key].to(device)

            output = model(data)
            pred = output['pred']

            pred_list.extend(pred.squeeze(-1).cpu().detach().numpy())
            label_list.extend(data['label'].squeeze(-1).cpu().detach().numpy())

        res_dict = dict()
        for metric in metric_list:
            if metric =='log_loss':
                res_dict[f'valid_{metric}'] = log_loss(label_list,pred_list, eps=1e-7)
            else:
                res_dict[f'valid_{metric}'] = eval(metric)(label_list,pred_list)

        return res_dict
    else:
        multi_task_pred_list = [[] for _ in range(num_task)]
        multi_task_label_list = [[] for _ in range(num_task)]
        for data in valid_loader:

            for key in data.keys():
                data[key] = data[key].to(device)

            output = model(data)

            for i in range(num_task):
                multi_task_pred_list[i].extend(list(output[f'task{i + 1}_pred'].squeeze(-1).cpu().detach().numpy()))
                multi_task_label_list[i].extend(list(data[f'task{i + 1}_label'].squeeze(-1).cpu().detach().numpy()))

        res_dict = dict()
        for i in range(num_task):
            for metric in metric_list:
                if metric == 'log_loss':
                    res_dict[f'valid_task{i+1}_{metric}'] = log_loss(multi_task_label_list[i], multi_task_pred_list[i], eps=1e-7)
                else:
                    res_dict[f'valid_task{i+1}_{metric}'] = eval(metric)(multi_task_label_list[i], multi_task_pred_list[i])
        return res_dict

def test_model(model, test_loader, device, metric_list=['roc_auc_score','log_loss'],num_task =1):
    model.eval()
    if num_task == 1:
        pred_list = []
        label_list = []
        for data in tqdm(test_loader):

            for key in data.keys():
                data[key] = data[key].to(device)

            output = model(data)
            pred = output['pred']

            pred_list.extend(pred.squeeze(-1).cpu().detach().numpy())
            label_list.extend(data['label'].squeeze(-1).cpu().detach().numpy())

        res_dict = dict()
        for metric in metric_list:
            if metric == 'log_loss':
                res_dict[f'test_{metric}'] = log_loss(label_list, pred_list, eps=1e-7)
            else:
                res_dict[f'test_{metric}'] = eval(metric)(label_list, pred_list)

        return res_dict
    else:
        multi_task_pred_list = [[] for _ in range(num_task)]
        multi_task_label_list = [[] for _ in range(num_task)]
        for data in test_loader:

            for key in data.keys():
                data[key] = data[key].to(device)

            output = model(data)

            for i in range(num_task):
                multi_task_pred_list[i].extend(list(output[f'task{i + 1}_pred'].squeeze(-1).cpu().detach().numpy()))
                multi_task_label_list[i].extend(list(data[f'task{i + 1}_label'].squeeze(-1).cpu().detach().numpy()))

        res_dict = dict()
        for i in range(num_task):
            for metric in metric_list:
                if metric == 'log_loss':
                    res_dict[f'test_task{i + 1}_{metric}'] = log_loss(multi_task_label_list[i], multi_task_pred_list[i],
                                                                 eps=1e-7)
                else:
                    res_dict[f'test_task{i + 1}_{metric}'] = eval(metric)(multi_task_label_list[i], multi_task_pred_list[i])
        return res_dict

class MultiTask_Trainer:
    def __init__(self, model_ckpt_dir: str = './model_ckpt'):
        self.model_ckpt_dir = model_ckpt_dir

    def fit(self, model, train_loader, valid_loader: Optional = None, epoch: int = 10, lr: float = 1e-3,
            device: torch.device = torch.device('cpu')):
        # Declare the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        model = model.to(device)
        logger.info('Model Starting Training ')
        for i in range(1, epoch + 1):
            # Model training
            train_metric = train_model(model, train_loader, optimizer=optimizer, device=device)
            logger.info(f"Train Metric:{train_metric}")
            # Model validation
            if valid_loader is not None:
                valid_metric = test_model(model, valid_loader, device)
                model_str = f'e_{i}'
                self.save_train_model(model, self.model_ckpt_dir, model_str)
                logger.info(f"Valid Metric:{valid_metric}")
        return valid_metric

    def save_model(self, model, model_ckpt_dir: str):

        os.makedirs(model_ckpt_dir, exist_ok=True, mode=0o777)
        save_dict = {'model': model.state_dict()}
        torch.save(save_dict, os.path.join(model_ckpt_dir, 'model.pth'))
        logger.info(f'Model Saved to {model_ckpt_dir}')

    def save_all(self, model, enc_dict: dict, model_ckpt_dir: str):
        os.makedirs(model_ckpt_dir, exist_ok=True, mode=0o777)
        save_dict = {'model': model.state_dict(),
                     'enc_dict': enc_dict}
        torch.save(save_dict, os.path.join(model_ckpt_dir, 'model.pth'))
        logger.info(f'Enc_dict and Model Saved to {model_ckpt_dir}')

    def save_train_model(self, model, model_ckpt_dir: str, model_str: str):
        os.makedirs(model_ckpt_dir, exist_ok=True, mode=0o777)
        save_dict = {'model': model.state_dict()}
        torch.save(save_dict, os.path.join(model_ckpt_dir, f'model_{model_str}.pth'))
        logger.info(f'Model Saved to {model_ckpt_dir}')

    def evaluate_model(self, model, test_loader, device: torch.device = torch.device('cpu')):
        test_metric = test_model(model, test_loader, device)
        logger.info(f"Test Metric:{test_metric}")
        return test_metric

    def predict_dataloader(self, model, test_loader, device: torch.device = torch.device('cpu')):
        model.eval()
        pred_list = []
        for data in test_loader:
            for key in data.keys():
                data[key] = data[key].to(device)
            output = model(data, is_training=False)
            pred = output['pred']
            pred_list.extend(pred.squeeze(-1).cpu().detach().numpy())
        return pred_list

    def predict_dataframe(self, model, test_df, enc_dict: dict, schema: dict,
                          device:torch.device = torch.device('cpu'), batch_size: int = 1024):
        test_dataset = RankDataset(schema, test_df, enc_dict=enc_dict)
        test_loader = D.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        return self.predict_dataloader(model, test_loader, device=device)