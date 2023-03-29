from typing import Optional
import torch
from loguru import logger
import os
from recome_wan.datasets import RankDataset
import torch.utils.data as D
import time
from sklearn.metrics import roc_auc_score, log_loss


def train_model(model: torch.nn.Module,
                train_loader: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer,
                device: torch.device,
                log_rounds: int = 100) -> dict:
    model.train()
    max_iter = int(train_loader.dataset.__len__() / train_loader.batch_size)

    pred_list = []
    label_list = []
    start_time = time.time()
    for idx,data in enumerate(train_loader):

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

        auc = round(roc_auc_score(label_list[-1000:], pred_list[-1000:]), 4)

        iter_time = time.time() - start_time
        remaining_time = round(((iter_time / (idx+1)) * (max_iter - idx + 1)) / 60, 2)

        if idx % log_rounds == 0 and device.type != 'cpu':
            logger.info(f'Iter {idx}/{max_iter} Remaining time:{remaining_time} min Loss:{round(float(loss.detach().cpu().numpy()), 4)} AUC:{auc} GPU Mem:{get_gpu_usage(device)}')
        elif idx % log_rounds == 0:
            logger.info(f'Iter {idx}/{max_iter} Remaining time:{remaining_time} min Loss:{round(float(loss.detach().cpu().numpy()), 4)} AUC:{auc}')
    res_dict = dict()
    res_dict[f'log_loss'] = round(log_loss(label_list, pred_list, eps=1e-7), 4)
    res_dict[f'roc_auc_score'] = round(roc_auc_score(label_list, pred_list), 4)
    return res_dict

def test_model(model: torch.nn.Module,
               test_loader: torch.utils.data.DataLoader,
               device: torch.device) -> dict:
    model.eval()

    pred_list = []
    label_list = []

    # Iterate over the test set
    for data in test_loader:

        # Move the data to the device
        for key in data.keys():
            data[key] = data[key].to(device)

        # Forward pass
        output = model(data)
        pred = output['pred']

        # Append predictions and labels to corresponding lists
        pred_list.extend(pred.squeeze(-1).cpu().detach().numpy())
        label_list.extend(data['label'].squeeze(-1).cpu().detach().numpy())

    # Initialize the results dictionary
    res_dict = dict()
    res_dict[f'log_loss'] = round(log_loss(label_list, pred_list, eps=1e-7), 4)
    res_dict[f'roc_auc_score'] = round(roc_auc_score(label_list, pred_list), 4)

    return res_dict

class RankTrainer:
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
