from typing import Optional, List
import torch
from loguru import logger
import os
import pandas as pd
import torch.utils.data as D
import time
from recome_wan.utils import get_recall_predict, evaluate_recall


def train_sequence_model(model: torch.nn.Module,
                         train_loader: torch.utils.data.DataLoader,
                         optimizer: torch.optim.Optimizer,
                         device: torch.device,
                         log_rounds: int = 100):
    model.train()

    # Calculate the number of iterations required to complete an epoch
    max_iter = train_loader.dataset.__len__() // train_loader.batch_size

    start_time = time.time()

    # Iterate over the dataset batches
    for idx, data in enumerate(train_loader):
        # Move the data to the device
        for key in data.keys():
            data[key] = data[key].to(device)

        # Forward pass
        output = model(data, is_training=True)
        loss = output['loss']

        # Backward pass
        loss.backward()
        optimizer.step()
        model.zero_grad()

        # Calculate time for iteration and remaining time
        iter_time = time.time() - start_time
        remaining_time = round(((iter_time / (idx + 1)) * (max_iter - idx + 1)) / 60, 2)

        # Log progress
        if idx % log_rounds == 0 and device.type != 'cpu':
            logger.info(f'Iter {idx}/{max_iter} Remaining time:{remaining_time} min Loss:{round(float(loss.detach().cpu().numpy()), 4)} GPU Mem:{get_gpu_usage(device)}')
        elif idx % log_rounds == 0:
            logger.info(f'Iter {idx}/{max_iter} Remaining time:{remaining_time} min Loss:{round(float(loss.detach().cpu().numpy()), 4)} ')

def test_sequence_model(model: torch.nn.Module,
                        test_loader: torch.utils.data.DataLoader,
                        device: torch.device,
                        topk_list: List[int] = [20, 50, 100]) -> dict:
    # Set the model in evaluation mode
    model.eval()

    # Get test ground truth
    test_gd = test_loader.dataset.get_test_gd()

    # Get sequence model's prediction for top N values
    preds = get_recall_predict(model, test_loader, device, topN=200)

    # Calculate recall metrics for each top k value
    metric_dict = {}
    for i, k in enumerate(topk_list):
        temp_metric_dict = evaluate_recall(preds, test_gd, k)
        logger.info(temp_metric_dict)
        metric_dict.update(temp_metric_dict)
    return metric_dict


class SequenceTrainer:
    def __init__(self, model_ckpt_dir: str = './model_ckpt'):

        self.log_df = pd.DataFrame()
        self.model_ckpt_dir = model_ckpt_dir

    def fit(self, model, train_loader, valid_loader: Optional = None, epoch: int = 50, lr: float = 1e-3,
            device: torch.device = torch.device('cpu'), topk_list: Optional[List[int]] = None, log_rounds: int = 100):
        if topk_list is None:
            topk_list = [20, 50, 100]


        # Define the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        model = model.to(device)

        # Training process
        logger.info('Model Starting Training')
        best_epoch = -1
        best_metric = -1
        best_metric_dict = dict()

        for i in range(1, epoch + 1):
            # Train the model
            train_sequence_model(model, train_loader, optimizer=optimizer, device=device, log_rounds=log_rounds)

            # Validate the model
            if valid_loader is not None:
                valid_metric = test_sequence_model(model=model, test_loader=valid_loader, topk_list=topk_list,
                                                   device=device)
                valid_metric['phase'] = 'valid'
                self.log_df = self.log_df.append(valid_metric, ignore_index=True)
                model_str = f'e_{i}'
                self.save_train_model(model, self.model_ckpt_dir, model_str)
                self.log_df.to_csv(os.path.join(self.model_ckpt_dir, 'log.csv'), index=False)

                logger.info(f"Valid Metric:{valid_metric}")

    def evaluate_model(self, model, test_loader, device: torch.device = torch.device('cpu'),
                       topk_list: Optional[List[int]] = None) -> dict:

        if topk_list is None:
            topk_list = [20, 50, 100]
        test_metric = test_sequence_model(model=model, test_loader=test_loader, topk_list=topk_list, device=device)
        test_metric['phase'] = 'test'
        self.log_df = self.log_df.append(test_metric, ignore_index=True)
        self.log_df.to_csv(os.path.join(self.model_ckpt_dir, 'log.csv'), index=False)
        logger.info(f"Test Metric:{test_metric}")
        return test_metric

    def save_model(self, model, model_ckpt_dir: str):

        os.makedirs(model_ckpt_dir, exist_ok=True, mode=0o777)
        save_dict = {'model': model.state_dict()}
        torch.save(save_dict, os.path.join(model_ckpt_dir, 'model.pth'))
        logger.info(f'Model Saved to {model_ckpt_dir}')

    def save_all(self, model, enc_dict, model_ckpt_dir: str):

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
