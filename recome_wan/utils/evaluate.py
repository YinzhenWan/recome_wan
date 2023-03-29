from typing import Dict, List
import math
import numpy as np
import faiss
import torch
from sklearn.preprocessing import normalize


def get_recall_predict(model: torch.nn.Module,
                       test_data: torch.utils.data.DataLoader,
                       device: torch.device,
                       topN: int = 20) -> dict:
    # Get the item embeddings and add them to Faiss index.
    item_embs = model.output_items().cpu().detach().numpy()
    item_embs = normalize(item_embs, norm='l2').astype('float32')
    hidden_size = item_embs.shape[1]
    faiss_index = faiss.IndexFlatIP(hidden_size)
    faiss_index.add(item_embs)

    # Iterate through all users in the test data and get their recommendations.
    preds = dict()
    for data in test_data:
        for key in data.keys():
            if key == 'user':
                continue
            data[key] = data[key].to(device)

        # Get user embeddings for the given data.
        model.eval()
        user_embs = model(data, is_training=False)['user_emb']
        user_embs = user_embs.cpu().detach().numpy().astype('float32')

        user_list = data['user'].cpu().numpy()

        # Get the recommendations using Faiss index.

        if len(user_embs.shape) == 2:

            # Non-multi-interest model.
            user_embs = normalize(user_embs, norm='l2').astype('float32')
            D, I = faiss_index.search(user_embs, topN)

            for i, user in enumerate(user_list):
                preds[str(user)] = I[i, :]

        else:

            # Multi-interest model.
            ni = user_embs.shape[1]
            user_embs = np.reshape(user_embs,
                                   [-1, user_embs.shape[-1]])
            user_embs = normalize(user_embs, norm='l2').astype('float32')
            D, I = faiss_index.search(user_embs, topN)
            for i, user in enumerate(user_list):
                item_list_set = []
                item_list = list(zip(np.reshape(I[i * ni:(i + 1) * ni], -1),
                                     np.reshape(D[i * ni:(i + 1) * ni], -1)))
                item_list.sort(key=lambda x: x[1], reverse=True)
                for j in range(len(item_list)):
                    if item_list[j][0] not in item_list_set and item_list[j][0] != 0:
                        item_list_set.append(item_list[j][0])
                        if len(item_list_set) >= topN:
                            break
                preds[str(user)] = item_list_set
    return preds


def evaluate_recall(preds: Dict[str, List[int]],
                    test_gd: Dict[str, List[int]],
                    topN: int = 50) -> Dict[str, float]:
    total_recall = 0.0
    total_ndcg = 0.0
    total_hitrate = 0

    # Iterate over each user in the test data
    for user in test_gd.keys():
        if user not in preds.keys():
            continue
        recall = 0
        dcg = 0.0
        item_list = test_gd[user]

        # Iterate over each actual item in the test data
        for no, item_id in enumerate(item_list):
            if item_id in preds[user][:topN]:
                # Increment recall for each correctly predicted item
                recall += 1
                # Calculate dcg
                dcg += 1.0 / math.log(no+2, 2)

            # Calculate idcg
            idcg = 0.0
            for no in range(recall):
                idcg += 1.0 / math.log(no+2, 2)

        # Calculate total recall, total ndcg, and total hitrate
        total_recall += recall * 1.0 / len(item_list)
        if recall > 0:
            total_ndcg += dcg / idcg
            total_hitrate += 1

    # Calculate overall recall, overall ndcg, and overall hitrate
    total = len(test_gd)
    recall = total_recall / total
    ndcg = total_ndcg / total
    hitrate = total_hitrate * 1.0 / total

    # Return a dictionary containing results
    return {f'recall@{topN}': round(recall,4), f'ndcg@{topN}': round(ndcg,4), f'hitrate@{topN}': round(hitrate,4)}
