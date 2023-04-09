import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as D
from torch.nn.init import xavier_normal_,constant_
import dgl.function as fn
from dgl import AddSelfLoop
from dgl import DGLGraph,graph

import dgl
from torch.utils.data import Dataset
from tqdm import tqdm
import faiss
from sklearn.preprocessing import normalize
import multiprocessing

import pandas as pd
import math
import numpy as np
config = {
    'train_path':'../data/gowalla/train.csv',
    'test_path':'../data/gowalla/test.csv',
    'num_user': 29858,
    'num_item': 40981,
    'embedding_dim':64,
    'lr':1e-3,
    'batch':20,
    'epoch':400,
    'device':1
}
# 我们的数据是只有正样本的！！！
train_df = pd.read_csv(config['train_path'])
test_df = pd.read_csv(config['test_path'])


class GeneralGraphDataset(Dataset):
    def __init__(self, df, num_user, num_item, phase='train'):
        self.df = df
        self.n_item = self.df['item_id'].nunique()
        self.phase = phase
        self.num_user = num_user # 用户数量
        self.num_item = num_item  # 物品数量
        self.generate_test_gd()
        if self.phase == 'train':
            self.encode_data()

    def encode_data(self):
        '''
        模型训练的时候，读数据和模型传播是串行的。

        对于“小模型”而言，模型传播的耗时很小，这个时候数据读取变成了主导训练效率的主要原因，
        所以这里通过提前提前将数据转换成tensor来加速数据读取
        '''
        self.data = dict()
        self.data['user_id'] = torch.Tensor(np.array(self.df['user_id'])).long()
        self.data['item_id'] = torch.Tensor(np.array(self.df['item_id'])).long()

    def generate_test_gd(self):
        self.test_gd = self.df.groupby('user_id')['item_id'].apply(list).to_dict()
        # groupby() 函数将训练数据按照用户ID进行分组，并且将每个用户喜欢的物品ID以列表形式存储在分组后的DataFrame中。
        # to_dict() 函数将DataFrame转换为字典对象，并将其存储在类实例的 test_gd 变量中。最后，将所有用户的ID存储在 user_list 列表中
        # # 生成的 test_gd 字典对象的键是每个用户的ID，值是一个列表，包含了该用户喜欢的物品ID
        self.user_list = list(self.test_gd.keys())


    def generate_graph(self):
        '''
        左闭右开
        user_id 范围: 0~num_user
        item_id 范围: num_user~num_user+num_item

        LightGCN构图没有添加自环边
        '''
        src_node_list = torch.cat([self.data['user_id'], self.data['item_id'] + self.num_user], axis=0)
        # 源节点
        dst_node_list = torch.cat([self.data['item_id'] + self.num_user, self.data['user_id']], axis=0)
        # 目标节点
        # 它们的长度等于训练数据中用户节点和物品节点的数量之和。这些张量可以用于构造一个二部图，以便在推荐系统中进行节点嵌入学习。
        g = graph((src_node_list, dst_node_list))

        src_degree = g.out_degrees().float()  # 代表着所有user和item的邻居个数:[num_user+num_item,1]
        # 计算了 g 中每个节点的出度
        norm = torch.pow(src_degree, -0.5).unsqueeze(1)  # compute norm
        # 计算了每个节点的标准化因子，以便后续对节点的嵌入进行标准化处理
        # unsqueeze(1) 方法会在第二个维度上增加一个新的维度
        g.ndata['norm'] = norm  # 节点粒度的norm

        edge_weight = norm[src_node_list] * norm[dst_node_list]  # 边标准化系数
        # 为了计算边权重，首先使用索引操作 norm[src_node_list] 和 norm[dst_node_list]
        # 分别取出源节点和目标节点的标准化因子，然后将它们逐元素相乘，即可得到每条边的权重。
        g.edata['edge_weight'] = edge_weight  # 边粒度norm
        return g

    def __getitem__(self, index):
        if self.phase == 'train':
            random_index = np.random.randint(0, self.num_item)
            while random_index in self.test_gd[self.df['user_id'].iloc[index]]:
                # 这行代码的作用是避免在生成负样本时，随机选择的物品已经在测试集中被评分过了，
                # 避免了负样本与测试集中的正样本过于相似的问题
                random_index = np.random.randint(0, self.num_item)
                # 判断变量 random_index 是否在 self.test_gd[self.df['user_id'].iloc[index]] 中，如果在其中，
                # 就继续生成一个新的随机整数，直到找到一个不在其中的整数。
            neg_item_id = torch.Tensor([random_index]).squeeze().long()
            # 这行代码创建了一个张量 neg_item_id，它的值是 random_index，并将其转换为一个形状为 (1,) 的张量（即行向量），
            # 然后通过 squeeze 方法将其维度降为 ( )，即将其转换为一个标量，最后将其转换为整数类型的张量（即索引）。

            data = {
                'user_id': self.data['user_id'][index],
                'pos_item_id': self.data['item_id'][index],
                'neg_item_id': neg_item_id
            }
        return data

    def __len__(self):
        if self.phase == 'train':
            return len(self.df)
        else:
            return self.df['user_id'].nunique() # 返回 df 中不同用户的数量

train_dataset = GeneralGraphDataset(train_df,num_user=config['num_user'],num_item=config['num_item'],phase='train')
test_dataset = GeneralGraphDataset(test_df,num_user=config['num_user'],num_item=config['num_item'],phase='test')


class LightGCNConv(nn.Module):
    def __init__(self):
        super(LightGCNConv, self).__init__()

    def message_fun(self, edges):
        return {'m': edges.src['h'] * edges.src['norm'] * edges.dst['norm']}
    # 其中键为 'm'，值为 edges.src['h'] * edges.src['norm'] * edges.dst['norm']。在这个表达式中，
    # edges.src['h'] 是源节点的特征向量，edges.src['norm'] 是源节点的归一化系数，edges.dst['norm']
    # 是目标节点的归一化系数。这个函数的作用是将源节点的特征向量乘上源节点和目标节点的归一化系数，然后作为消息传递给目标节点。

    def forward(self, g, ego_embedding):
        '''
        input ego_embedding : [num_user+num_item,emb_dim]
        output h : [num_user+num_item,hidden_dim]
        '''
        g.ndata['h'] = ego_embedding # 将节点的嵌入ego_embedding存入二部图中，并计算每条边的权重
        # 可以通过 g.ndata[key] 来获取特定节点特征 key 的值，或通过 g.ndata[key] = value 来设置节点特征 key 的值
        # 尽量使用DGL自带的message fun和reduce fun！！！
        g.update_all(message_func=fn.u_mul_e('h', 'edge_weight', 'm'),
                     reduce_func=fn.sum(msg="m", out="h"))
        # 对所有节点的嵌入进行消息传递和汇聚
        # 使用了u_mul_e()方法作为消息传递函数，表示将源节点的嵌入与边权重的乘积作为消息进行传递。
        # 同时，使用了sum()方法作为汇聚函数，将所有收到的消息进行加和得到节点的新嵌入。
        # 'h'表示节点特征，'edge_weight'表示边权重，'m'表示计算得到的消息。这个函数的作用是将源节点特征（即'h'）
        # 乘以边权重（即'edge_weight'）得到一条边上的消息（即'm'）。
        # fn.sum(msg="m", out="h")定义了聚合函数。它将所有收到的消息（即'm'）按照目标节点进行求和，
        # 并将结果存储在目标节点的特征中（即'h'）
        #         g.update_all(message_func=self.message_fun,
        #                      reduce_func=fn.sum(msg="m",out="h"))

        h = F.normalize(g.ndata['h'], dim=1, p=2)
        #g.ndata是DGL图的节点数据字典，可以存储任何与节点相关的数据，例如特征向量，标签等。
        # 这里g.ndata['h']是指DGL图中所有节点的特征向量，这个特征向量是由上一层的GCN层计算得到的，
        # 作为输入传入这一层的forward函数中。这一行代码的作用是对这些特征向量进行L2归一化，以便于后续的计算。
        # 具体来说，F.normalize()函数就是对张量进行归一化的函数，dim参数指定了在哪个维度上进行归一化，
        # p参数指定了使用哪种范数进行归一化。
        # 在这个例子中，dim=1表示在第二维度上归一化（也就是特征维度），p=2表示使用L2范数。
        #


        return h


def drop_adj(g, node_dropout=0.2):
    row, col = g.edges()
    mask = torch.rand(row.size(0), device=row.device) >= node_dropout
    filter_row, filter_col = row[mask], col[mask]
    g1 = graph((filter_row, filter_col), num_nodes=g.num_nodes())
    g1 = dgl.add_self_loop(g1)

    src_degree = g1.out_degrees().float()
    norm = torch.pow(src_degree, -0.5).unsqueeze(1)  # compute norm
    g1.ndata['norm'] = norm  # 节点粒度的norm

    g1 = g1.to(row.device)

    return g1


class LightGCN(nn.Module):
    def __init__(self, g, num_user, num_item, embedding_dim, num_layers=3,
                 lmbd=1e-4):
        super(LightGCN, self).__init__()
        self.g = g
        self.num_user = num_user
        self.num_item = num_item
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.lmbd = lmbd

        self.user_emb_layer = nn.Embedding(self.num_user, self.embedding_dim)
        self.item_emb_layer = nn.Embedding(self.num_item, self.embedding_dim)

        self.gcn_layer = LightGCNConv()

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = (users * pos_items).sum(1)
        neg_scores = (users * neg_items).sum(1)

        mf_loss = nn.LogSigmoid()(pos_scores - neg_scores).mean()
        mf_loss = -1 * mf_loss

        regularizer = (torch.norm(users) ** 2 + torch.norm(pos_items) ** 2 + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.lmbd * regularizer / users.shape[0]

        return mf_loss + emb_loss
    # 在BPR中，正样本和负样本的得分之间存在差距，而该差距越大，损失函数的值越小。
    # BPR损失函数一般通过对数sigmoid函数来计算两个物品的得分之间的差距，并使用平均对数似然作为目标函数，
    # 从而让模型更好地拟合用户的偏好。
    # 最大化正确推荐的物品的概率，最小化不正确推荐物品的概率
    # 最大化 用户和未交互的物品的得分 与 用户和已经交互的物品的得分的差异

    def get_ego_embedding(self):
        user_emb = self.user_emb_layer.weight
        item_emb = self.item_emb_layer.weight

        return torch.cat([user_emb, item_emb], 0)
    # get_ego_embedding 是一个方法，用于获取当前模型的节点嵌入。在该方法中，首先获取用户和物品的嵌入向量ser_emb 和 item_emb，
    # 然后将它们拼接成一个新的张量并返回，这个张量表示了所有用户和物品的嵌入向量，被称为节点嵌入。

    def forward(self, data, is_training=True):

        ego_embedding = self.get_ego_embedding()  # [num_user+num_item,emb_dim]
        user_embeds = [self.user_emb_layer.weight]
        item_embeds = [self.item_emb_layer.weight]

        hidden = ego_embedding
        for _ in range(self.num_layers):
            hidden = self.gcn_layer(g, hidden)
            temp_user_emb, temp_item_emb = torch.split(hidden, [self.num_user, self.num_item])
            user_embeds.append(temp_user_emb)
            item_embeds.append(temp_item_emb)

        user_embd = torch.stack(user_embeds, 1)
        user_embd = torch.mean(user_embd, 1)

        item_embd = torch.stack(item_embeds, 1)
        item_embd = torch.mean(item_embd, 1)

        output_dict = dict()
        if is_training:
            u_g_embeddings = user_embd[data['user_id'], :]
            pos_i_g_embeddings = item_embd[data['pos_item_id'], :]
            neg_i_g_embeddings = item_embd[data['neg_item_id'], :]
            loss = self.create_bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)  # pair-wise范式的训练方法
            output_dict['loss'] = loss
        else:
            output_dict['user_emb'] = user_embd
            output_dict['item_emb'] = item_embd

        return output_dict
def evaluate_recall(preds,test_gd, topN=50):
    total_recall = 0.0
    total_ndcg = 0.0
    total_hitrate = 0
    for user in test_gd.keys():
        recall = 0
        dcg = 0.0
        item_list = test_gd[user]
        for no, item_id in enumerate(item_list):
            if item_id in preds[user][:topN]:
                recall += 1
                dcg += 1.0 / math.log(no+2, 2)
            idcg = 0.0
            for no in range(recall):
                idcg += 1.0 / math.log(no+2, 2)
        total_recall += recall * 1.0 / len(item_list)
        if recall > 0:
            total_ndcg += dcg / idcg
            total_hitrate += 1
    total = len(test_gd)
    recall = total_recall / total
    ndcg = total_ndcg / total
    hitrate = total_hitrate * 1.0 / total
    return {f'recall@{topN}': round(recall,4), f'ndcg@{topN}': round(ndcg,4), f'hitrate@{topN}': round(hitrate,4)}


# 训练模型，验证模型
def train_graph_model(model, train_loader, optimizer, device):
    model.train()
    pred_list = []
    label_list = []
    pbar = tqdm(train_loader)
    epoch_loss = 0
    for data in pbar:

        for key in data.keys():
            data[key] = data[key].to(device)

        output = model(data)
        loss = output['loss']

        loss.backward()
        optimizer.step()
        model.zero_grad()

        epoch_loss += loss.item()
        pbar.set_description("Loss {}".format(round(epoch_loss, 4)))
    return epoch_loss


def test_graph_model(model, train_gd, test_gd, device, hidden_size, topN=50):
    model.eval()
    output = model(_, is_training=False)
    user_embs = output['user_emb'].detach().cpu().numpy()
    item_embs = output['item_emb'].detach().cpu().numpy()

    test_user_list = list(test_gd.keys())

    faiss_index = faiss.IndexFlatIP(hidden_size)
    faiss_index.add(item_embs)

    preds = dict()

    for i in tqdm(range(0, len(test_user_list), 1000)):
        user_ids = test_user_list[i:i + 1000]
        batch_user_emb = user_embs[user_ids, :]
        D, I = faiss_index.search(batch_user_emb, 1000)

        for i, iid_list in enumerate(user_ids):  # 每个用户的label列表，此处item_id为一个二维list，验证和测试是多label的
            train_items = train_gd.get(user_ids[i], [])
            preds[user_ids[i]] = [x for x in list(I[i, :]) if x not in train_items]
    return evaluate_recall(preds, test_gd, topN=topN)


def set_device(gpu=-1):
    if gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu}")
    else:
        device = torch.device("cpu")
    return device
train_loader = D.DataLoader(train_dataset,batch_size=config['batch'],shuffle=True,num_workers=4)
device = set_device(config['device'])
g = g.to(device)
model = LightGCN(g,config['num_user'],config['num_item'],config['embedding_dim'])
optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
model = model.to(device)

train_gd = train_dataset.test_gd
test_gd = test_dataset.test_gd
#模型训练流程
for i in range(config['epoch']):
    #模型训练
    epoch_loss = train_graph_model(model,train_loader,optimizer=optimizer,device=device)
    print(f'Epoch {i} Total Loss:{epoch_loss}')
    #模型验证
    if i%10==0:

        test_metric = test_graph_model(model,train_gd,test_gd,device,config['embedding_dim'],20)

        print(f"Epoch {i} Test Metric:")
        print(test_metric)