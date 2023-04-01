import torch
from torch.utils.data import Dataset
import random


class SequenceDataset(Dataset):
    def __init__(self, config, df, enc_dict=None, phase='train'):
        self.config = config
        self.df = df
        self.enc_dict = enc_dict
        self.max_length = self.config['max_length']
        self.user_col = self.config['user_col']
        self.item_col = self.config['item_col']
        self.time_col = self.config.get('time_col', None)

        if self.time_col:
            self.df = self.df.sort_values(by=[self.user_col, self.time_col])
        # 按照时间顺序进行排序

        if self.enc_dict==None:
            self.get_enc_dict()
        self.enc_data()

        self.user2item = self.df.groupby(self.user_col)[self.item_col].apply(list).to_dict()   # key 记录的是用户id,value记录的是用户的行为点击序列itemid的list
        self.user_list = self.df[self.user_col].unique()  #把前7个作为一个训练的数据，第八个作为label
        self.phase = phase

    def get_enc_dict(self):
        #计算enc_dict
        if self.enc_dict==None:
            sparse_cols = [self.item_col]
            self.enc_dict = dict(zip( list(sparse_cols),[dict() for _ in range(len(sparse_cols))]))
            for f in sparse_cols:
                self.df[f] = self.df[f].astype('str')
                map_dict = dict(zip(sorted(self.df[f].unique()), range(1,1+self.df[f].nunique())))
                self.enc_dict[f] = map_dict
                self.enc_dict[f]['vocab_size'] = self.df[f].nunique()+1
        else:
            return self.enc_dict

    def enc_data(self):
        sparse_cols = [self.item_col]
        for f in sparse_cols:
            self.df[f] = self.df[f].astype('str')
            self.df[f] = self.df[f].apply(lambda x:self.enc_dict[f].get(x,0))

    def __getitem__(self, index):
        user_id = self.user_list[index] #获取用户id
        item_list = self.user2item[user_id] #通过用户id 来获取用户对应的itemid序列
        hist_item_list = []
        hist_mask_list = []
        if self.phase == 'train':

            k = random.choice(range(4, len(item_list)))  # 从[4,len(item_list))中随机选择一个index,比如随机选择k这个时间点
            item_id = item_list[k]  # 该index对应的item加入item_id_list
            # 把k之前的20个连续(max_length)当作行为序列，把k当作一个标签label

            if k >= self.max_length:  # 选取max_len个物品
                hist_item_list.append(item_list[k - self.max_length: k])  # 选20个，k=32,max_length=20，从32往前数20个 【32-20=12：32】
                hist_mask_list.append([1.0] * self.max_length) # 全是有效的 全1
            else:
                hist_item_list.append(item_list[:k] + [0] * (self.max_length - k)) # 不够补0 ，比如k =15，max_length=20,要选取20个，把五个补零
                hist_mask_list.append([1.0] * k + [0.0] * (self.max_length - k))  # mask? 序列里面一般用mask 来标记数据是否是有效位 前15个是有效的补1，后面不够的那五个补0
            data = {
                'hist_item_list':torch.Tensor(hist_item_list).squeeze(0).long(),
                'hist_mask_list':torch.Tensor(hist_mask_list).squeeze(0).long(),
                'target_item':torch.Tensor([item_id]).long()
            }
        else:  #测试阶段有很多的方法，但是我给写死了！
            k = int(0.8 * len(item_list))
            if k >= self.max_length:  # 选取seq_len个物品
                hist_item_list.append(item_list[k - self.max_length: k])
                hist_mask_list.append([1.0] * self.max_length)
            else:
                hist_item_list.append(item_list[:k] + [0] * (self.max_length - k))
                hist_mask_list.append([1.0] * k + [0.0] * (self.max_length - k))
            data = {
                'user': user_id,
                'hist_item_list': torch.Tensor(hist_item_list).squeeze(0).long(),
                'hist_mask_list': torch.Tensor(hist_mask_list).squeeze(0).long(),
            }
        return data

    def __len__(self):
        return len(self.user_list)

    def get_test_gd(self):
        self.test_gd = dict()
        for user in self.user2item:
            item_list = self.user2item[user]
            test_item_index = int(0.8 * len(item_list))
            self.test_gd[str(user)] = item_list[test_item_index:]
        return self.test_gd