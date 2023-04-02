# recome_wan
>> * 1.本项目通过pytorch 框架复现了十篇经典的推荐算法论文，其中包含四篇排序论文和六篇推荐召回论文。<br>
>> * 2.本项目一共包含demo和recome_wan这两个文件夹。<br>
>> * 3.在demo里面分别选了一个召回模型和排序模型来作为示例，如果想调试其他的召回和排序模型，可以直接修改demo里面的rank_example.py文件或者recall_example.py的代码即可。<br>
>> * 4.在recome_wan这个文件夹里，一共包含datasets、models、trainer、utils这四个大的模块。<br> 
>> * 5.其中datasets文件夹主要是数据类型和数据编码的处理，models里面包含了layers、rank_models和recall_models这三个文件夹。layers主要存放的是一些通用的层比如embedding层、Mlp层。
> rank_models里存放的就是排序相关的模型，recall_models里存放的就是召回相关的模型。trainer主要是用来训练、验证、测试召回和排序的模型。utils包含一些关于模型的评价。


| model                         | paper                       | note                               |
| -------------| ------------- | ------------ |
| dcn   | [DCN:Deep & Cross Network for Ad Click Predictions](https://arxiv.org/abs/1708.05123)  | 单元格3   |
| deepfm   | [DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](https://www.ijcai.org/proceedings/2017/0239.pdf)   | 单元格6   |
| fibinet  |[FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction](https://arxiv.org/pdf/1905.09433.pdf)  | 单元格6   |
| mmoe  | [MMOE:Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts](https://dl.acm.org/doi/abs/10.1145/3219819.3220007) | 单元格6   |
| line   | [LINE: Large-scale Information Network Embedding](https://arxiv.org/pdf/1503.03578.pdf)   | [LINE：算法原理，实现和应用](https://zhuanlan.zhihu.com/p/56478167)   |
| comirec   | [COMIREC:Controllable Multi-Interest Framework for Recommendation](https://arxiv.org/pdf/2005.09347.pdf)   | 单元格6   |
| gru4rec  | [GRU4REC:Session-based Recommendations with Recurrent Neural Networks](https://arxiv.org/abs/1511.06939) | 单元格6   |
| mind  | [MIND:Multi-interest network with dynamic routing for recommendation at Tmall](https://arxiv.org/pdf/1904.08030.pdf) | 单元格6   |
| yotubednn   | [YOTUBEDNN:Deep Neural Networks for YouTube Recommendations](https://www.researchgate.net/publication/307573656_Deep_Neural_Networks_for_YouTube_Recommendations)  | 单元格6   |
| lightgcn  | [LIGHTGCN: Simplifying and Powering Graph Convolution Network for Recommendation](https://arxiv.org/abs/2002.02126)   | 单元格6   |
