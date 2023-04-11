# recome_wan
>> * 1.本项目通过pytorch 框架大概复现了推荐系统相关的22个模型，其中包含多篇排序论文和多篇序列召回和图召回论文。<br>
>> * 2.本项目一共包含demo和recome_wan这两个文件夹。<br>
>> * 3.在demo里面分别选了一个召回模型和排序模型来作为示例，如果想调试其他的召回和排序模型，可以直接修改demo里面的rank_example.py文件或者recall_example.py的代码即可。<br>
>> * 4.在recome_wan这个文件夹里，一共包含datasets、models、trainer、utils这四个大的模块。<br> 
>> * 5.其中datasets文件夹主要是数据类型和数据编码的处理，models里面包含了layers、rank_models和recall_models这三个文件夹。layers主要存放的是一些通用的层比如embedding层、Mlp层。
> rank_models里存放的就是排序相关的模型，recall_models里存放的就是召回相关的模型。trainer主要是用来训练、验证、测试召回和排序的模型。utils包含一些关于召回模型的评价。
> 注：以下表格只展示了一些重要的模型，对一些简单的模型不做表格展示。


| model                         | paper                       |
| -------------| ------------- |
| dcn   | [DCN:Deep & Cross Network for Ad Click Predictions](https://arxiv.org/abs/1708.05123)  |
| deepfm   | [DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](https://www.ijcai.org/proceedings/2017/0239.pdf)   |
| fibinet  |[FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction](https://arxiv.org/pdf/1905.09433.pdf)  | 
| mmoe  | [MMOE:Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts](https://dl.acm.org/doi/abs/10.1145/3219819.3220007) |
| line   | [LINE: Large-scale Information Network Embedding](https://arxiv.org/pdf/1503.03578.pdf)   | 
| comirec   | [COMIREC:Controllable Multi-Interest Framework for Recommendation](https://arxiv.org/pdf/2005.09347.pdf)   | 
| gru4rec  | [GRU4REC:Session-based Recommendations with Recurrent Neural Networks](https://arxiv.org/abs/1511.06939) |
| mind  | [MIND:Multi-interest network with dynamic routing for recommendation at Tmall](https://arxiv.org/pdf/1904.08030.pdf) |
| youtubednn   | [YOUTUBEDNN:Deep Neural Networks for YouTube Recommendations](https://www.researchgate.net/publication/307573656_Deep_Neural_Networks_for_YouTube_Recommendations)  |
| lightgcn  | [LightGcn: Simplifying and Powering Graph Convolution Network for Recommendation](https://arxiv.org/abs/2002.02126)   |
| deepwalk  | [DeepWalk: Online Learning of Social Representations](http://www.perozzi.net/publications/14_kdd_deepwalk.pdf)   | 
| node2vec  | [Node2Vev: Scalable Feature Learning for Networks](https://www.kdd.org/kdd2016/papers/files/rfp0218-groverA.pdf)   |
| din  | [DIN:Deep Interest Network for Click-Through Rate Prediction](https://arxiv.org/abs/1706.06978)   | 
| sdne  | [SDNE:Structural Deep Network Embedding](https://arxiv.org/abs/1706.06978)   |    
|graphsage  | [GRAPHSAGE:Inductive Representation Learning on Large Graphs](https://zhuanlan.zhihu.com/p/62750137)   |
|eges  | [EGES:Enhanced Graph Embedding with Side Information](https://www.bilibili.com/video/av899559357/?vd_source=34c01d8d4f1a113a7ed5aa95bd04d882)   |  
