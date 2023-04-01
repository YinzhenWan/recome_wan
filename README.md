# recome_wan
>> * 1.本项目通过pytorch 框架复现了十篇经典的推荐算法论文，其中包含四篇排序论文和六篇推荐召回论文。<br>
>> * 2.本项目一共包含demo和recome_wan这两个文件夹。<br>
>> * 3.在demo里面分别选了一个召回模型和排序模型来作为示例，如果想调试其他的召回和排序模型，可以直接修改demo里面的rank_example.py文件或者recall_example.py的代码即可。<br>
>> * 4.在recome_wan这个文件夹里，一共包含datasets、models、trainer、utils这四个大的模块。<br> 
>> * 5.其中datasets文件夹主要是数据类型和数据编码的处理，models里面包含了layers、rank_models和recall_models这三个文件夹。layers主要存放的是一些通用的层比如embedding层、Mlp层。
> rank_models里存放的就是排序相关的模型，recall_models里存放的就是召回相关的模型。trainer主要是用来训练、验证、测试召回和排序的模型。utils包含一些关于模型的评价。

