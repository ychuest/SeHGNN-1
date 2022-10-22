# SeHGNN
论文笔记&amp;复现：SeHGNN, 2022, arXiv。

# 论文笔记

## 简介

提出了一种新的==基于元路径==的HGNN。

* 作者发现同一种关系下的==node-level attention是不需要的==，用mean即可。于是作者移除了训练阶段的neighbor aggregation，放在了预处理步骤，这样==只需一次neighbor aggregation==。
* 提出一种新的==transformer-based semantic aggregator==来聚合不同元路径的不同语义信息。

## 方法论

### 第一部分：Simplified Neighbor Aggregation

对于异构图中每个结点，通过==mean==聚合其==基于元路径的邻居==的特征，得到该结点基于该元路径的表征：

<img src="./img/image-20221022152855427.png" alt="image-20221022152855427" style="zoom:50%;" />

其中，$p_{i, ..., j}$表示元路径$P$的一个实例，从结点$i$到结点$j$。

# 论文复现

TODO
