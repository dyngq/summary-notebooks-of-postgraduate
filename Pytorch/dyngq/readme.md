# pytorch

## 产生分布的函数

函数 | 功能
:-: | :-:
tensor.uniform_(-10, 10) | 均匀分布
tensor.normal_(mean, std) | 标准正态分布

!['dyngq_images'](images/dyngq_2020-02-04-23-43-48.png)

## 一些基本操作

函数 | 功能
:-: | :-:
trace | 对角线元素之和(矩阵的迹)
diag | 对角线元素
triu/tril | 矩阵的上三角/下三角，可指定偏移量
mm/bmm | 矩阵乘法，batch的矩阵乘法
addmm/addbmm/addmv/addr/baddbmm.. | 矩阵运算
t | 转置
dot/cross | 内积/外积
inverse | 求逆矩阵
svd | 奇异值分解

PyTorch中的Tensor支持超过一百种操作，包括转置、索引、切片、数学运算、线性代数、随机数等等，可参考[官方文档](https://pytorch.org/docs/stable/tensors.html)。
