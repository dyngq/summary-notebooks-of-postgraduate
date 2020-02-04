# pytorch

## Tips

* Parameter类其实是Tensor的子类

## 产生分布的函数

函数 | 功能
:-: | :-:
tensor.uniform_(-10, 10) | 均匀分布
tensor.normal_(mean, std) | 标准正态分布

!['dyngq_images'](dyngq/images/dyngq_2020-02-04-23-43-48.png)

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

## Ques

1. log_softmax比softmax多计算一次log，意义在于 加快计算速度，数值上也更稳定。 参考资料：[PyTorch学习笔记——softmax和log_softmax的区别、CrossEntropyLoss() 与 NLLLoss() 的区别、log似然代价函数](https://blog.csdn.net/hao5335156/article/details/80607732)

2. [Pytorch中torch.nn.Softmax的dim参数含义](https://blog.csdn.net/sunyueqinghit/article/details/101113251) 就是在第几维上 sum=1

3. tf.nn.softmax中dim默认为-1,即,tf.nn.softmax会以最后一个维度作为一维向量计算softmax 注意：tf.nn.softmax函数默认（dim=-1）是对张量最后一维的shape=(p,)向量进行softmax计算，得到一个概率向量。不同的是,tf.nn.sigmoid函数对一个张量的每一个标量元素求得一个概率。也就是说tf.nn.softmax默认针对1阶张量进行运算,可以通过指定dim来针对1阶以上的张量进行运算,但不能对0阶张量进行运算。而tf.nn.sigmoid是针对0阶张量,。 !['dyngq_images'](images/dyngq_2019-12-27-20-25-40.png) 参考资料：[tensorflow中交叉熵系列函数](https://zhuanlan.zhihu.com/p/27842203)

4. ？？？？ python 深拷贝、浅拷贝

5. mean std(标准差) !['dyngq_images'](images/dyngq_2019-12-27-21-14-02.png)

6. ？？？？ numpy.triu torch.from_numpy !['dyngq_images'](images/dyngq_2019-12-27-21-35-01.png)

7. ？？？？ 负的维度的使用 !['dyngq_images'](images/dyngq_2019-12-27-21-36-24.png)

8. ？？？？ torch.view .transpose

9. ？？？？ 标签平滑 KL散度评价 !['dyngq_images'](images/dyngq_2019-12-27-21-48-40.png)

10. !['dyngq_images'](images/dyngq_2019-12-28-11-25-48.png)
