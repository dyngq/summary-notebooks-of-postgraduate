# Word2Vector

!['dyngq_images'](images/dyngq_2019-10-09-16-47-42.png)

## 入门NLP课程

!['dyngq_images'](images/dyngq_2019-10-09-16-50-10.png)

### 语言模型

!['dyngq_images'](images/dyngq_2019-10-09-16-51-01.png)
!['dyngq_images'](images/dyngq_2019-10-09-17-02-29.png)

有些词组合在一起的概率是很小很小的，所以会比较稀疏，有很多是很小没有太大意义的,并且造成参数空间太大。
!['dyngq_images'](images/dyngq_2019-10-09-17-03-55.png)

#### 马尔科夫假设

本状态至于前边的状态有关

基于马尔科夫假设，提出：

下一个词的出现仅依赖于它前面的一个词或几个词

!['dyngq_images'](images/dyngq_2019-10-09-17-14-53.png)

于是提出n-gram模型：
!['dyngq_images'](images/dyngq_2019-10-09-20-19-23.png)
n取1、2、3，称为unigram bigram trigram（一元二元三元语法）
n同样可以取的更大

[(四)N-gram语言模型与马尔科夫假设](https://blog.csdn.net/hao5335156/article/details/82730983)
!['dyngq_images'](images/dyngq_2019-10-09-20-06-15.png)
!['dyngq_images'](images/dyngq_2019-10-09-21-03-42.png)

### 词向量

#### One-Hot

!['dyngq_images'](images/dyngq_2019-10-09-21-11-04.png)
实际工程中很少使用One-Hot向量，无法处理新出现的词典里现在没有的词汇

#### 分布式表示 Distributed Word Representation

Embedding || CBOW，Skip-gram（word2vec）

Embedding层一般可以选择：

1. 加载预训练的词嵌入（比如常用Glove预训练的词嵌入）
2. 利用Word2Vector通过自己的语料库自己训练词嵌入

[参考链接：word2vec和word embedding有什么区别](https://www.zhihu.com/question/53354714)
!['dyngq_images'](images/dyngq_2019-10-09-21-34-53.png)
!['dyngq_images'](images/dyngq_2019-10-09-21-46-53.png)
!['dyngq_images'](images/dyngq_2019-10-09-21-48-10.png)
训练词向量
!['dyngq_images'](images/dyngq_2019-10-09-21-49-38.png)

## 精读论文

!['dyngq_images'](images/dyngq_2019-10-10-11-40-46.png)
不熟悉的知识点有NNLM/RNNLM、LSA、LDA

> word2vec的本质就是一个语言模型

!['dyngq_images'](images/dyngq_2019-10-10-17-26-59.png)

**关于负采样**
[（三）通俗易懂理解——Skip-gram的负采样](https://zhuanlan.zhihu.com/p/39684349)
[word_embedding的负采样算法,Negative Sampling 模型](http://www.imooc.com/article/41635)
!['dyngq_images'](images/dyngq_2019-10-10-19-45-11.png)
[百度百科：负采样](https://baike.baidu.com/item/负采样/22884020?fr=aladdin)

论文结构
!['dyngq_images'](images/dyngq_2019-10-10-20-50-11.png)

NNLM神经网路语言模型（Nerual Network Language Model）
[神经网路语言模型(NNLM)的理解](https://blog.csdn.net/lilong117194/article/details/82018008)

最大似然
!['dyngq_images'](images/dyngq_2019-10-10-21-55-29.png)
!['dyngq_images'](images/dyngq_2019-10-10-21-57-47.png)

### 模型结构

!['dyngq_images'](images/dyngq_2019-10-11-11-26-55.png)
!['dyngq_images'](images/dyngq_2019-10-11-20-04-05.png)
!['dyngq_images'](images/dyngq_2019-10-11-20-54-48.png)

* 具体过程：

1. 假设滑动窗口大小为 4。既：给定前面三个词 预测 第四个词会是谁出现的概率最大
2. 输入为前三个词的one-hot，输出为第四个此的one-hot（第四个词已知，所以是监督学习，学习一个矩阵C）
3. 初始化C矩阵。
4. 输入为One-Hot，每个词只有一个位置为1，其他为0，所以经过与C矩阵相乘，得到每个词的一行自定义维（300维等）的属于自己的稠密向量。（投影层）
5. 300维的稠密向量在隐藏层全连接（线：3 * 100 跟线），也就是说hidden layer每个神经元都有3条线相连，使用非线性的tan函数结合H和B进行激活输出。（隐藏层）
6. 输出层采用全连接，有 100 * 10W(语料库词总数)条线。使用softmax结合U和D激活输出最后的概率，既第四个词为某个词的概率值（10W个全部的词语的所有概率）。（输出层）
7. 反向传播更新 **矩阵C（最重要）**、H、B、U、D。

BP(back propagation)神经网络，反向传播
SGD（Stochastic gradient descent） 随机梯度下降

!['dyngq_images'](images/dyngq_2019-10-11-21-46-44.png)

参考资料：

[神经网路语言模型(NNLM)的理解](https://blog.csdn.net/lilong117194/article/details/82018008)

[NNLM(神经网络语言模型)](https://blog.csdn.net/maqunfi/article/details/84455434)

[神经网络语言模型（NNLM）](https://www.jianshu.com/p/c28517cdfb3d)

[神经网络语言建模系列之一：基础模型](https://www.jianshu.com/p/a02ea64d6459)
