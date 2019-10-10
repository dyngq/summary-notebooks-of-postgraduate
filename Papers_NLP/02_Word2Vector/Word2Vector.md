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
[参考链接：word2vec和word embedding有什么区别](https://www.zhihu.com/question/53354714)
!['dyngq_images'](images/dyngq_2019-10-09-21-34-53.png)
!['dyngq_images'](images/dyngq_2019-10-09-21-46-53.png)
!['dyngq_images'](images/dyngq_2019-10-09-21-48-10.png)
训练词向量
!['dyngq_images'](images/dyngq_2019-10-09-21-49-38.png)

## 精读论文

!['dyngq_images'](images/dyngq_2019-10-10-11-40-46.png)
