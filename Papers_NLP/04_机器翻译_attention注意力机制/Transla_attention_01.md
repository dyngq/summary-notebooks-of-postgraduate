# Transla_attention

> 机器翻译 注意力机制

端到端

深度学习 表示学习

!['dyngq_images'](images/dyngq_2019-10-13-20-02-22.png)
!['dyngq_images'](images/dyngq_2019-10-13-20-02-35.png)
!['dyngq_images'](images/dyngq_2019-10-13-20-02-48.png)
!['dyngq_images'](images/dyngq_2019-10-13-20-03-00.png)
!['dyngq_images'](images/dyngq_2019-10-13-20-03-10.png)

核心技术： **端到端 End-to-End** 包括：Encoder-Decoder模型、Sequence-to-Sequence模型

!['dyngq_images'](images/dyngq_2019-10-13-20-03-23.png)

!['dyngq_images'](images/dyngq_2019-10-13-20-03-52.png)
!['dyngq_images'](images/dyngq_2019-10-13-20-04-42.png)
!['dyngq_images'](images/dyngq_2019-10-13-20-04-53.png)

* 从上图是利用RNN来得到一种句子的向量表示。到最后的节点（实际上就是那一个节点，不断在循环而已，所以）就存储了整个句子的所有信息。
* 深度学习带给我们的 **革命性的变化** 就是 **信息的表达的方式** 。
* 机器翻译的核心就是不同语言之间的等价转换。
* 传统上，我们习惯了用**离散表示**，用词、短句、句法树。
* 深度学习强调的是，我们用**连续的表示**， 用**数字**。

## Encoder Decoder 编码器-解码器框架

* 从最后包含整个句子信息的节点中 把信息再分别解码出来。

!['dyngq_images'](images/dyngq_2019-10-13-20-05-04.png)

我们经常使用LSTM（长短时记忆网络：核心就在于记忆，memory单元）来解决简单SampleRNN的梯度消失爆炸问题。

但这同时也是有一些缺点的，不管多长的句子的所有信息都只会被凝聚在最后一个节点的中，变成固定维度的向量，这样对于之后的解码、尤其是同样有长度损耗的解码来说是不友好的。

!['dyngq_images'](images/dyngq_2019-10-13-20-05-17.png)

针对那样的缺点，引入注意力机制。
!['dyngq_images'](images/dyngq_2019-10-13-20-05-26.png)
!['dyngq_images'](images/dyngq_2019-10-13-20-05-47.png)

## Core

!['dyngq_images'](images/dyngq_2019-10-18-22-12-16.png)
!['dyngq_images'](images/dyngq_2019-10-18-22-12-29.png)
!['dyngq_images'](images/dyngq_2019-10-18-22-12-42.png)

## Ci生成-注意力机制

生成下一个单词时、哪一些单词对这个单词的生成是最有价值的。
!['dyngq_images'](images/dyngq_2019-10-18-22-14-46.png)

> 这个关于Ci和αi的作用的地方的理解出现过偏差，需要注意

软对齐模型，用来衡量第j个源单词与目标端第i个词的匹配程度
传统的对齐模型需要源语言每个单词明确对齐到目标语言的一个或多个词，软对齐模型则不需要，并且可以融入到整个模型框架中反向传播求梯度。

!['dyngq_images'](images/dyngq_2019-10-18-22-15-27.png)

!['dyngq_images'](images/dyngq_2019-10-18-22-22-34.png)

!['dyngq_images'](images/dyngq_2019-10-18-22-22-45.png)

!['dyngq_images'](images/dyngq_2019-10-18-22-23-08.png)

[对齐模型参考论文：词语对齐的快速增量式训练方法研究](02.pdf)
