# NLP方向  

[收藏 | 帮你精选NLP/CV方向经典+顶会论文，科研看这些就够了！](https://mp.weixin.qq.com/s/yRASMVM9_wRsIRGjaHa2vw)

本篇文章会跟随每个月发布的论文量随时更新具体的论文数目和名称！
添加客服微信，回复“论文” 可以随时查看论文更新通知！

## 第一部分：表示学习

1. 《Distributed Representations of Sentences and Documents》有代码讲解
Google Citation：3955
GitHub Star：229
页数：9
作者：Quoc Le
单位：Google
出处：ICML 2014
本文的作者是word2vec的作者，文中提出通过训练的方式来生成文本的embedding，最终取得了不错的效果，文本的embedding表示也是自然语言处理研究的重点。

2. 《GloVe: Global Vectors for Word Representation》
Google Citation：8118
GitHub Star：1609
页数：12
作者：Jeffrey   Pennington
单位：Stanford
出处：EMNLP 2014
发表于EMNLP 2014，基于矩阵分解的做法来获取词向量。相比word2vec，GloVe更加充分的利用了词的共现信息，word2vec粗暴的让两个向量的点乘相比其他词的点乘最大，而GloVe 直接拟合词对的共现频率。目前基于GloVe预训练的词向量仍是大部分实验的首选。文中再word2vec的基础上增加了一部分新的信息训练得到词向量，并且开源了在大规模语料上训练得到的word embedding，也是我们最常用的预训练word embedding。

3. 《Skip-Thought Vectors》--有代码讲解
Google Citation：1082
GitHub Star：1852
页数：9
作者：Ryan   Kiros
单位：University of Toronto
出处：NIPS 2015
发表于NIPS 2015，通用句表示领域的经典工作，将Skip-gram拓展到了句表示领域。Skip-thoughts 直接在句子间进行预测，也就是将 Skip-gram 中以词为基本单位，替换成了以句子为基本单位，具体做法就是选定一个窗口，遍历其中的句子，然后分别利用当前句子去预测和输出它的上一句和下一句。对于句子的建模利用的 RNN 的sequence 结构，预测上一个和下一个句子时候，也是利用的一个 sequence 的 RNN 来生成句子中的每一个词，所以这个结构本质上就是一个 Encoder-Decoder 框架，只不过和普通框架不一样的是，Skip-thoughts有两个 Decoder

## 第二部分：序列建模

◆◆
序列分类
◆◆

1. 句分类
《Convolutional Neural Networks for Sentence Classification》--有代码讲解
Google Citation：4463
GitHub Star：4786
页数：6
作者：Yoon   Kim
单位：New York University
出处：EMNLP 2014
本⽂是第⼀篇真正意义上使⽤神经⽹络进⾏⽂本分类任务的⼯作，同时也对卷积神经⽹络和词向量的使⽤进⾏了简单的探索性实验，是⾮常适合深度学习和⾃然语⾔处理初学者的⽂章。⽬前，论⽂引⽤已超4463 次

2. 文本分类
《Character-level Convolutional Networks for Text Classification》--有代码讲解
Google Citation：1247
GitHub Star：2199
页数：9
作者：Xiang   Zhang
单位：New York University
出处：NIPS 2015
谷歌学术引用量1247，这篇文章使用CNN来做文本分类，将句子中的字符转化为one-hot排列在一起，这篇文章也是后来很多文本分类工作的对比工作，基本在之后的所有文本分类的对比工作中，都有它的身影

3. 文档分类
文档分类《Hierarchical Attention Networks for Document Classification》
Google Citation：970
GitHub Star：402
页数：10
作者： Zichao Yang
单位：CMU
出处：ACL 2016
发表于NAACL 2016，在文本分类任务中首次提出了层次化attention，有两个显著的特点：(1)采用“词-句子-文章”的层次化结构来表示一篇文本。(2)该模型有两个层次的attention机制，分别存在于词层次(word level)和句子层次(sentence level)。从而使该模型具有对文本中重要性不同的句子和词的能力给予不同“注意力”的能力。通过本文，可以了解注意力机制在文本分类中的作用，目前引用970次。

4. 关系抽取
《Neural Relation Extraction with Selective Attention over Instances》--有代码讲解
Google Citation：223
GitHub Star：652
页数：10
作者： Yankai   Lin
单位：Tsinghua University
出处：ACL 2016
本文面向远程监督关系抽取任务，是当前关系抽取任务的一个重要基线模型。论文使用句子级别的attention机制来解决远程监督标注数据带来的wrong label问题，对不同的句子赋予不同的权重，有效降低噪声数据对模型训练的贡献度，在计算句子权重时，给每个关系赋予一个关系向量，通过计算关系向量和句子向量之间的匹配程度来给不同句子赋予不同的权重。目前，论文引用已超200次。

◆◆
序列标注
◆◆

《End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF》--有代码讲解
Google Citation：632
GitHub Star：1465
页数：10
作者： Xuezhe   Ma
单位：CMU
出处：ACL 2016
发表于ACL 2016，是深度学习应用到序列标注任务中的经典之作，现有方法基本都是在此方法上的改进。传统基于特征的方法需要繁杂的特征模板。本文通过神经网络编码隐层信息作为CRF的输入特征，取得了良好的效果。目前引用632次。通过本文可以了解条件随机场在NLP任务中的应用，并了解目前序列标注的基本架构。

◆◆
序列标注到序列学习
◆◆

1. 序列到序列学习《Sequence to Sequence Learning with Neural Networks》--有代码讲解
Google Citation：6836
GitHub Star：2816
页数：9
作者： Ilya   Sutskever
单位：Google
出处：NIPS 2014
这篇文章发表在NIPS2014上，目前google学术引用量6928。本文提出使用多层的LSTM用于seq2seq模型，并取得了非常好的效果，这篇论文之后的机器翻译模型基本都是默认采用的多层LSTM

2. 序列到序列学习《Convolutional Sequence to Sequence Learning》
Google Citation：706
GitHub Star：4607
页数：10
作者：Jonas   Gehring
单位：Facebook AI Research
出处：ICML 2017
发表于ICML 2017， 本文提出了基于卷积神经网络（CNN）的 seq2seq 架构，和基于循环神经网络（RNN）的 seq2seq 相比，其更易于加速训练，在 GPU 上达到 9.8 倍加速，平均每个 CPU 核上也达到 17 倍加速。此外，本文工作在 WMT’14 English-German 和 WMT’14 English-French 两个数据集上，也取得相对更好的 BLUE Score。目前引用706次

3. 机器翻译Google’s Neural Machine Translation System: Bridging the Gap between Human and Machine Translation
Google Citation：1491
GitHub Star：2711
页数：23
作者： Yonghui   Wu
单位：Google
出处：Arxiv 2016
谷歌学术引用量1491次，是谷歌2016年发表的论文，这时，谷歌也正式在自家翻译产品上使用神经机器翻译方法来代替传统机器翻译方法，这篇文章发出之后，各种机器翻译超过人类的新闻层出不穷，都是出自这篇论文

4. 机器翻译Phrase-Based & Neural Unsupervised Machine Translation
Google Citation：69
GitHub Star：1239
页数：14
作者：Guillaume   Lample
单位：Facebook
出处：EMNLP 2018
本文面向无监督机器翻译领域，突破了原有的神经机器翻译需要有足够大的平行语料库的限制，克服了平行语料库不足的难题，在此前无监督翻译的基础上得到了极大地提升，并且达到了和有将近10万份翻译参考样本的监督式方法的水平，是EMNLP 2018的最佳论文，同时也被评为 2018 年 NLP 的十大突破之一

5. 自动摘要Get To The Point: Summarization with Pointer-Generator Networks
Google Citation：345
GitHub Star：1276
页数：10
作者： Abigail   See
单位：Stanford
出处：ACL 2017
发表于ACL 2017，sequence-to-sequence模型应用于摘要生成时存在两个主要的问题：（1）难以准确复述原文的事实细节、无法处理原文中的未登录词(OOV)；（2）生成的摘要中存在重复的片段。针对这两个问题，本文提出融合了seq2seq模型和pointer network的pointer-generator network以及覆盖率机制(coverage mechanism)，在CNN/Daily Mail数据集上，相比于state-of-art，ROUGE分数提升了两个点。目前引用351次。


第三部分：综合性进阶NLP任务

问答和阅读理解
1. End-To-End Memory Networks--有代码讲解
Google Citation：1064
GitHub Star：1600
页数：9
作者：Sainbayar   Sukhbaatar
单位：New York University
出处：NIPS 2015

2. QANet: Combining Local Convolution with Global Self-Attention for Reading Comprehension--有代码讲解
Google Citation：113
GitHub Star：878
页数：16
作者： Adams   Wei Yu
单位：CMU
出处：ICLR 2018
本⽂的关键动机是，卷积能够捕获⽂本的局部结构，⽽⾃注意⼒机制能够学习到每⼀对 词语之间的全局相互作⽤。⽂章仅使⽤卷积和⾃注意⼒机制作为构建编码器的模块，通 过注意⼒机制来学习语境和问题之间的交互，⼀度获得 SQuAD 的第⼀名

3. 《Bidirectional Attention Flow for Machine Comprehension》--有代码讲解
Google Citation：490
GitHub Star：1220
页数：13
作者： Minjoon   Seo
单位：University of   Washington
出处：ICLR 2017
谷歌引用量490，本文是早期做机器阅读理解的文章，本文提出使用Query2Context attention 和Context2Query attention两个来提取Query和Content的之间的信息。

文本生成
1.Adversarial Learning for Neural Dialogue Generation
Google Citation：301
GitHub Star：166
页数：13
作者： Jiwei   Li
单位：Stanford
出处：EMNLP 2017
发表于EMNLP 2017，本文首次提出了基于GAN框架的自然语言对话生成模型，目前自然语言生成领域的各类基于强化学习的方法仍然是基于本文提出的框架。被引用301次。

2. 《SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient》--有代码讲解
Google Citation：528
GitHub Star：1586
页数：11
作者：Lantao   Yu
单位：SJTU
出处：AAAI 2017
发表于AAAI 2017， GAN 在生成连续离散序列时会遇到两个问题：一是因为生成器的输出是离散的，梯度更新从判别器传到生成器比较困难；二是判别器只有当序列被完全生成后才能进行判断，但此刻指导用处已不太大，而如果生成器生成序列的同时判别器来判断，如何平衡当前序列的分数和未来序列的分数又是一个难题。在这篇论文中，作者提出了一个序列生成模型——SeqGAN ，来解决上述这两个问题。作者将生成器看作是强化学习中的 stochastic policy，这样 SeqGAN 就可以直接通过 gradient policy update 避免生成器中的可导问题。同时，判别器对整个序列的评分作为强化学习的奖励信号可以通过 Monte Carlo 搜索传递到序列生成的中间时刻。是GAN和强化学习应用到自然语言生成任务中的开山之作，目前引用507次。


知识图谱
1. Modeling Relational Data with Graph Convolutional Networks
Google Citation：196
GitHub Star：715
页数：9
作者：Michael   Schlichtkrull
单位：University of   Amsterdam
出处：ESWC 2018
发表于ESWC 2018，首次在知识图谱中引入relational graph convolutional networks (R-GCNs)，来建模关系数据，面向link prediction（从缺失事实中恢复，如主语谓语宾语三元组）和 实体分类（恢复实体缺失的属性）两个任务。借助这篇论文可以了解近年来十分流行的GCN在关系图网络中的应用。目前引用196次


第四部分：语言建模

1. Exploring the Limits of Language Modeling
Google Citation：507
GitHub Star：54722
页数：11
作者：Rafal   Jozefowicz
单位：Google
出处：arXiv   2016




2.Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context
Google Citation：33
GitHub Star：7514
页数：11
作者： Zhilin   Yang
单位：CMU
出处：ICLR   2019
发表于ACL 2019，CMU和谷歌对Transformer的最新改进，以往的Transformer 网络由于受到上下文长度固定的限制，学习长期以来关系的潜力有限。本文提出的新神经架构 Transformer-XL 可以在不引起时间混乱的前提下，可以超越固定长度去学习依赖性，同时还能解决上下文碎片化问题，在五个语言模型数据集上都获得了很好的结果。目前引用33次

3. An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling
Google Citation：189
GitHub Star：1755
页数：14
作者：Shaojie   Bai
单位：CMU
出处：arXiv 2018
本文的主要工作是提出一种 TCN （Temporal Convolutional Networks） 网络结构，用卷积的方式进行序列数据的建模，并且在序列建模任务上，通过多组实验，证明 TCN 取得了和复杂 RNN、LSTM、GRU 等模型相当的精度。文章使用因果卷积来保证卷积过程从未来到过去没有信息泄露，即只使用过去和当前的信息，使用空洞卷积来扩大卷积网络的感受野，使用残差卷积来使得网络更深，从而解决长时依赖问题


第五部分：未来Future

1. 《Deep contextualized word representations》--有代码讲解
Google Citation：801
GitHub Star：6464
页数：15
作者：Matthew   E. Peters
单位：University of   Washington
出处：NAACL 2018
发表于NAACL 2018，并获得了当年该会议的最佳论文。该研究打开了预训练语言模型的大门，提出了一种新型深度语境化词表征，可对词使用的复杂特征（如句法和语义）和词使用在语言语境中的变化进行建模（即对多义词进行建模）。这些表征可以轻松添加至已有模型，并在 6 个 NLP 问题中显著提高当前最优性能。目前被引用801次

2.《BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding》
Google Citation：676
GitHub Star：16240
页数：16
作者： Jacob   Delvin
单位：Google
出处：NAACL 2019
发表于NAACL 2019，并获得了当年该会议的最佳论文。论文介绍了一种新的语言表征模型 BERT，意为来自 Transformer 的双向编码器表征。BERT 旨在基于所有层的左、右语境来预训练深度双向表征。因此，预训练的 BERT 表征可以仅用一个额外的输出层进行微调，进而为很多任务（如问答和语言推断任务）创建当前最优模型，无需对任务特定架构做出大量修改。BERT 的概念很简单，但实验效果很强大。它刷新了 11 个 NLP 任务的当前最优结果，包括将 GLUE 基准提升至 80.4%（7.6% 的绝对改进）、将 MultiNLI 的准确率提高到 86.7%（5.6% 的绝对改进），以及将 SQuAD v1.1 的问答测试 F1 得分提高至 93.2 分（提高 1.5 分）——比人类表现还高出 2 分。目前被引用533次



从经典到前沿


经典篇
1.《Efficient Estimation of Word Representations in Vector Space》 
word2vec是将词汇向量化，这样我们就可以进行定量的分析，分析词与词之间的关系，这是one-hot encoding做不到的。Google的Tomas Mikolov 在2013年发表的这篇论文给自然语言处理领域带来了新的巨大变革，提出的两个模型CBOW (Continuous Bag-of-Words Model)和Skip-gram (Continuous Skip-gram Model)，创造性的用预测的方式解决自然语言处理的问题，而不是传统的词频的方法。奠定了后续NLP处理的基石。并将NLP的研究热度推升到了一个新的高度。 

2.《Neural Machine Translation by Jointly Learning to Align and Translate》    
Attention机制最初由图像处理领域提出，后来被引入到NLP领域用于解决机器翻译的问题，使得机器翻译的效果得到了显著的提升。attention是近几年NLP领域最重要的亮点之一，后续的Transformer和Bert都是基于attention机制。

3.《Transformer: attention is all you need》 
这是谷歌与多伦多大学等高校合作发表的论文，提出了一种新的网络框架Transformer，是一种新的编码解码器，与LSTM地位相当。

Transformer是完全基于注意力机制（attention mechanism)的网络框架，使得机器翻译的效果进一步提升，为Bert的提出奠定了基础。该论文2017年发表后引用已经达到1280，GitHub上面第三方复现的star2300余次。可以说是近年NLP界最有影响力的工作，NLP研究人员必看！

前沿篇
1.《A Convolutional Neural Network for Modelling Sentences》
前沿篇：《fasttext：Bag of Tricks for Efficient Text Classification》
第一篇论文将CNN引入NLP来进行文本分类，巧妙地设计了filter的结构，将n-gram的思想用于NLP，很巧妙。第二篇的思想比较简单，效果很好，效率很高。

2.《Siamese recurrent architectures for learning sentence similarity》
该篇论文仅仅是Siamese network的一个使用场景。在Kaggle question pair match和国内的很多NLP比赛中，比赛的冠军们无一不使用siamese network，该网络模型在图像、语音等领域也有广泛的应用，是非常实用的模型。

