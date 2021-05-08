# NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE

> 核心在于注意力机制，注意力机制的核心在于找到一个源语言与目标语言之间的软连接。

## Abstract

> In this paper, we conjecture that the use of a fixed-length vector is a bottleneck in improving the performance of this basic encoder–decoder architecture,and propose to extend this by allowing a model to automatically (soft-)search for parts of a source sentence that are relevant to predicting a target word,without having to form these parts as a hard segment explicitly.

## Introduction

> 对于传统的 Encoder-Decoder架构来说：This may make it difficult for the neural network to cope with long sentences,especially those that are longer than the sentences in the training corpus.
>
> 而对于改进的注意力机制来说
