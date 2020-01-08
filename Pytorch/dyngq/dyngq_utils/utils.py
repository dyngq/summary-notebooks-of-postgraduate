import collections
import math
import os
import random
import sys
import tarfile
import time
import json
import zipfile
from tqdm import tqdm
from PIL import Image
from collections import namedtuple

from IPython import display
from matplotlib import pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchtext
import torchtext.vocab as Vocab
import numpy as np



def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples)) # 1 2 3 4 5 6 ... 998 999
    random.shuffle(indices)  # 样本的读取顺序是随机的
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)]) # 最后一次可能不足一个batch
#         print(indices[i: min(i + batch_size, num_examples)])
#         print(j)
        yield  features.index_select(0, j), labels.index_select(0, j)
# yield就是用来迭代返回的，区别于return

def linreg(X, w, b):  # 下面是线性回归的矢量计算表达式的实现。我们使用mm函数做矩阵乘法。
    return torch.mm(X, w) + b

