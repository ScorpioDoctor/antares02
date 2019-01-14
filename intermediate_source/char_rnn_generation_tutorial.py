# -*- coding: utf-8 -*-
"""
使用字符级RNN生成名字
*******************************************
**翻译者**: `Antares博士 <http://www.studyai.com/antares>`_

在 :doc:`上一篇教程 </intermediate/char_rnn_classification_tutorial>` 中，我们使用RNN将名字(names)分类为它们所属的语言(language)。
这一次，我们将转过来，从语言(languages)中生成名字(names)。

::

    > python sample.py Russian RUS
    Rovakov
    Uantov
    Shavakov

    > python sample.py German GER
    Gerren
    Ereng
    Rosher

    > python sample.py Spanish SPA
    Salla
    Parer
    Allan

    > python sample.py Chinese CHI
    Chan
    Hang
    Iun

我们仍然手工制作一个带有几个线性层的小RNN。最大的区别是，在读完一个名字的所有字母(letter)之后，
我们不再预测一个类别，而是输入一个类别(category)，然后一次输出一个字母。
递归地(Recurrently)预测字符以形成语言(这也可以用单词或其他高阶结构来完成)通常被称为“语言模型(language model)”。

**推荐阅读:**

我假定你最近才安装了 PyTorch, 知道 Python, 并且理解 张量(Tensors)是什么东西:

-  https://pytorch.org/ 查看安装指南
-  :doc:`/beginner/deep_learning_60min_blitz` 在这个章节获得PyTorch的起步知识
-  :doc:`/beginner/pytorch_with_examples` 获得一个宽泛而有深度的概览
-  :doc:`/beginner/former_torchies_tutorial` 如果您是前Lua Torch用户

如果了解 RNNs 并知道它们的工作原理将会很有用:

-  `递归神经网络的不合理有效性 <http://karpathy.github.io/2015/05/21/rnn-effectiveness/>`__ 展示了好多的真实生活案例。
-  `理解 LSTM 网络 <http://colah.github.io/posts/2015-08-Understanding-LSTMs/>`__ 这篇文章是专门关于LSTMs的，也有关于RNNs的通用的信息。

我还推荐大家阅读上一篇教程： :doc:`/intermediate/char_rnn_classification_tutorial`


准备数据
==================

.. Note::
   从 `这儿 <https://download.pytorch.org/tutorial/data.zip>`_
   下载数据并抽取到当前目录。

关于这个处理过程的更多信息请参考上一篇教程。简单点说, 我们有一堆普通文本文件 ``data/names/[Language].txt`` ，文件中每一行是一个名字(name)。
我们将很多行(lines)划分成一个数组，再从 Unicode 转换为 ASCII, 最后构造了一个字典：  ``{language: [names ...]}`` 。

"""
from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string

all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 1 # Plus EOS marker

def findFiles(path): return glob.glob(path)

# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

# Build the category_lines dictionary, a list of lines per category
category_lines = {}
all_categories = []
for filename in findFiles('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

if n_categories == 0:
    raise RuntimeError('Data not found. Make sure that you downloaded data '
        'from https://download.pytorch.org/tutorial/data.zip and extract it to '
        'the current directory.')

print('# categories:', n_categories, all_categories)
print(unicodeToAscii("O'Néàl"))


######################################################################
# 创建网络
# ====================
#
# 该网络扩展了 :doc:`上一篇教程的RNN网络 </intermediate/char_rnn_classification_tutorial>` ，并为类别张量提供了额外的参数，
# 与其他类张量连接(concatenate)在一起。类别张量(category tensor)是一个同字母输入(letter input)一样的one-hot向量.
#
# 我们将把输出解释为下一个字母的概率。采样时，最有可能的输出字母被用作下一个输入字母。
#
# 我添加了第二个线性层  ``o2o`` (在合并了隐藏和输出之后)来给它提供更多的工作空间。
# 还有一个dropout层，它随机地用给定的概率(这里是0.1)对其输入的部分进行置零，
# 并且通常用于模糊输入以防止过度拟合。在这里，我们在接近网络的末尾使用它，
# 目的是增加一些混乱和增加采样多样性。
#
# .. figure:: https://i.imgur.com/jzVrf7f.png
#    :alt:
#
#

import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(n_categories + input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(n_categories + input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input, hidden):
        input_combined = torch.cat((category, input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


######################################################################
# 训练
# =========
# 为训练做准备
# ----------------------
#
# 在接受训练之前，我们应该做一些辅助函数获得 (category, line) 的随机对(random pairs):
#

import random

# Random item from a list
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

# Get a random category and random line from that category
def randomTrainingPair():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    return category, line


######################################################################
# 对于每个时间步(即，对于训练单词中的每个字母)，网络的输入将是 ``(category, current letter, hidden state)`` ，
# 输出将是 ``(next letter, next hidden state)`` 。
# 因此，对于每个训练集，我们需要category, a set of input letters, 和 a set of output/target letters。
#
# 由于我们在每个时间步从当前字母预测下一个字母，所以字母对是行中的连续字母组，例如，对于 ``"ABCD<EOS>"`` ，我们将创建
# ("A", "B"), ("B", "C"), ("C", "D"), ("D", "EOS").
#
# .. figure:: https://i.imgur.com/JH58tXY.png
#    :alt:
#
# 类别张量(category tensor)是一个 `one-hot tensor <https://en.wikipedia.org/wiki/One-hot>`__, 
# 其 size 是 ``<1 x n_categories>`` 。当训练的时候，我们在每个时间步把它传到网络中---这是一种设计上的选择，它可以作为
# 初始状态或其他某个策略的一部分被包括进来。
#

# One-hot vector for category
def categoryTensor(category):
    li = all_categories.index(category)
    tensor = torch.zeros(1, n_categories)
    tensor[0][li] = 1
    return tensor

# One-hot matrix of first to last letters (not including EOS) for input
def inputTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor

# LongTensor of second letter to end (EOS) for target
def targetTensor(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1) # EOS
    return torch.LongTensor(letter_indexes)


######################################################################
# 为了训练过程的方便，我们将做一个 ``randomTrainingExample`` 函数，获取一个随机(category, line)对，
# 并将它们转化为所需的(category, input, target)张量。
#

# Make category, input, and target tensors from a random category, line pair
def randomTrainingExample():
    category, line = randomTrainingPair()
    category_tensor = categoryTensor(category)
    input_line_tensor = inputTensor(line)
    target_line_tensor = targetTensor(line)
    return category_tensor, input_line_tensor, target_line_tensor


######################################################################
# 训练网络
# --------------------
#
# 与只使用最后一个输出(做预测以及计算损失)的分类网络相比，我们在每一步都要进行预测，
# 所以我们在每一步都计算损失。
#
# 自动梯度的魔力让你在每一步都能简单地将这些损失相加，并在结束时调用反向传播。
#

criterion = nn.NLLLoss()

learning_rate = 0.0005

def train(category_tensor, input_line_tensor, target_line_tensor):
    target_line_tensor.unsqueeze_(-1)
    hidden = rnn.initHidden()

    rnn.zero_grad()

    loss = 0

    for i in range(input_line_tensor.size(0)):
        output, hidden = rnn(category_tensor, input_line_tensor[i], hidden)
        l = criterion(output, target_line_tensor[i])
        loss += l

    loss.backward()

    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item() / input_line_tensor.size(0)


######################################################################
# 为了跟踪训练所需的时间，我添加了一个 ``timeSince(timestamp)`` 函数，它返回一个人类可读的字符串:
#

import time
import math

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


######################################################################
# 训练就像往常一样---多次调用 ``train``` ，等待几分钟，每隔 ``print_every`` 
# 个样例就打印当前时间，损失，
# 并把 ``print_every`` 个样例上的平均损失 保存到 ``all_losses`` ，供以后绘图使用。
#

rnn = RNN(n_letters, 128, n_letters)

n_iters = 100000
print_every = 5000
plot_every = 500
all_losses = []
total_loss = 0 # Reset every plot_every iters

start = time.time()

for iter in range(1, n_iters + 1):
    output, loss = train(*randomTrainingExample())
    total_loss += loss

    if iter % print_every == 0:
        print('%s (%d %d%%) %.4f' % (timeSince(start), iter, iter / n_iters * 100, loss))

    if iter % plot_every == 0:
        all_losses.append(total_loss / plot_every)
        total_loss = 0


######################################################################
# 画损失曲线图
# -------------------
#
# 从 ``all_losses`` 中绘制历史损失展示网络学习：
#

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.figure()
plt.plot(all_losses)


######################################################################
# 对网络采样
# ====================
#
# 为了进行采样，我们给网络一个字母并询问下一个字母是什么，再将其答案作为下一个字母输入，并重复直到EOS token。
#
# -  为 输入类别，起始字母，和 空隐藏状态 创建张量
# -  使用起始字母创建一个字符串 ``output_name`` 
# -  直到达到 最大输出长度,
#
#    -  将当前字母喂给网络
#    -  从最高输出中获取下一个字母，并获得下一个隐藏状态
#    -  如果字母是EOS，就停在这里
#    -  如果是一个常规的字母, 把它添加到 ``output_name`` 然后继续
#
# -  返回最终的name
#
# .. Note::
#    另一种策略不是给它一个起始字母，而是在训练中包含一个 “start of string” token，让网络选择自己的起始字母。
#

max_length = 20

# Sample from a category and starting letter
def sample(category, start_letter='A'):
    with torch.no_grad():  # no need to track history in sampling
        category_tensor = categoryTensor(category)
        input = inputTensor(start_letter)
        hidden = rnn.initHidden()

        output_name = start_letter

        for i in range(max_length):
            output, hidden = rnn(category_tensor, input[0], hidden)
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == n_letters - 1:
                break
            else:
                letter = all_letters[topi]
                output_name += letter
            input = inputTensor(letter)

        return output_name

# Get multiple samples from one category and multiple starting letters
def samples(category, start_letters='ABC'):
    for start_letter in start_letters:
        print(sample(category, start_letter))

samples('Russian', 'RUS')

samples('German', 'GER')

samples('Spanish', 'SPA')

samples('Chinese', 'CHI')


######################################################################
# 练习
# =========
#
# -  尝试具有 category -> line 这种结构的其他数据集, 例如 :
#
#    -  系列小说 -> 人物名字
#    -  语义(Part of speech) -> 单词(Word)
#    -  国家 -> 城市
#
# -  使用一个 "start of sentence" token 以便对网络的采样可以在不用选择起始字符的情况下进行
# -  用一个更大和/或更好的网络获得更好的效果
#
#    -  尝试使用 nn.LSTM 层 和 nn.GRU 层
#    -  将这些RNNs中的多个合并为更高级的网络。
#
