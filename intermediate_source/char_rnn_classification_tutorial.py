# -*- coding: utf-8 -*-
"""
使用字符级RNN对名字分类
*********************************************
**翻译者**: `Antares博士 <http://www.studyai.com/antares>`_

我们将建立和训练一个基本的字符级RNN来对单词进行分类。
字符级RNN将单词读取为一系列字符(characters)-在每一步输出一个预测和“隐藏状态(hidden state)”，
将其先前的隐藏状态输入到下一步。我们将最后的预测作为输出，即单词属于哪一类。

具体来说，我们将对来自18种语言的几千个姓氏(surnames)进行训练，并根据拼写预测一个名字来自哪种语言：

::

    $ python predict.py Hinton
    (-0.47) Scottish
    (-1.52) English
    (-3.57) Irish

    $ python predict.py Schmidhuber
    (-0.19) German
    (-2.48) Czech
    (-2.68) Dutch


**推荐阅读:**

我假定你最近才安装了 PyTorch, 知道 Python, 并且理解 张量(Tensors)是什么东西:

-  https://pytorch.org/ 查看安装指南
-  :doc:`/beginner/deep_learning_60min_blitz` 在这个章节获得PyTorch的起步知识
-  :doc:`/beginner/pytorch_with_examples` 获得一个宽泛而有深度的概览
-  :doc:`/beginner/former_torchies_tutorial` 如果您是前Lua Torch用户

如果了解 RNNs 并知道它们的工作原理将会很有用:

-  `递归神经网络的不合理有效性 <http://karpathy.github.io/2015/05/21/rnn-effectiveness/>`__ 展示了好多的真实生活案例。
-  `理解 LSTM 网络 <http://colah.github.io/posts/2015-08-Understanding-LSTMs/>`__ 这篇文章是专门关于LSTMs的，也有关于RNNs的通用的信息。

准备数据
==================

.. Note::
   从 `这儿 <https://download.pytorch.org/tutorial/data.zip>`_
   下载数据并抽取到当前目录。

包含在 ``data/names`` 目录中的是 18 个文本文件，取名为 "[Language].txt"。 每个文件包含很多名字，一个名字占一行
大多数用拉丁字母拼写 (但是我们仍然需要将它们从 Unicode 转换为 ASCII)。

最后，我们将得到一本词典，其中列出了每种语言的名字(names)列表， ``{language: [names ...]}`` 。
泛化变量“category”和“line”(在本例中用于语言和名字)用于以后的可扩展性。
"""
from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os

def findFiles(path): return glob.glob(path)

print(findFiles('data/names/*.txt'))

import unicodedata
import string

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

print(unicodeToAscii('Ślusàrski'))

# Build the category_lines dictionary, a list of names per language
category_lines = {}
all_categories = []

# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

for filename in findFiles('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)


######################################################################
# 现在我们有了 ``category_lines`` ，这是一个字典，将每个类别(语言)映射到一行(名字)的列表中。
# 我们还跟踪了 ``all_categories`` (只是一个语言的列表)和 ``n_categories`` ，以供以后参考。
#

print(category_lines['Italian'][:5])


######################################################################
# 把名字(Names)转换为张量
# --------------------------
#
# 现在我们已经组织好了所有的名字，我们需要把它们转换成张量来使用它们。
#
# 为了表示单个字母，我们使用了一个size为 ``<1 x n_letters>`` 的 “one-hot vector” 。
# one-hot vector填充了0，除了当前字母索引处的1，例如, e.g. ``"b" = <0 1 0 0 0 ...>`` 。
#
# 为了创造一个词，我们将这些元素加入到一个2D矩阵 ``<line_length x 1 x n_letters>`` 中。
#
# 这个额外的一维是因为PyTorch假设所有的东西都是按批次的(batches)-我们只是在这里使用batch size 为 1 的batch。
#

import torch

# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)

# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

print(letterToTensor('J'))

print(lineToTensor('Jones').size())


######################################################################
# 创建网络
# ====================
#
# 在自动梯度之前，在Torch中创建一个递归神经网络需要在若干个时间步中克隆一个层的参数。
# 层包含隐藏状态和梯度，现在完全由图自己来处理。
# 这意味着您可以非常“纯”的方式实现RNN，作为常规的前馈层。
#
# 这个 RNN module (主要是从 `为Torch用户写的PyTorch教程 <https://pytorch.org/tutorials/beginner/former_torchies/nn_tutorial.html#example-2-recurrent-net>`__ 里面复制的)
# 只是两个线性层，它们在输入和隐藏状态下工作，输出后有一个LogSoftmax层。
#
# .. figure:: https://i.imgur.com/Z2xbySO.png
#    :alt:
#
#

import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)


######################################################################
# 要运行这个网络的一个step，我们需要传递一个输入(在我们的例子中，是当前字母的张量)和
# 一个先前的隐藏状态(我们首先将其初始化为零)。我们将返回输出(每种语言的概率)和
# 下一个隐藏状态(我们将在下一步保持这种状态)。
#

input = letterToTensor('A')
hidden =torch.zeros(1, n_hidden)

output, next_hidden = rnn(input, hidden)


######################################################################
# 为了提高效率，我们不想为每一步创建一个新的张量，所以我们将使用 ``lineToTensor``  
# 而不是 ``letterToTensor`` 并使用切片。
# 这可以通过预计算张量的batches来进一步优化。
#

input = lineToTensor('Albert')
hidden = torch.zeros(1, n_hidden)

output, next_hidden = rnn(input[0], hidden)
print(output)


######################################################################
# 如您所见，输出是 ``<1 x n_categories>`` 张量，其中每个item都是该类别的可能性(likelihood)
# (更高的可能性更大)。
#


######################################################################
#
# 训练
# ========
# 为训练做准备
# ----------------------
#
# 在接受训练之前，我们应该做一些辅助函数。首先是解释网络的输出，
# 我们知道网络输出是每个类别的可能性。我们可以使用 ``Tensor.topk`` 获得最大值对应的索引：
#

def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

print(categoryFromOutput(output))


######################################################################
# 我们还需要一种快速获取训练样例(名称及其语言)的方法： 
#

import random

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor

for i in range(10):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    print('category =', category, '/ line =', line)


######################################################################
# 训练网络
# --------------------
#
# 现在训练这个网络所需要的就是给它展示一堆样例，让它去猜测，如果它错了，就告诉它犯错了。
#
# 对于损失函数  ``nn.NLLLoss`` 是合适的，因为RNN的最后一层是 ``nn.LogSoftmax`` 。 
#

criterion = nn.NLLLoss()


######################################################################
# 训练的每个循环都:
#
# -  创建输入张量和目标张量
# -  创建零初始化的隐藏状态
# -  把每一个字符读进来，并且
#
#    -  为下一个字符保留隐藏状态
#
# -  将最终输出与目标值进行对比。
# -  反向传播
# -  返回输出和损失
#

learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn

def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()

    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item()


######################################################################
# 现在我们只需要用一堆样例来分析这个问题。由于 ``train`` 函数返回输出和损失，我们可以打印它的猜测，
# 也可以跟踪损失进行绘图。由于有1000个样例(examples)，我们只打印每一个 ``print_every`` 样例，
# 并取一个平均损失。
#

import time
import math

n_iters = 100000
print_every = 5000
plot_every = 1000



# Keep track of losses for plotting
current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    # Print iter number, loss, name and guess
    if iter % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0


######################################################################
# 绘制结果
# --------------------
#
# 从 ``all_losses`` 绘制历史损失来展示网络的学习:
#

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.figure()
plt.plot(all_losses)


######################################################################
# 评估结果
# ======================
#
# 为了查看网络在不同类别上的性能，我们将创建一个混淆矩阵，为每种实际语言(行)指示网络猜测的语言(列)。
# 为了计算混淆矩阵，运行 ``evaluate()`` 使一堆样本在网络中通过，这与除去了反向传播的 ``train()`` 相同。
#

# Keep track of correct guesses in a confusion matrix
confusion = torch.zeros(n_categories, n_categories)
n_confusion = 10000

# Just return an output given a line
def evaluate(line_tensor):
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output

# Go through a bunch of examples and record which are correctly guessed
for i in range(n_confusion):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output = evaluate(line_tensor)
    guess, guess_i = categoryFromOutput(output)
    category_i = all_categories.index(category)
    confusion[category_i][guess_i] += 1

# Normalize by dividing every row by its sum
for i in range(n_categories):
    confusion[i] = confusion[i] / confusion[i].sum()

# Set up plot
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

# Set up axes
ax.set_xticklabels([''] + all_categories, rotation=90)
ax.set_yticklabels([''] + all_categories)

# Force label at every tick
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

# sphinx_gallery_thumbnail_number = 2
plt.show()


######################################################################
# 你可以从主轴上找出亮点，显示它猜错了哪种语言，例如Chinese for Korean, 和 Spanish for Italian。
# 它似乎在希腊语方面做得很好，在英语方面则很差(也许是因为与其他语言的重叠)。
#


######################################################################
# 运行我们的用户输入
# ---------------------
#

def predict(input_line, n_predictions=3):
    print('\n> %s' % input_line)
    with torch.no_grad():
        output = evaluate(lineToTensor(input_line))

        # Get top N categories
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('(%.2f) %s' % (value, all_categories[category_index]))
            predictions.append([value, all_categories[category_index]])

predict('Dovesky')
predict('Jackson')
predict('Satoshi')


######################################################################
# `PyTorch 实战 <https://github.com/spro/practical-pytorch/tree/master/char-rnn-classification>`__ 
# 的最终版本将上面的代码分成几个文件:
#
# -  ``data.py`` (加载文件)
# -  ``model.py`` (定义 RNN)
# -  ``train.py`` (运行训练过程)
# -  ``predict.py`` (使用命令行参数运行 ``predict()`` )
# -  ``server.py`` (使用 bottle.py 把预测作为JSON API)
#
# 运行 ``train.py`` 来训练和保存网络。
#
# 使用一个 name 运行 ``predict.py`` 来查看预测结果:
#
# ::
#
#     $ python predict.py Hazaki
#     (-0.42) Japanese
#     (-1.39) Polish
#     (-3.51) Czech
#
# 运行 ``server.py`` 并访问  http://localhost:5533/Yourname 来获取预测结果
#


######################################################################
# 练习
# =========
#
# -  尝试具有 line -> category 这种结构的其他数据集, 例如:
#
#    -  任意单词 -> 语言
#    -  First name -> 性别
#    -  人物角色 -> 作者
#    -  页标题 -> 博客
#
# -  用一个更大 和/或 更好的网络获得更好的效果
#
#    -  添加更多的线性层
#    -  尝试 ``nn.LSTM`` 和 ``nn.GRU`` layers
#    -  将这些RNNs中的多个合并为更高级别的网络。
#
