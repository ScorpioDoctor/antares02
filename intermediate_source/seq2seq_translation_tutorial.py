# -*- coding: utf-8 -*-
"""
使用序列到序列的网络和注意机制进行翻译
*************************************************************
**翻译者**: `Antares博士 <http://www.studyai.com/antares>`_

在这个项目中，我们将教一个神经网络，把法语翻译成英语。

::

    [KEY: > input, = target, < output]

    > il est en train de peindre un tableau .
    = he is painting a picture .
    < he is painting a picture .

    > pourquoi ne pas essayer ce vin delicieux ?
    = why not try that delicious wine ?
    < why not try that delicious wine ?

    > elle n est pas poete mais romanciere .
    = she is not a poet but a novelist .
    < she not not a poet but a novelist .

    > vous etes trop maigre .
    = you re too skinny .
    < you re all alone .

... 取得不同程度的成功。

这是因为简单而有力的 `序列到序列的网络 <http://arxiv.org/abs/1409.3215>`__ 思想，
其中两个递归神经网络一起工作，将一个序列转换成另一个序列。
编码器网络将输入序列压缩为向量，解码器网络将该向量展开为新序列。

.. figure:: /_static/img/seq-seq-images/seq2seq.png
   :alt:

为了改进这个模型，我们将使用一种注意机制(`attention mechanism <https://arxiv.org/abs/1409.0473>`__)，
它让解码器学会将注意力集中在输入序列的特定范围上。

**推荐阅读:**

我假定你最近才安装了 PyTorch, 知道 Python, 并且理解 张量(Tensors)是什么东西:

-  https://pytorch.org/ 查看安装指南
-  :doc:`/beginner/deep_learning_60min_blitz` 在这个章节获得PyTorch的起步知识
-  :doc:`/beginner/pytorch_with_examples` 获得一个宽泛而有深度的概览
-  :doc:`/beginner/former_torchies_tutorial` 如果您是前Lua Torch用户


如果了解 序列到序列网络(Sequence to Sequence networks) 并知道它们的工作原理将会很有用:

-  `使用用于统计机器翻译的RNN编解码器学习短语表示 <http://arxiv.org/abs/1406.1078>`__
-  `使用神经网络进行序列到序列的学习 <http://arxiv.org/abs/1409.3215>`__
-  `联合学习对齐与翻译的神经机器翻译 <https://arxiv.org/abs/1409.0473>`__
-  `一种神经会话模型 <http://arxiv.org/abs/1506.05869>`__

你还会发现之前的教程 :doc:`/intermediate/char_rnn_classification_tutorial`
和 :doc:`/intermediate/char_rnn_generation_tutorial` 是非常有帮助的，因为这些概念是与之前
介绍过的Encoder 和 Decoder模型非常相似。

要了解更多信息，请阅读介绍这些主题的论文:

-  `Learning Phrase Representations using RNN Encoder-Decoder for
   Statistical Machine Translation <http://arxiv.org/abs/1406.1078>`__
-  `Sequence to Sequence Learning with Neural
   Networks <http://arxiv.org/abs/1409.3215>`__
-  `Neural Machine Translation by Jointly Learning to Align and
   Translate <https://arxiv.org/abs/1409.0473>`__
-  `A Neural Conversational Model <http://arxiv.org/abs/1506.05869>`__


**需要导入的包**
"""
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

######################################################################
# 加载数据文件
# ==================
#
# 这个项目的数据是一套成千上万的英法翻译对(English to French translation pairs)。
#
# `在这个 Open Data Stack Exchange 上的问题 <http://opendata.stackexchange.com/questions/3888/dataset-of-sentences-translated-into-many-languages>`__
# 把我指向一个开放翻译网站 http://tatoeba.org/ ，它有下载可用 http://tatoeba.org/eng/downloads - 
# 更好的是，有人在这里做了额外的工作，将语言对分割成单独的文本文件: http://www.manythings.org/anki/
#
# 从英语到法语的配对太大，无法包含在仓库(repo)中，所以在继续之前请下载到 ``data/eng-fra.txt`` 。
# 该文件是一个以tab分隔的翻译对列表:
#
# ::
#
#     I am cold.    J'ai froid.
#
# .. Note::
#    从 `这儿 <https://download.pytorch.org/tutorial/data.zip>`_ 下载数据，并且抽取到当前目录下。

######################################################################
# 类似于字符级rnn教程中使用的字符编码，我们将一种语言中的每个单词表示为one-hot vector，
# 或除单个零向量(在单词索引处)外的巨大零向量。与某一种语言中可能存在的几十个字符相比，
# 单词的数量可能大得多，因此编码向量要大得多。然而，我们将欺骗一点，并减少数据，
# 使其每种语言只使用几千个单词。
#
# .. figure:: /_static/img/seq-seq-images/word-encoding.png
#    :alt:
#
#


######################################################################
# 我们需要为每个单词分配一个唯一的索引以便在稍后用于网络的输入(inputs)和目标(targets)。为了跟踪这一切，
# 我们将使用一个名为 ``Lang`` 的辅助类，该类具有word → index (``word2index``)字典和
# index → word(``index2word``)字典，以及每个单词的数量(``word2count``)用于稍后替换稀有单词。
#

SOS_token = 0
EOS_token = 1


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


######################################################################
# 所有的文件都是Unicode编码的, 为了简化，我们将把Unicode编码的字符转变为ASCII,
# 全部转换为小写，裁减掉标点符号。
#

# 把一个 Unicode 字符串转变为简单的 ASCII 字符串, 感谢
# http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# 小写, 裁剪, 并移除非字母字符

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


######################################################################
# 要读取数据文件，我们将文件(file)拆分成行(lines)，然后将行(lines)拆分成对(pairs)。
# 这些文件都是 英语→其他语言，因此，如果我们想要把其他语言翻译为英语，我添加了一个
# ``reverse`` 标记来反转pairs。
#

def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


######################################################################
# 因为有 *很多* 样例句子，我们想要快速地训练一些东西，所以我们将把数据集中到相对较短和简单的句子上。
# 这里的最大长度是10个单词(包括结束标点符号)，我们过滤掉翻译成“I am”或“he is”等形式的句子。
# (较早时用撇号代替)。
#

MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


######################################################################
# 准备数据的全流程如下:
#
# -  读取文本文件，拆分成行，将行拆分成对(pairs)
# -  归一化文本，使用长度和内容过滤
# -  从句子中制作单词对的列表
#

def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
print(random.choice(pairs))


######################################################################
# The Seq2Seq Model
# =================
#
# 循环神经网络(Recurrent Neural Network, or RNN) 是一种对序列进行操作并利用自己的输出作为后续步骤的输入的网络。
#
# 一个序列到序列网络(`Sequence to Sequence network <http://arxiv.org/abs/1409.3215>`__) , 或
# seq2seq 网络, 或 编码解码网络(`Encoder Decoder network <https://arxiv.org/pdf/1406.1078v3.pdf>`__), 是一个模型，
# 它由两个称之为编码器和解码器的RNNs构成。编码器读取输入序列然后输出单个向量(single vector),而解码器读取编码器输出的那个向量
# 然后再产生一个输出序列。
#
# .. figure:: /_static/img/seq-seq-images/seq2seq.png
#    :alt:
#
# 与使用单个RNN进行序列预测(在这里每一个输入对应一个输出)不一样，seq2seq 模型
# 将我们从序列长度和顺序中解放出来，这使得它适合于两种语言之间的翻译(translation)。
#
# 请考虑句子 "Je ne suis pas le chat noir" → "I am not the black cat" 。
# 输入句子的大多数单词在输出语句中都有直接的翻译，但是两个句子的顺序有些不一样，
# e.g. "chat noir" 和 "black cat"。 因为 "ne/pas" 的构造，输入句子中还多出了一个单词。
# 我们可以看到，如果从输入单词的序列直接产生一个正确的翻译是一件很难的事情。
#
# 有了 seq2seq 模型之后，编码器创建一个单个向量(single vector)，这个向量，在理想情况下，编码了输入单词序列的意义("meaning")。
# 该单个向量是句子的某个N维空间中的单个点。
#


######################################################################
# The Encoder
# -----------
#
# seq2seq网络的编码器是一个RNN，它为来自输入句子中的每个单词输出一些值。对于每个输入单词，编码器输出一个向量和一个隐藏状态，
# 并为下一个输入单词使用隐藏状态。
#
# .. figure:: /_static/img/seq-seq-images/encoder-network.png
#    :alt:
#
#

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

######################################################################
# The Decoder
# -----------
#
# 解码器是另一个RNN，它接受编码器输出向量并输出一系列单词以创建翻译。
# 

######################################################################
# 简单解码器
# ^^^^^^^^^^^^^^
#
# 在最简单的seq2seq解码器中，我们只使用编码器的最后输出。最后一个输出有时被称为上下文向量(*context vector*)，
# 因为它从整个输入序列中编码上下文。该上下文向量用作解码器的初始隐藏状态。
#
# 在解码的每一步，给解码器一个输入令牌和隐藏状态。初始输入令牌是字符串的起始 ``<SOS>`` 令牌，
# 第一个隐藏状态是上下文向量(编码器的最后一个隐藏状态)。
#
# .. figure:: /_static/img/seq-seq-images/decoder-network.png
#    :alt:
#
#

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

######################################################################
# 我鼓励你们训练和观察这个模型的结果，但为了节省空间，我们将直接奔着金牌去
# (作者意思是直接给大家介绍最好的模型)，并引入注意机制(Attention Mechanism)。
#


######################################################################
# 注意力解码器
# ^^^^^^^^^^^^^^^^^
#
# 如果只有上下文向量在编码器和解码器之间传递，那么该向量就承担了对整个句子进行编码的负担。
#
# Attention允许解码器网络对自身输出的每一步“聚焦”编码器输出的不同部分。
# 首先，我们计算了一组注意力权重(*attention weights*)。这些将被乘以编码器输出向量，以创建加权组合。
# 结果(在代码中称为 ``attn_applied`` )应该包含有关输入序列的特定部分的信息，
# 从而帮助解码器选择正确的输出单词。
#
# .. figure:: https://i.imgur.com/1152PYf.png
#    :alt:
#
# 利用解码器的输入和隐藏状态作为输入，用另一个前馈层 ``attn`` 进行注意力权值的计算。
# 由于训练数据中有各种大小的句子，为了实际创建和训练这一层，我们必须选择一个
# 最大的句子长度(输入长度，用于编码器输出)。最大长度的句子将使用所有的注意力权重，
# 而较短的句子只使用前几个。
#
# .. figure:: /_static/img/seq-seq-images/attention-decoder-network.png
#    :alt:
#
#

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


######################################################################
# .. note:: 还有其他形式的注意力机制，通过使用相对位置方法，绕过长度限制。
#   阅读 `基于注意力的神经机器翻译的有效方法 <https://arxiv.org/abs/1508.04025>`__ 
#   中的“局部注意” 。
#
# 训练
# ========
#
# 准备训练数据
# -----------------------
#
# 为了训练，对于每个pair我们将需要一个输入张量(输入句子中单词的索引)和目标张量(目标句子中单词的索引)。
# 在创建这些向量时，我们将EOS令牌追加到这两个序列中。
#

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)


######################################################################
# 训练模型
# ------------------
#
# 为了训练，我们让输入语句通过编码器，并跟踪每个输出和最新的隐藏状态。
# 然后，解码器被赋予 ``<SOS>`` 令牌作为它的第一个输入，编码器的最后一个隐藏状态
# 作为它的第一个隐藏状态。
#
# "教师强迫(Teacher forcing)" 是这样一个概念(concept),它将真正的目标输出作为下一个输入，
# 而不是使用解码器的猜测作为下一个输入。使用教师强迫可以使其收敛更快，但是
# `当经过训练的网络被利用(exploited)时，它可能会表现出不稳定性 
# <http://minds.jacobs-university.de/sites/default/files/uploads/papers/ESNTutorialRev.pdf>`__ 。
#
# 你可以观察到教师强迫网络的输出-用连贯的语法阅读，却远离正确的翻译-直观地说，它已经学会了表示输出语法，
# 并且一旦老师告诉它前几个单词，它就可以“捡起(pick up)”意义，但它一开始就没有学会如何从翻译中创造句子。
#
# 由于PyTorch的autograd给我们的自由，我们可以使用一条简单的if语句随机选择使用老师强迫或不使用。
# 把 ``teacher_forcing_ratio`` 提高到更多地使用它。
#

teacher_forcing_ratio = 0.5


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


######################################################################
# 这是一个辅助函数，用于在给定消逝的时间和进度百分比的情况下打印经过的时间和估计的剩余时间。
#

import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


######################################################################
# 整个训练过程如下 :
#
# -  开启一个计时器
# -  初始化优化器和准则
# -  创建一组训练对(training pairs)
# -  准备一个空损失数组用于绘图
#
# 然后我们调用 ``train`` 很多次，并且偶尔打印进度 (样例百分比, 目前所花费时间, 估计剩余时间) 和 平均损失。
#

def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(pairs))
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)


######################################################################
# 绘制结果
# ----------------
#
# 绘图是使用matplotlib完成的，使用训练时保存的损失值 ``plot_losses`` 的数组。
#

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


######################################################################
# 评估
# ==========
#
# 评估过程的大部分与训练是相同的，但没有目标函数，所以我们简单地在每一步将解码器的预测反馈给自己。
# 每次它预测一个单词时，我们都会将它添加到输出字符串中，如果它预测到EOS令牌，就会停止。
# 我们还存储解码器的注意力输出，以供以后显示。
#

def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


######################################################################
# 我们可以从训练集中对随机句子进行评估，并打印出输入、目标和输出，
# 从而做出一些主观的质量判断：
#

def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


######################################################################
# 训练 和 评估
# =======================
#
# 有了所有这些辅助函数(它看起来像是额外的工作，但它使运行多个实验更容易)，
# 我们实际上就可以初始化一个网络并开始训练。
#
# 记住，输入的句子是经过严格过滤的。对于这个小数据集，我们可以使用相对较小的网络，
# 包括256个隐藏节点和一个GRU层。在MacBook的CPU上大约40分钟后，我们将得到一些合理的结果。
#
# .. Note::
#    如果你运行这个notebook，你可以训练，中断内核，评估，并在以后继续训练。
#    对编码器和解码器初始化的行进行注释，并再次运行 ``trainIters`` 。
#

hidden_size = 256
encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

trainIters(encoder1, attn_decoder1, 75000, print_every=5000)

######################################################################
#

evaluateRandomly(encoder1, attn_decoder1)


######################################################################
# 可视化注意力网络输出
# ---------------------
#
# 注意力机制的一个有用的特性是其高度可解释的输出。由于它用于加权编码器输出的特定部分，
# 因此我们可以查看在每个时间步网络最聚焦的输出部分是哪里。
#
# 您只需简单运行 ``plt.matshow(attentions)`` 就可以将注意力输出显示为一个矩阵，
# 其中矩阵的列是输入步(input steps)，矩阵的行是输出步(output steps)：
#

output_words, attentions = evaluate(
    encoder1, attn_decoder1, "je suis trop froid .")
plt.matshow(attentions.numpy())


######################################################################
# 为了获得更好的可视化体验，我们将做额外的工作，添加轴和标签：
#

def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def evaluateAndShowAttention(input_sentence):
    output_words, attentions = evaluate(
        encoder1, attn_decoder1, input_sentence)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions)


evaluateAndShowAttention("elle a cinq ans de moins que moi .")

evaluateAndShowAttention("elle est trop petit .")

evaluateAndShowAttention("je ne crains pas de mourir .")

evaluateAndShowAttention("c est un jeune directeur plein de talent .")


######################################################################
# 练习
# =========
#
# -  尝试使用不同的数据集
#
#    -  Another language pair
#    -  Human → Machine (e.g. IOT commands)
#    -  Chat → Response
#    -  Question → Answer
#
# -  将嵌入替换为预先训练过的单词嵌入，例如 word2vec 或 GloVe。
# -  尝试用更多的层，更多的隐藏单元，更多的句子。比较训练时间和结果。
# -  如果您使用一个翻译文件，其中 pairs 有两个相同的短语(``I am test \t I am test``)，
#    则可以将其用作自动编码器。试试这个:
#
#    -  作为自编码器进行训练
#    -  只保存编码器网络
#    -  训练一个解码器用于翻译
#
