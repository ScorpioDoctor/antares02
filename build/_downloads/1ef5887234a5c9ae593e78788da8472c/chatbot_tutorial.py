# -*- coding: utf-8 -*-

"""
聊天机器人教程
================
**翻译者**: `Antares博士 <http://www.studyai.com/antares>`_
"""


######################################################################
# 在本教程中，我们将探索一个有趣而有意思的用例-递归序列到序列模型(recurrent sequence-to-sequence models)的用例。
# 我们将训练一个简单的聊天机器人使用康奈尔电影剧本-对话语料库
# (`Cornell Movie-Dialogs Corpus <https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html>`__)。
#
# 会话模型(Conversational models)是人工智能研究的一个热点。聊天机器人(Chatbots)可以在多种场景中找到，包括客户服务应用程序和在线帮助台。
# 这些机器人通常由基于检索的模型(retrieval-based models)进行驱动，这些模型输出对某些形式的问题的预定义的回答。
# 在一个高度受限的领域，比如一个公司的IT服务台，这些模型可能已经足够了，但是对于更一般的用例来说，它们还不够健壮。
# 教一台机器在多个领域与人进行有意义的对话是一个远未解决的研究问题。最近，深度学习的热潮已经允许强大的生成模型，
# 如谷歌的神经会话模型(`Neural Conversational Model <https://arxiv.org/abs/1506.05869>`__)，
# 这标志着向多域生成会话模型(multi-domain generative conversational models)迈出了一大步。在本教程中，我们将在PyTorch中实现这种模型。
#
# .. figure:: /_static/img/chatbot/bot.png
#    :align: center
#    :alt: bot
#
# .. code:: python
#
#   > hello?
#   Bot: hello .
#   > where am I?
#   Bot: you re in a hospital .
#   > who are you?
#   Bot: i m a lawyer .
#   > how are you doing?
#   Bot: i m fine .
#   > are you my friend?
#   Bot: no .
#   > you're under arrest
#   Bot: i m trying to help you !
#   > i'm just kidding
#   Bot: i m sorry .
#   > where are you from?
#   Bot: san francisco .
#   > it's time for me to leave
#   Bot: i know .
#   > goodbye
#   Bot: goodbye .
#
# **本教程亮点**
#
# -  加载和预处理 `Cornell Movie-Dialogs Corpus <https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html>`__
#    数据集
# -  实现一个带有 `Luong 注意力机制(s) <https://arxiv.org/abs/1508.04025>`__ 的序列到序列模型(sequence-to-sequence model)
# -  使用 mini-batches 联合训练编码器和解码器模型(encoder and decoder models)
# -  实现贪婪搜索解码模块
# -  与训练好的chatbot进行交互
#
# **鸣谢**
#
# 本教程借用下列来源的代码:
#
# 1) Yuan-Kuei Wu’s pytorch-chatbot implementation:
#    https://github.com/ywk991112/pytorch-chatbot
#
# 2) Sean Robertson’s practical-pytorch seq2seq-translation example:
#    https://github.com/spro/practical-pytorch/tree/master/seq2seq-translation
#
# 3) FloydHub’s Cornell Movie Corpus preprocessing code:
#    https://github.com/floydhub/textutil-preprocess-cornell-movie-corpus
#


######################################################################
# 预备工作
# ------------
#
# 首先，从 `这里 <https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html>`__ 下载数据的ZIP文件，
# 并将其放在当前目录下的 ``data/`` 目录中。
#
# 之后，我们要导入一些必要的包。
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math


USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


######################################################################
# 加载 & 预处理数据
# ----------------------
#
# 下一步是重新格式化数据文件，并将数据加载到我们可以使用的结构中。
#
# `Cornell 电影对话语料库 <https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html>`__
# 是一个电影人物对话的丰富的数据集。
#
# -  10,292 对电影角色之间的220，579次会话交流
# -  来自 617 部电影的9,035个人物角色
# -  304,713 total utterances(话语)
#
# 该数据集庞大多样，语言形式、时间周期、情感等都有很大的变化。
# 我们希望这种多样性使我们的模型对多种形式的输入和查询都有很强的抵抗力。
#
# 首先，我们将查看数据文件的一些行，以查看原始格式。
#

corpus_name = "cornell movie-dialogs corpus"
corpus = os.path.join("./data", corpus_name)

def printLines(file, n=10):
    with open(file, 'rb') as datafile:
        lines = datafile.readlines()
    for line in lines[:n]:
        print(line)

printLines(os.path.join(corpus, "movie_lines.txt"))


######################################################################
# 创建格式化数据文件
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 为了方便起见，我们将创建一个格式良好的数据文件，其中每一行包含一个tab分隔
# 的查询语句(*query sentence*)和一个响应语句对(*response sentence*)。
#
# 以下函数有助于解析原始的 *movie_lines.txt* 数据文件。
#
# -  ``loadLines`` ：将文件的每一行拆分为字段字典(lineID, characterID, movieID, character, text)。
# -  ``loadConversations`` ：基于 *movie_conversations.txt* 把从 ``loadLines`` 得到的lines的字段分组为回话
# -  ``extractSentencePairs`` 从会话中抽取句子对
#

# 将文件的每一行拆分为字段字典。
def loadLines(fileName, fields):
    lines = {}
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            # Extract fields
            lineObj = {}
            for i, field in enumerate(fields):
                lineObj[field] = values[i]
            lines[lineObj['lineID']] = lineObj
    return lines


# 基于 *movie_conversations.txt* 把从 ``loadLines`` 得到的lines的字段分组为回话
def loadConversations(fileName, lines, fields):
    conversations = []
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            # Extract fields
            convObj = {}
            for i, field in enumerate(fields):
                convObj[field] = values[i]
            # Convert string to list (convObj["utteranceIDs"] == "['L598485', 'L598486', ...]")
            lineIds = eval(convObj["utteranceIDs"])
            # Reassemble lines
            convObj["lines"] = []
            for lineId in lineIds:
                convObj["lines"].append(lines[lineId])
            conversations.append(convObj)
    return conversations


# 从会话中抽取句子对
def extractSentencePairs(conversations):
    qa_pairs = []
    for conversation in conversations:
        # Iterate over all the lines of the conversation
        for i in range(len(conversation["lines"]) - 1):  # We ignore the last line (no answer for it)
            inputLine = conversation["lines"][i]["text"].strip()
            targetLine = conversation["lines"][i+1]["text"].strip()
            # Filter wrong samples (if one of the lists is empty)
            if inputLine and targetLine:
                qa_pairs.append([inputLine, targetLine])
    return qa_pairs


######################################################################
# 现在我们将调用这些函数并创建文件。我们将其命名为 *formatted_movie_lines.txt* 。
#

# 定义指向新文件的路径
datafile = os.path.join(corpus, "formatted_movie_lines.txt")

delimiter = '\t'
# Unescape the delimiter
delimiter = str(codecs.decode(delimiter, "unicode_escape"))

# 初始化 lines dict, conversations list, and field ids
lines = {}
conversations = []
MOVIE_LINES_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
MOVIE_CONVERSATIONS_FIELDS = ["character1ID", "character2ID", "movieID", "utteranceIDs"]

# 加载 lines 并处理 conversations
print("\nProcessing corpus...")
lines = loadLines(os.path.join(corpus, "movie_lines.txt"), MOVIE_LINES_FIELDS)
print("\nLoading conversations...")
conversations = loadConversations(os.path.join(corpus, "movie_conversations.txt"),
                                  lines, MOVIE_CONVERSATIONS_FIELDS)

# 写入到新的 csv 文件
print("\nWriting newly formatted file...")
with open(datafile, 'w', encoding='utf-8') as outputfile:
    writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')
    for pair in extractSentencePairs(conversations):
        writer.writerow(pair)

# 打印输出 lines 的样本
print("\nSample lines from file:")
printLines(datafile)


######################################################################
# 加载 并 裁剪 数据
# ~~~~~~~~~~~~~~~~~~
#
# 我们下一步的工作是创建一个词汇表，并将 查询/响应语句对 加载到内存中。
#
# 注意，我们处理的是 **words** 序列，它们没有隐式映射到离散的数值空间。
# 因此，我们必须通过将我们在DataSet中遇到的每个唯一单词映射到一个索引值来创建一个索引。
#
# 为此，我们定义了一个 ``Voc`` 类，它保持从单词到索引的映射、索引到单词的反向映射、每个单词的计数和总单词计数。
# 该类提供了将单词添加到词汇表(``addWord``)、在句子中添加所有单词(``addSentence``)
# 和修剪少见单词(``trim``)的方法。稍后更多关于修剪的内容。
#

# 默认的单词标记(word tokens)
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # 句子起始的 token
EOS_token = 2  # 句子结束的 token

class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3  # Count SOS, EOS, PAD

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # 移除那些出现次数低于某个阈值的单词
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3 # Count default tokens

        for word in keep_words:
            self.addWord(word)


######################################################################
# 现在，我们可以组装我们的词汇表和查询/响应句子对。在准备使用这些数据之前，
# 我们必须执行一些预处理。
#
# 首先，我们必须使用 ``unicodeToAscii`` 将Unicode字符串转换为ASCII。
# 接下来，我们应该将所有字母转换为小写，并修剪除基本标点符号(``normalizeString``)以外的所有非字母字符。
# 最后，为了帮助训练收敛，我们将过滤出长度大于 ``MAX_LENGTH`` 阈值的句子(``filterPairs``)。
#

MAX_LENGTH = 10  # Maximum sentence length to consider

# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

# Read query/response pairs and return a voc object
def readVocs(datafile, corpus_name):
    print("Reading lines...")
    # Read the file and split into lines
    lines = open(datafile, encoding='utf-8').\
        read().strip().split('\n')
    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    voc = Voc(corpus_name)
    return voc, pairs

# Returns True iff both sentences in a pair 'p' are under the MAX_LENGTH threshold
def filterPair(p):
    # Input sequences need to preserve the last word for EOS token
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH

# Filter pairs using filterPair condition
def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

# Using the functions defined above, return a populated voc object and pairs list
def loadPrepareData(corpus, corpus_name, datafile, save_dir):
    print("Start preparing training data ...")
    voc, pairs = readVocs(datafile, corpus_name)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    print("Counted words:", voc.num_words)
    return voc, pairs


# Load/Assemble voc and pairs
save_dir = os.path.join("data", "save")
voc, pairs = loadPrepareData(corpus, corpus_name, datafile, save_dir)
# Print some pairs to validate
print("\npairs:")
for pair in pairs[:10]:
    print(pair)


######################################################################
# 另一种有利于在训练中实现更快收敛的策略是将很少使用的单词从我们的词汇表中删掉。
# 减少特征数量也会降低模型必须学习近似的函数的难度。我们将采取两步行动：
#
# 1) 使用函数 ``voc.trim`` 裁剪出现次数少于 ``MIN_COUNT`` 阈值的单词。
#
# 2) Filter out pairs with trimmed words.
#

MIN_COUNT = 3    # 用于修剪的最小单词量阈值

def trimRareWords(voc, pairs, MIN_COUNT):
    # Trim words used under the MIN_COUNT from the voc
    voc.trim(MIN_COUNT)
    # Filter out pairs with trimmed words
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        # Check input sentence
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        # Check output sentence
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break

        # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs


# Trim voc and pairs
pairs = trimRareWords(voc, pairs, MIN_COUNT)


######################################################################
# 为模型准备数据
# -----------------------
#
# 虽然我们已经付出了很大的努力来准备和装饰我们的数据到一个很好的词汇表对象和句子对列表，
# 我们的模型最终期望的是数值型的torch tensors作为输入。
# 一种为模型准备处理数据的方法可以在(`seq2seq translation tutorial 
# <https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html>`__)中找到。
# 在该教程中，我们使用的 batch size是1，这意味着我们所要做的就是将句子对中的单词
# 从词汇表中转换成相应的索引，并将其提供给模型。
# 
# 然而，如果您有兴趣加快训练和/或希望利用GPU并行化功能，您将需要训练 mini-batches 。
# 
# 使用mini-batches也意味着我们必须注意批次中句子长度的变化。为了在同一批次中容纳不同大小的句子，
# 我们将把批量输入的张量的shape都变成 *(max_length, batch_size)*，
# 那些小于 *max_length* 的句子将在 *EOS_Token* 之后进行零填充补全。
#
# 如果我们简单地 通过把单词转换为索引(\ ``indexesFromSentence``) 和zero-pad 将我们的英语句子转换为张量，
# 那么我们的张量的shape为 *(batch_size, max_length)* ，索引第一个维度将返回在所有时间步中的一个完整序列。
# 然而，我们需要能够沿着时间索引我们的batch，并跨越batch中的所有序列。因此，我们将输入batch的shape转换
# 为 *(max_length, batch_size)* ，以便跨第一维索引返回batch中所有句子的时间步长。
# 我们在 ``zeroPadding`` 函数中隐式地处理这个转置。
# 
# .. figure:: /_static/img/chatbot/seq2seq_batches.png
#    :align: center
#    :alt: batches
#
# ``inputVar`` 函数处理句子转换为张量的过程，最终创建一个形状正确的零填充张量。
# 它还返回batch中每个序列的 ``lengths`` 的张量，稍后将传递给我们的解码器。
#
# ``outputVar`` 函数与 ``inputVar`` 函数执行相似的功能, 但是它返回的不是 ``lengths`` 张量，而是
# 二值掩模张量(binary mask tensor) 和 最大目标句子长度(maximum target sentence length)。
# 二值掩模张量的shape与输出目标张量的shape是相同的，但里面每一个元素除了 *PAD_token* 是0之外，其余地方都是1。
#
# ``batch2TrainData`` 简单的接收 a bunch of pairs，并使用上述函数返回输入张量和目标张量。
#

def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]


def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

# Returns padded input sequence tensor and lengths
def inputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths

# Returns padded target sequence tensor, padding mask, and max target length
def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.ByteTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len

# Returns all items for a given batch of pairs
def batch2TrainData(voc, pair_batch):
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len


# Example for validation
small_batch_size = 5
batches = batch2TrainData(voc, [random.choice(pairs) for _ in range(small_batch_size)])
input_variable, lengths, target_variable, mask, max_target_len = batches

print("input_variable:", input_variable)
print("lengths:", lengths)
print("target_variable:", target_variable)
print("mask:", mask)
print("max_target_len:", max_target_len)


######################################################################
# 定义模型
# -------------
#
# Seq2Seq Model
# ~~~~~~~~~~~~~
#
# 我们的聊天机器人的大脑是一个序列到序列(Seq2seq)模型。seq2seq模型的目标是以可变长度序列作为输入，
# 使用固定大小的模型(fixed-sized model)返回可变长度序列作为输出。
#
# `Sutskever 等人 <https://arxiv.org/abs/1409.3215>`__ 发现利用两个分开的
# 递归神经网络(recurrent neural nets)可以完成这一任务。
# 一个RNN充当编码器(**encoder**)，将可变长度的输入序列编码成一个固定长度的上下文向量。
# 理论上，这个上下文向量(RNN的最后一个隐藏层)将包含 输入到bot的查询语句 的语义信息。
# 第二个RNN是解码器(**decoder**)，它接受输入单词和上下文向量，并返回序列中下一个单词的猜测和
# 下一次迭代使用的隐藏状态。
#
# .. figure:: /_static/img/chatbot/seq2seq_ts.png
#    :align: center
#    :alt: model
#
# 图片来源:
# https://jeddy92.github.io/JEddy92.github.io/ts_seq2seq_intro/
#


######################################################################
# 编码器(Encoder)
# ~~~~~~~~~~~~~~~~~
#
# 编码器RNN一次迭代输入句子的一个标记(e.g. word)，在每一时间步输出一个 “output” vector
# 和 “hidden state” vector。然后将隐藏状态向量传递到下一时间步，同时记录输出向量。
# 编码器将在序列中的每一点上看到的上下文转换为高维空间中的一组点，
# 解码器将使用这些点为给定任务生成有意义的输出。
#
# 我们编码器的核心是一个由 `CHO 等人 <https://arxiv.org/pdf/1406.1078v3.pdf>`__ 
# 2014年发明的多层的门控递归单元(multi-layered Gated Recurrent Unit)。
# 我们将使用GRU的双向变体(bidirectional variant)，
# 这意味着本质上有两个独立的RNN：一个以正常顺序输入序列，另一个以反向顺序输入序列。
# 每个网络的输出在每个时间步被求和。使用双向GRU将为我们提供编码过去上下文和未来上下文的优势。
#
# Bidirectional RNN:
#
# .. figure:: /_static/img/chatbot/RNN-bidirectional.png
#    :width: 70%
#    :align: center
#    :alt: rnn_bidir
#
# 图片来源: http://colah.github.io/posts/2015-09-NN-Types-FP/
#
# 注意，嵌入层(``embedding`` layer)用于在任意大小的特征空间中编码我们的单词索引。对于我们的模型，
# 这个层将把每个单词映射到一个大小为 *hidden_size* 的特征空间。
# 在训练时，这些值应该编码同义词之间的语义相似性(semantic similarity)。
#
# 最后，如果将一批填充好的序列传递给RNN module，则必须使用 
# ``torch.nn.utils.rnn.pack_padded_sequence`` 和 ``torch.nn.utils.rnn.pad_packed_sequence`` 
# 分别对RNN传递周围的填充进行打包(pack)和解压(unpack)。
#
# **计算图:**
#
#    1) 将单词索引转换为嵌入(Convert word indexes to embeddings)。
#    2) 为 RNN module 打包填充好的批量序列
#    3) 前向传递使其通过GRU.
#    4) 去除填充值(Unpack padding).
#    5) 把双向GRU的输出加起来
#    6) 返回输出和最后的隐藏层
#
# **输入:**
#
# -  ``input_seq``: 输入序列的batch; shape=\ *(max_length, batch_size)*
# -  ``input_lengths``: 对应于batch中的每一个句子的序列长度的列表; shape=\ *(batch_size)*
# -  ``hidden``: 隐藏状态; shape=\ *(n_layers x num_directions, batch_size, hidden_size)*
#
# **输出:**
#
# -  ``outputs``: GRU的最后一个隐藏层的输出特征(双向输出之和); shape=\ *(max_length, batch_size, hidden_size)*
# -  ``hidden``: 从GRU更新的隐藏状态; shape=\ *(n_layers x num_directions, batch_size, hidden_size)*
#
#

class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)
        # Pack padded batch of sequences for RNN module
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)
        # Unpack padding
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        # Return output and final hidden state
        return outputs, hidden


######################################################################
# 解码器(Decoder)
# ~~~~~~~~~~~~~~~~~~~~
#
# 解码器RNN以令牌接令牌(token-by-token)的方式生成响应语句(reponse sentance)。
# 它使用编码器的上下文向量和内部隐藏状态来生成序列中的下一个单词。
# 它连续生成单词，直到输出 *EOS_token* ，表示句子的结尾。
# 一个普通的seq2seq解码器的一个常见问题是：
# 如果我们依靠上下文向量来编码整个输入序列的意义，我们很可能会有信息丢失。
# 尤其是在处理长输入序列时，大大限制了解码器的性能。
# 
# 与此作斗争，`Bahdanau et al. <https://arxiv.org/abs/1409.0473>`__ 
# 创建了一个“注意机制(attention mechanism)”，
# 允许解码器注意输入序列的某些部分，而不是在每一步都使用整个固定的上下文。
#
# 在较高的层次上，注意力是利用解码器的当前隐藏状态和编码器的输出来计算的。
# 输出注意权重(output attention weights)与输入序列具有相同的shape，
# 可以将它们与编码器输出相乘，给出一个加权和，表示编码器输出中需要注意的部分。
# `Sean Robertson’s <https://github.com/spro>`__  的图片很好地描述了这一点:
#
# .. figure:: /_static/img/chatbot/attn2.png
#    :align: center
#    :alt: attn2
#
# `Luong et al. <https://arxiv.org/abs/1508.04025>`__ 通过创造“全局注意(Global attention)” 在Bahdanau等人的基础上加以改进。
# 关键的区别在于，对于“全局注意”，我们考虑编码器的所有隐藏状态，
# 而不是Bahdanau等人的“局部注意”，后者只考虑编码器的当前时间步的隐藏状态。
# 另一个不同之处在于，对于“全局注意”，我们只从当前的时间步中使用解码器的隐藏状态来计算注意力权重或能量。
# Bahdanau等人的注意力计算需要从前一时间步了解解码器的状态。另外，Luong等人提供各种方法来计算编码器输出
# 和解码器输出之间的注意力能量，称为 “score functions” :
#
# .. figure:: /_static/img/chatbot/scores.png
#    :width: 60%
#    :align: center
#    :alt: scores
#
# 其中 :math:`h_t` = current target decoder state 和 :math:`\bar{h}_s` = all encoder states.
#
# 总体而言, 全局注意机制可以由下面的这个图来说明。注意到我们将实现 “Attention Layer” 
# 作为单独分开的 ``nn.Module`` ，称之为 ``Attn`` 。该module的输出是一个shape为 
# *(batch_size, 1, max_length)* 的 softmax normalized weights tensor。
#
# .. figure:: /_static/img/chatbot/global_attn.png
#    :align: center
#    :width: 60%
#    :alt: global_attn
#

# Luong attention layer
class Attn(torch.nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = torch.nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = torch.nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


######################################################################
# 现在我们已经定义了我们的注意子模块(attention submodule)，我们可以实现实际的解码器模型。
# 对于解码器，我们将一次一个时间步手动输入我们的batch。这意味着我们嵌入的单词张量和GRU输出都将具有
# shape 为 *(1, batch_size, hidden_size)* 。
#
# **计算图:**
#
#    1) 获取当前输入单词的嵌入(embedding)
#    2) 前向传递通过双向GRU.
#    3) 从上面第(2)步的当前GRU的输出 计算注意力权重.
#    4) 把注意力权重乘到编码器输出上来获得新的"加权和"上下文向量.
#    5) 使用 Luong eq. 5 串接(Concatenate)加权上下文向量和GRU输出。
#    6) 使用 Luong eq. 6 预测下一个单词(没有 softmax).
#    7) 返回输出和最终的隐藏状态.
#
# **输入:**
#
# -  ``input_step``: one time step (one word) of input sequence batch; shape=\ *(1, batch_size)*
# -  ``last_hidden``: GRU的最终隐藏层; shape=\ *(n_layers x num_directions, batch_size, hidden_size)*
# -  ``encoder_outputs``: 编码器的输出; shape=\ *(max_length, batch_size, hidden_size)*
#
# **输出:**
#
# -  ``output``: 给定 解码得到的序列中的每个单词是正确的下一个单词的概率 下 的 softmax normalized tensor ;
#    shape=\ *(batch_size, voc.num_words)*
# -  ``hidden``: GRU的最终隐藏状态; shape=\ *(n_layers x num_directions, batch_size, hidden_size)*
#

class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        # Forward through unidirectional GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        # Return output and final hidden state
        return output, hidden


######################################################################
# 定义训练步骤
# -------------------------
#
# 掩模损失(Masked loss)
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 由于我们处理的是一个批次的填充序列(padded sequences)，所以在计算损失时不能简单地考虑张量的所有元素。
# 根据解码器输出张量、目标张量和描述目标张量填充的二值掩模张量(binary mask tensor)，
# 定义了 ``maskNLLLoss`` 来计算我们的损失。此损失函数计算与掩模张量中的 *1* 对应的元素的平均负对数似然(average negative log likelihood)。
#

def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()


######################################################################
# 单个训练迭代
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#
# ``train`` 函数包含单个训练迭代(单个输入批次)的算法。
#
# 我们将使用一些巧妙的技巧来帮助收敛:
#
# -  第一个诀窍是利用老师的强迫(**teacher forcing**)。这意味着，在一定的概率下，根据 ``teacher_forcing_ratio`` 设置，
#    我们使用当前目标词作为解码器的下一个输入，而不是使用解码器当前的猜测。
#    这种技术作为解码器的 training wheels，有助于更有效的训练。
#    然而，在推理过程中，教师的强迫会导致模型的不稳定性，
#    因为在训练过程中，解码器可能没有足够的机会真正地完成自己的输出序列。
#    因此，我们必须注意如何设置 ``teacher_forcing_ratio`` ，而不是被快速收敛所愚弄。
#
# -  我们实现的第二个技巧是梯度裁剪 **(gradient clipping)** 。这是对付“爆炸梯度(exploding gradient)”问题的一种常用技术。
#    从本质上说，通过将梯度裁剪或阈值化到一个最大值，我们可以防止梯度指数增长， 或者在代价函数中溢出(NaN)或过陡的悬崖。
#
# .. figure:: /_static/img/chatbot/grad_clip.png
#    :align: center
#    :width: 60%
#    :alt: grad_clip
#
# 图片来源: Goodfellow et al. *Deep Learning*. 2016. http://www.deeplearningbook.org/
#
# **Sequence of Operations:**
#
#    1) 把整个输入批次前向传递使其通过编码器.
#    2) 以 SOS_token 初始化解码器输入, 和 以编码器的最终隐藏状态初始化 hidden state .
#    3) 一次一个时间步 前向传递输入批次序列 使其通过解码器.
#    4) 如果使用教师强迫: 把当前状态设置为下一个解码器输入; 如果不用: 把当前解码器的输出设置为下一个解码器输入.
#    5) 计算并积累损失.
#    6) 执行向后传播.
#    7) 裁剪梯度.
#    8) 更新编码器和解码器的模型参数.
#
#
# .. Note ::
#
#   PyTorch的RNN modules(``RNN``, ``LSTM``, ``GRU``)可以像其他非递归层(non-recurrent layers)一样使用，
#   只需将整个输入序列(或批处理序列)传递给它们。
#   我们在 ``encoder`` 中就是这样使用 ``GRU`` 层的。实际情况是，在底层有一个迭代过程在每个时间步上循环计算隐藏状态。
#   或者，您可以一次一个时间步的运行这些modules。在这种情况下，我们在训练过程中手动循环序列，就像我们必须为 ``decoder`` 模型所做的那样。
#   只要您维护这些modules的正确概念模型，实现顺序模型就会非常简单。
#
#


def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding,
          encoder_optimizer, decoder_optimizer, batch_size, clip, max_length=MAX_LENGTH):

    # Zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Set device options
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)

    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0

    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # Determine if we are using teacher forcing this iteration
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # Forward batch of sequences through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # Teacher forcing: next input is current target
            decoder_input = target_variable[t].view(1, -1)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    # Perform backpropatation
    loss.backward()

    # Clip gradients: gradients are modified in place
    _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Adjust model weights
    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals


######################################################################
# 多个训练迭代
# ~~~~~~~~~~~~~~~~~~~
#
# 最后，是时候将整个训练过程与数据结合起来了。给定传递的模型、优化器、数据等，
# ``trainIters`` 函数负责训练的 ``n_iterations`` 次迭代。
# 这个函数是相当不言自明的，因为我们已经用 ``train`` 函数做了繁重的工作。
#
# 需要注意的是，当我们保存模型时，我们保存了一个tarball，它包含编码器和解码器的 state_dicts (参数)、
# 优化器的 state_dicts、损失、迭代等等。以这种方式保存模型将使我们对检查点(checkpoint)具有最终的灵活性。
# 在加载检查点之后，我们将能够使用模型参数来运行推理，或者我们可以继续在我们停止的地方进行训练。
#

def trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size, print_every, save_every, clip, corpus_name, loadFilename):

    # Load batches for each iteration
    training_batches = [batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size)])
                      for _ in range(n_iteration)]

    # Initializations
    print('Initializing ...')
    start_iteration = 1
    print_loss = 0
    if loadFilename:
        start_iteration = checkpoint['iteration'] + 1

    # Training loop
    print("Training...")
    for iteration in range(start_iteration, n_iteration + 1):
        training_batch = training_batches[iteration - 1]
        # Extract fields from batch
        input_variable, lengths, target_variable, mask, max_target_len = training_batch

        # Run a training iteration with batch
        loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                     decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip)
        print_loss += loss

        # Print progress
        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration, iteration / n_iteration * 100, print_loss_avg))
            print_loss = 0

        # Save checkpoint
        if (iteration % save_every == 0):
            directory = os.path.join(save_dir, model_name, corpus_name, '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'voc_dict': voc.__dict__,
                'embedding': embedding.state_dict()
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))


######################################################################
# 定义评估
# -----------------
#
# 在训练了一个模型之后，我们希望自己能够和机器人交谈。首先，我们必须定义我们希望模型如何解码编码的输入。
#
# 贪婪解码
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 贪婪解码(Greedy decoding)是我们在 **不** 使用教师强迫(teacher forcing)的情况下，在训练过程中使用的一种解码方法。
# 换句话说，对于每个时间步，我们只需选择具有最高softmax值的 ``decoder_output`` 中的单词。
# 这种解码方法在单个时间步水平(single time-step level)上是最优的. 
#
# 为了方便贪婪解码操作，我们定义了 ``GreedySearchDecoder`` 类。当运行时，
# 该类的对象接受shape为 *(input_seq length, 1)* 的输入序列(``input_seq``)、标量输入长度(``input_length``)张量，
# 以及 ``max_length`` 来限制响应语句的长度。输入句子使用以下计算图进行计算：
#
# **计算图:**
#
#    1) 向前传递输入使其通过编码器模型.
#    2) 准备编码器的最终隐藏层作为解码器第一隐藏层的输入.
#    3) 初始化解码器的第一个输入为 SOS_token.
#    4) 初始化要追加到解码出的单词上的张量.
#    5) 一次迭代解码一个单词标记(word token):
#        a) 向前传递通过解码器.
#        b) 获得最有可能的单词标记(word token)和对应的(softmax score).
#        c) 记录 token 和 score.
#        d) 准备把当前 token 作为下一个解码器输入.
#    6) 返回本次迭代过程得到的 word tokens 和 scores 的集合(collections).
#

class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:decoder.n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens, all_scores


######################################################################
# 评估我的文本
# ~~~~~~~~~~~~~~~~
#
# 现在我们已经定义了解码方法，我们可以编写用于评估(evaluate)字符串输入语句的函数。
# ``evaluate`` 函数管理处理输入语句的低级过程。我们首先将句子格式化为单词索引的输入batch(其中 *batch_size==1* )。
# 我们将句子中的单词转换成相应的索引，并调换维度顺序来为我们的模型准备张量。我们还创建了一个 ``lengths`` 张量，
# 它包含我们输入句子的长度。在这种情况下，``lengths`` 是标量，因为我们一次只计算一个句子(batch_size==1)。
# 然后，利用 ``GreedySearchDecoder`` 对象(``searcher``)得到解码后的响应语句张量。
# 最后，我们将响应的索引转换为单词，并返回解码得到的单词列表。
#
# ``evaluateInput`` 充当聊天机器人的用户界面。调用时将显示输入文本框，我们可以在其中输入查询语句。
# 输入句子并按 *Enter* 键后，我们的文本以与训练数据相同的方式归一化，
# 并最终被输入到 ``evaluate`` 函数中，得到解码后的输出语句。我们循环这个过程，
# 以便我们可以一直与我们的机器人聊天，直到我们输入“q” 或 “quit”。
#
# 最后，如果输入的句子中包含一个不在词汇表中的单词，我们将通过打印错误消息并提示
# 用户输入另一个句子来优雅地处理这个问题。
#

def evaluate(encoder, decoder, searcher, voc, sentence, max_length=MAX_LENGTH):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexesFromSentence(voc, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


def evaluateInput(encoder, decoder, searcher, voc):
    input_sentence = ''
    while(1):
        try:
            # Get input sentence
            input_sentence = input('> ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit': break
            # Normalize sentence
            input_sentence = normalizeString(input_sentence)
            # Evaluate sentence
            output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
            # Format and print response sentence
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            print('Bot:', ' '.join(output_words))

        except KeyError:
            print("Error: Encountered unknown word.")


######################################################################
# 运行模型
# ---------
#
# 最后，是时候运行我们的模型了!
#
# 无论我们是要训练还是测试聊天机器人模型，我们都必须初始化单个的编解码模型。
# 在下面的块中，我们设置了所需的配置，选择从头开始或设置要加载的检查点，
# 并构建和初始化模型。可以随意使用不同的模型配置来优化性能。
#

# Configure models
model_name = 'cb_model'
attn_model = 'dot'
#attn_model = 'general'
#attn_model = 'concat'
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64

# Set checkpoint to load from; set to None if starting from scratch
loadFilename = None
checkpoint_iter = 4000
#loadFilename = os.path.join(save_dir, model_name, corpus_name,
#                            '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
#                            '{}_checkpoint.tar'.format(checkpoint_iter))


# Load model if a loadFilename is provided
if loadFilename:
    # If loading on same machine the model was trained on
    checkpoint = torch.load(loadFilename)
    # If loading a model trained on GPU to CPU
    #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']


print('Building encoder and decoder ...')
# Initialize word embeddings
embedding = nn.Embedding(voc.num_words, hidden_size)
if loadFilename:
    embedding.load_state_dict(embedding_sd)
# Initialize encoder & decoder models
encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
if loadFilename:
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
# Use appropriate device
encoder = encoder.to(device)
decoder = decoder.to(device)
print('Models built and ready to go!')


######################################################################
# 运行训练过程
# ~~~~~~~~~~~~
#
# 如果要训练模型，请运行以下块。
#
# 首先我们设置训练参数，然后初始化我们的优化器，最后我们调用 ``trainIters`` 函数来运行我们的训练迭代。
#

# Configure training/optimization
clip = 50.0
teacher_forcing_ratio = 1.0
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_iteration = 4000
print_every = 1
save_every = 500

# Ensure dropout layers are in train mode
encoder.train()
decoder.train()

# Initialize optimizers
print('Building optimizers ...')
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
if loadFilename:
    encoder_optimizer.load_state_dict(encoder_optimizer_sd)
    decoder_optimizer.load_state_dict(decoder_optimizer_sd)

# Run training iterations
print("Starting Training!")
trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
           embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size,
           print_every, save_every, clip, corpus_name, loadFilename)


######################################################################
# 运行评估过程
# ~~~~~~~~~~~~~~
#
# 若要与模型聊天，请运行以下块。
#

# Set dropout layers to eval mode
encoder.eval()
decoder.eval()

# Initialize search module
searcher = GreedySearchDecoder(encoder, decoder)

# Begin chatting (uncomment and run the following line to begin)
# evaluateInput(encoder, decoder, searcher, voc)


######################################################################
# 总结
# ----------
#
# 伙计们，这件事就到此为止了。恭喜你，你现在知道建立一个生成式聊天机器人模型的基本原理了！
# 如果您感兴趣，您可以尝试裁剪(tailoring)聊天机器人的行为，方法是调整模型和训练参数，
# 并自定义您训练模型的数据。
#
# 查看PyTorch中的其他教程，以获得更酷的深度学习应用！
#
