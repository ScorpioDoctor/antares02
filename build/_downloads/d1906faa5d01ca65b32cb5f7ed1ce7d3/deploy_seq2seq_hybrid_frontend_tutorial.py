# -*- coding: utf-8 -*-
"""
使用混合前端部署seq2seq模型
==================================================
**翻译者:** `Antares <http://wwww.studyai.com/antares>`_
"""


######################################################################
# 本教程将介绍使用PyTorch的混合前端(Hybrid Frontend)将序列到序列模型(seq2seq model)
# 转换为Torch Script的过程。我们将转换的模型是 `Chatbot 教程 <https://pytorch.org/tutorials/beginner/chatbot_tutorial.html>`__ 的Chatbot模型。
# 您可以将本教程视为Chatbot教程的“第2部分”，并部署您自己的预训练模型，
# 也可以从本文档开始，并使用我们提供的预训练模型。在后一种情况下，您可以参考原始的Chatbot教程，
# 了解有关数据预处理、模型理论和定义以及模型训练的详细信息。
# 
#什么是混合前端(Hybrid Frontend)?
# -----------------------------------
#
# 在基于深度学习的项目的研究和开发阶段，与像PyTorch这样的(**急切的,eager**)、
# 命令式的(imperative)界面进行交互是有利的。
# 这使用户能够编写熟悉的、惯用的Python，允许使用Python数据结构、控制流操作、打印语句和调试实用程序。
# 虽然急切的接口(eager interface)对于研究和实验应用程序是一个有益的工具，但是当涉及到在生产环境(production environment)中
# 部署模型时，基于图形的模型表示(**graph**-based model representation)是非常有益的。
# 推迟的图形表示(deferred graph representation)允许优化，例如无序执行，
# 以及针对硬件体系结构进行高度优化的能力。
# 此外，基于图形的模型表示支持与框架无关的模型导出.
# 
# PyTorch提供了将急切模式代码(eager-mode code)逐步转换为Torch Script的机制，
# 这是Python的一个静态可分析和可优化的子集，Torch使用它来表示独立于Python运行时的深度学习程序。
#
# 用于将急切模式下PyTorch程序转换为Torch Script的API位于torch.jit模块中。
# 该模块有两种核心模式，用于将急切模式模型转换为Torch Script图形表示： **跟踪(tracing)** 和
# **脚本(scripting)**。
# ``torch.jit.trace`` 函数接受一个模块或函数以及一组样例输入。然后，它通过函数或模块运行样例输入，
# 同时跟踪所遇到的计算步骤，并输出执行跟踪操作的基于图形的函数。
# **Tracing** 对于不涉及数据依赖的控制流的简单模块和函数是很好的，例如标准卷积神经网络。
# 但是，如果跟踪具有数据依赖的if语句和循环的函数，则只记录样例输入沿执行路径调用的操作。
# 换句话说，控制流本身不会被捕获。为了转换包含数据依赖控制流的模块和函数，PyTorch还提供了
# 一种脚本(**scripting**)机制。Scripting显式地将模块或函数代码转换为Torch Script，
# 包括所有可能的控制流路径。
# 要使用脚本模式，请确保从 ``torch.jit.ScriptModule`` 基类(而不是 ``torch.nn.Module`` )继承，
# 并将 ``torch.jit.script`` 装饰器添加到Python函数或将 ``torch.jit.script_method`` 
# 装饰器添加到模块的方法(module’s methods)中。
# 使用scripting时要注意的一点是，它只支持Python的一个受限子集。有关支持特性的所有细节，
# 请参阅 Torch Script `语言参考 <https://pytorch.org/docs/master/jit.html>`__ 。
# 为了提供最大的灵活性，
# 可以组合Torch Script的模式来表示整个程序，并且这些技术可以逐步应用。
#
# .. figure:: /_static/img/chatbot/pytorch_workflow.png
#    :align: center
#    :alt: workflow
#



######################################################################
# 鸣谢
# ----------------
#
# 该教程的写作受到了以下这些代码的启发:
#
# 1) Yuan-Kuei Wu 的 Pytorch 聊天机器人实现:
#    https://github.com/ywk991112/pytorch-chatbot
#
# 2) Sean Robertson’s 实操PyTorch之seq2seq翻译的实现:
#    https://github.com/spro/practical-pytorch/tree/master/seq2seq-translation
#
# 3) FloydHub 的香奈儿电影语料库预处理代码:
#    https://github.com/floydhub/textutil-preprocess-cornell-movie-corpus
#


######################################################################
# 准备环境
# -------------------
#
# 首先，我们将导入所需的模块并设置一些常量。如果您计划使用自己的模型，
# 请确保 ``MAX_LENGTH`` 常量设置正确。
# 作为提醒，这个常量定义了训练期间允许的最大句子长度和模型能够产生的最大长度输出。
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import os
import unicodedata
import numpy as np

device = torch.device("cpu")


MAX_LENGTH = 10  # Maximum sentence length

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token


######################################################################
# Model 预览
# --------------
#
# 如前所述，我们使用的模型是 `序列到序列(Seq2seq)模型 <https://arxiv.org/abs/1409.3215>`__ 。
# 当我们的输入是可变长度序列，而且输出也是一个可变长度序列，但不一定是输入的一对一映射 的时候，
# 这种Seq2seq模型经常被使用。
# 一个seq2seq模型由两个协同工作的递归神经网络(RNNs)组成：
# 一个编码器(**encoder**)和一个解码器(decoder)。
#
# .. figure:: /_static/img/chatbot/seq2seq_ts.png
#    :align: center
#    :alt: model
#
#
# 图片来源:
# https://jeddy92.github.io/JEddy92.github.io/ts_seq2seq_intro/
#
# 编码器(Encoder)
# ~~~~~~~~~~~~~~~~~~~~~
#
# 编码器RNN一次迭代输入句子的一个token(e.g. word)，在每一时间步输出 “output”向量 
# 和 “hidden state”向量。
# 然后将隐藏状态向量传递到下一时间步，同时记录输出向量。编码器把在序列中的每一点上看到
# 的上下文转换为高维空间中的一组点，解码器将使用这些点为给定任务生成有意义的输出。
#
# 解码器(Decoder)
# ~~~~~~~~~~~~~~~~~~~
#
# 解码器RNN以token-by-token的方式生成响应语句。它使用编码器的上下文向量和自身内部隐藏状态
# 来生成序列中的下一个单词。它连续生成单词，直到输出 *EOS_token* ，表示句子的结尾。
# 在产生输出时，我们使用解码器中的注意机制(`attention mechanism <https://arxiv.org/abs/1409.0473>`__)
# 来帮助它“注意(pay attention)”输入的某些部分。
# 对于我们的模型，我们实现了 `Luong 等人的 <https://arxiv.org/abs/1508.04025>`__ “全局注意”模块，
# 并将其用作解码模型中的子模块。
#


######################################################################
# 数据处理
# -------------
#
# 虽然我们的模型概念上来说处理的是标记序列(sequences of tokens)，但在现实中，
# 它们和所有机器学习模型一样处理数字。
# 在这种情况下，在训练前建立的模型词汇表中的每个单词都映射到一个整数索引。
# 我们使用一个 ``Voc`` 对象来包含从单词(Word)到索引(index)的映射，以及词汇表中的单词总数。
# 在运行模型之前，我们将加载 ``Voc`` 对象。
#
# 此外，为了能够运行评估，我们必须提供一个处理我们输入的字符串的工具。
# ``normalizeString`` 函数将字符串中的所有字符转换为小写，并删除所有非字母字符
# (non-letter characters)。``indexesFromSentence`` 函数接受一个有若干单词的句子
# 并返回相应的单词索引的序列(sequence of word indexes)。
#

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

    # Remove words below a certain count threshold
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


# Lowercase and remove non-letter characters
def normalizeString(s):
    s = s.lower()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


# Takes string sentence, returns sentence of word indexes
def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]


######################################################################
# 定义 编码器
# --------------
#
# 我们用 ``torch.nn.GRU`` 模块实现编码器的RNN，
# 我们给该模块提供一个批次的句子(词嵌入的向量,vectors of word embeddings)，
# 它会在内部迭代句子，一次一个标记(token)的计算隐藏状态。我们将这个模块初始化为双向的(bidirectional)，
# 这意味着我们有两个独立的GRUs：一个按时间顺序迭代序列，另一个反向迭代。
# 我们最终返回这两个GRUs的输出的和。由于我们的模型是使用batch训练的，
# 所以我们的 ``EncoderRNN`` 模型的 ``forward`` 函数希望接收一个填充过的输入批次(a padded input batch)。
# 为了把可变长度的句子打包到同一个batch，我们允许一个句子中有最大的 *MAX_LENGTH* tokens，
# 并且batch中所有小于 *MAX_LENGTH* tokens 的语句在其结尾处都会用我们专用的 *PAD_token* tokens 进行填充。
# 
# 若要在PyTorch RNN模块中使用 padded batches，我们必须用 
# ``torch.nn.utils.rnn.pack_padded_sequence`` 
# 和 ``torch.nn.utils.rnn.pad_packed_sequence`` 数据转换器 封装我们的前向传递过程。
# 注意， ``forward`` 函数还接受一个 ``input_lengths`` 列表，
# 其中包含batch中每个句子的长度。这个输入在进行填充(padding)时由 
# ``torch.nn.utils.rnn.pack_padded_sequence`` 函数使用。
#
# Hybrid Frontend Notes:
# ~~~~~~~~~~~~~~~~~~~~~~
#
# 由于编码器的 ``forward`` 函数不包含任何依赖于数据的控制流，我们将使用 **tracing** 将其转换为脚本模式。
# 当跟踪模块时，我们可以让模块定义保持原样。我们将在本文档末尾在运行评估之前初始化所有模型。
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
# 定义解码器的注意力模块
# ---------------------------------
#
# 接下来，我们将定义我们的attention模块 (``Attn``)。
# 请注意，此模块将用作我们的解码器模型中的子模块。Luong等人考虑了各种 “score functions”，
# 即接收当前解码器RNN的输出和整个编码器的输出，并返回注意力“能量”。
# 这个注意力能量张量与编码器的输出大小相同，两者最终被乘起来，从而得到一个加权张量，
# 其最大值代表在特定解码时间步的查询语句中最重要的部分。
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
# 定义解码器
# --------------
#
# 类似于 ``EncoderRNN`` ，我们使用 ``torch.nn.GRU`` 模块作为解码器的RNN。
# 然而，这一次，我们使用了单向(unidirectional) GRU。需要注意的是，与编码器不同的是，
# 我们将给解码器RNN一次喂一个单词(word)。
# 我们从获取当前单词的嵌入(embedding)开始，并应用一个 `dropout <https://pytorch.org/docs/stable/nn.html?highlight=dropout#torch.nn.Dropout>`__ 。
# 
# 接下来，我们将嵌入(embedding)和最后隐藏状态 向前传送到GRU，并获得当前GRU输出和隐藏状态。
# 然后，我们使用我们的 ``Attn`` 模块作为一个层来获得注意力权值，
# 然后乘以编码器的输出来获得被注意力加权过的编码器输出。
# 我们使用这个注意力加权编码器输出作为 ``context`` 张量，它表示一个加权和，
# 指示编码器输出的哪些部分需要被注意。在这里，我们使用线性层和Softmax归一化来
# 选择输出序列中的下一个单词。
#
# Hybrid Frontend Notes:
# ~~~~~~~~~~~~~~~~~~~~~~
#
# 与 ``EncoderRNN`` 类似，此模块不包含任何依赖于数据的控制流。 因此，在初始化该模型并加载其参数之后，
# 我们可以再次使用 **tracing** 将该模型转换为 Torch Script 。
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
# 定义评估
# -----------------
#
# 贪婪搜索解码器
# ~~~~~~~~~~~~~~~~~~~~~
#
# 在聊天机器人教程中，我们使用 ``GreedySearchDecoder`` 模块来简化实际的解码过程。
# 该模块把经过训练的编码器和解码器模型作为其属性，驱动编码器对输入语句(a vector of word indexes)
# 进行编码，并一次一个单词(word index)的迭代式的解码一个输出响应序列。
#
# 对输入序列进行编码是简单直接的：只需将整个序列张量及其对应的长度向量向前馈送给编码器 ``encoder`` 。
# 需要注意的是，这个模块一次只处理一个输入序列，而 **不是** 一批序列(batches of sequences)。
# 因此，当常数  **1** 用于声明张量大小时，这对应于batch size为 1。为了解码一个给定的解码器输出，
# 我们必须迭代地向前遍历我们的解码器模型，该模型会输出 对应于每个单词成为解码序列中正确的下一个单词
# 的概率的 Softmax分数。我们把 ``decoder_input``  初始化为包含 *SOS_token* 的张量。
# 在每次通过解码器 ``decoder`` 后，我们贪婪地(*greedily*)将具有最高Softmax概率的单词附加
# 到 ``decoded_words`` 列表中。
# 我们还使用这个词作为下一次迭代的 ``decoder_input`` 。如果 ``decoded_words`` 列表的长度达到
# *MAX_LENGTH* ，或者如果预测的单词是 *EOS_token*，则解码过程终止。
# 
#
# Hybrid Frontend Notes:
# ~~~~~~~~~~~~~~~~~~~~~~
#
# 该模块的 ``forward`` 方法在一次一个单词的解码一个输出序列时涉及到在 :math:`[0, max\_length)` 
# 范围上进行迭代 。因此，我们应该使用 **scripting** 将这个模块转换为Torch Script。
# 与我们可以跟踪(trace)的编码、解码模型不同，
# 我们必须对 ``GreedySearchDecoder`` 模块进行一些必要的更改，以便在没有错误的情况下初始化一个对象。
# 换句话说，我们必须确保我们的模块遵守脚本机制的规则(rules of the scripting mechanism)，
# 并且不使用Torch Script所包含的Python子集之外的任何语言特性。
#
# 为了了解可能需要的一些操作，我们将从聊天机器人教程中的 ``GreedySearchDecoder`` 实现
# 与我们在下面的单元格中使用的实现之间的差异。请注意，以红色突出显示的行是从原始实现中删除的行，
# 以绿色突出显示的行是新增加的。这么做可以很明显的看出我们对原始的 ``GreedySearchDecoder`` 类做了哪些
# 改变。
#
# .. figure:: /_static/img/chatbot/diff.png
#    :align: center
#    :alt: diff
#
# 改动的地方:
# ^^^^^^^^^^^^^
#
# -  ``nn.Module`` -> ``torch.jit.ScriptModule``
#
#    -  为了在模块上使用PyTorch的脚本机制，该模块必须继承 ``torch.jit.ScriptModule`` 。
#
#
# -  把 ``decoder_n_layers`` 添加到构造器参数
#
#    -  这一变化源于这样一个事实：我们传递给这个模块(module)的编码器和解码器模型将是
#       ``TracedModule`` (not ``Module``) 的子模块。因此，我们不能使用 ``decoder.n_layers`` 
#       访问解码器的层数。相反，我们对此进行规划，并在模块构造期间传递此值。
#
#
# -  将新属性存储为常量
#
#    -  在最初的实现中，我们可以在 ``GreedySearchDecoder`` 的 ``forward`` 方法中自由地使用来自
#       周围(全局)范围的变量。但是，既然我们使用的是脚本(scripting)，我们就没有这种自由，
#       因为脚本的假设是，我们不一定要保留Python对象，尤其是在导出时。
#       对此，一个简单的解决方案是将这些值从全局范围存储为构造函数中的模块的属性，
#       并将它们添加到一个名为 ``__constants__`` 的特殊列表中，
#       以便在 ``forward`` 方法中构造图时可以将它们用作文字值(literal values)。
#       这种用法的一个例子是在 **新** 的第19行中，我们没有使用  ``device`` 和 ``SOS_token`` 
#       全局值，而是使用常量属性 ``self._device`` 和 ``self._SOS_token`` 。
#
#
# -  把 ``torch.jit.script_method`` 装饰器添加到 ``forward`` 方法
#
#    -  添加这个装饰器让JIT编译器知道它正在修饰的函数应该是脚本化的。
#
#
# -  强制转换 ``forward`` 方法参数的类型
#
#    -  默认情况下，Torch Script 函数的所有参数都假定为张量(Tensor)。
#       如果我们需要传递一个不同类型的参数，我们可以使用 
#       `PEP 3107 <https://www.python.org/dev/peps/pep-3107/>`__ 
#       中引入的函数类型注释。
#       此外，可以使用MyPy样式的类型注释声明不同类型的参数
#       (参见 `doc <https://pytorch.org/docs/master/jit.html#types>`__)。
#
#
# -  修改 ``decoder_input`` 的初始化
#
#    -  在最初的实现中，我们使用 ``torch.LongTensor([[SOS_token]])`` 
#       初始化了 ``decoder_input`` 张量。在编写脚本时，
#       不允许我们以这样的文字方式(literal fashion)初始化张量。
#       相反，我们可以用一个显式的torch函数(如  ``torch.ones`` )初始化张量。
#       在这种情况下，我们可以很容易地复制标量 ``decoder_input`` 张量，
#       方法是将 1 乘以存储在常量 ``self._SOS_token`` 中的SOS_token值。
#

class GreedySearchDecoder(torch.jit.ScriptModule):
    def __init__(self, encoder, decoder, decoder_n_layers):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self._device = device
        self._SOS_token = SOS_token
        self._decoder_n_layers = decoder_n_layers

    __constants__ = ['_device', '_SOS_token', '_decoder_n_layers']

    @torch.jit.script_method
    def forward(self, input_seq : torch.Tensor, input_length : torch.Tensor, max_length : int):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:self._decoder_n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=self._device, dtype=torch.long) * self._SOS_token
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=self._device, dtype=torch.long)
        all_scores = torch.zeros([0], device=self._device)
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
# 评估一个输入
# ~~~~~~~~~~~~~~~~~~~
#
# 接下来，我们定义了一些用于评估输入的函数。``evaluate`` 函数接受规范化的字符串语句，
# 将其处理为相应的单词索引(batch size为1)的张量，并将此张量传递给名为 ``searcher`` 
# 的 ``GreedySearchDecoder`` 对象的实例，以处理编码/解码过程。 ``searcher`` 
# 返回 输出单词索引向量 和对应于每个解码单词标记的Softmax分数的分数张量。
# 最后一步是使用 ``voc.index2word`` 将每个单词索引转换回其字符串表示形式。
#
# 我们还定义了评估输入句子的两个函数。``evaluateInput`` 函数提示用户输入，并对其进行计算。
# 它将继续要求另一个输入，直到用户输入‘q’或‘退出’。
#
# ``evaluateExample`` 函数简单地将字符串输入语句作为参数，对其进行规范化、计算并打印响应。
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


# Evaluate inputs from user input (stdin)
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

# Normalize input sentence and call evaluate()
def evaluateExample(sentence, encoder, decoder, searcher, voc):
    print("> " + sentence)
    # Normalize sentence
    input_sentence = normalizeString(sentence)
    # Evaluate sentence
    output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
    output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
    print('Bot:', ' '.join(output_words))


######################################################################
# 加载预先训练的参数
# --------------------------
#
# 好! 是时候加载我们的模型了！！！
#
# 使用别人训练好的模型
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# 要加载别人训练好的模型（hosted model）:
#
# 1) 下载模型在 `这里 <https://download.pytorch.org/models/tutorials/4000_checkpoint.tar>`__.
#
# 2) 设置 ``loadFilename`` 变量，指向下载下来的检查点文件(checkpoint file)的路径
#
# 3) 取消对 ``checkpoint = torch.load(loadFilename)`` 行的注释, 因为 hosted model 是用CPU训练的。
#
# 使用自己的模型
# ~~~~~~~~~~~~~~~~~~
#
# 要加载自己预先训练的模型:
#
# 1) 设置 ``loadFilename`` 变量，指向你自己的检查点文件(checkpoint file)的路径。
#    请注意如果你根据聊天机器人教程的约定规范保存了模型，你就得做一些改动：
#    ``model_name``, ``encoder_n_layers``, ``decoder_n_layers``,
#    ``hidden_size``, 和 ``checkpoint_iter`` 
#    (因为这些值被用到了模型的路径中了)。
#
# 2) 如果你是在CPU上训练的模型, 请确保你打开检查点文件的时候使用 
#    ``checkpoint = torch.load(loadFilename)`` 。
#    如果你是在GPU上训练的模型GPU但是在CPU上运行的此教程, 请取消 
#    ``checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))`` 
#    行的注释。
#
# Hybrid Frontend Notes:
# ~~~~~~~~~~~~~~~~~~~~~~
#
# 注意，我们像往常一样初始化参数并将参数加载到编码器和解码器模型中。
# 此外，在跟踪模型之前，我们必须调用 ``.to(device)`` 来设置模型的设备选项，
# 并调用 ``.eval()`` 来将dropout layers设置为测试模式。
# ``TracedModule`` 对象不继承 ``to`` 或 ``eval`` 方法。
#

save_dir = os.path.join("data", "save")
corpus_name = "cornell movie-dialogs corpus"

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

# If you're loading your own model
# Set checkpoint to load from
checkpoint_iter = 4000
# loadFilename = os.path.join(save_dir, model_name, corpus_name,
#                             '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
#                             '{}_checkpoint.tar'.format(checkpoint_iter))

# If you're loading the hosted model
loadFilename = 'data/4000_checkpoint.tar'

# Load model
# Force CPU device options (to match tensors in this tutorial)
checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
encoder_sd = checkpoint['en']
decoder_sd = checkpoint['de']
encoder_optimizer_sd = checkpoint['en_opt']
decoder_optimizer_sd = checkpoint['de_opt']
embedding_sd = checkpoint['embedding']
voc = Voc(corpus_name)
voc.__dict__ = checkpoint['voc_dict']


print('Building encoder and decoder ...')
# Initialize word embeddings
embedding = nn.Embedding(voc.num_words, hidden_size)
embedding.load_state_dict(embedding_sd)
# Initialize encoder & decoder models
encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
# Load trained model params
encoder.load_state_dict(encoder_sd)
decoder.load_state_dict(decoder_sd)
# Use appropriate device
encoder = encoder.to(device)
decoder = decoder.to(device)
# Set dropout layers to eval mode
encoder.eval()
decoder.eval()
print('Models built and ready to go!')


######################################################################
# 把模型转换为Torch Script
# -----------------------------
#
# 编码器(Encoder)
# ~~~~~~~~~~~~~~~~~
#
# 如前所述，为了将编码器模型转换为 Torch Script，我们使用 **tracing** 。跟踪任何模块都需要
# 通过模型的 ``forward`` 方法运行样例输入，并跟踪数据遇到的计算图。编码器模型接受输入序列
# 和对应的长度张量。因此，我们创建一个样例输入序列张量 ``test_seq`` ，
# 它具有适当的大小(MAX_LENGTH, 1)， 包含适当范围 :math:`[0, voc.num\_words)` 内的数字，
# 并且具有适当的类型(int64)。
# 我们还创建了一个 ``test_seq_length`` 标量，该标量实际上包含了与 ``test_seq`` 中的单词数量相对应的值。
# 下一步是使用 ``torch.jit.trace`` 函数来跟踪模型。请注意，我们传递的第一个参数是要跟踪的模块(module)，
# 第二个参数是模块的 ``forward`` 方法的参数元组。
#
# 解码器(Decoder)
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# 我们对解码器进行跟踪的过程与对编码器的跟踪过程相同。
# 请注意，针对一组随机输入，我们调用 traced_encoder 的 
# ``forward`` 方法，以获得解码器所需的输出。
# 这不是必需的，因为我们也可以简单地制造一个形状、类型和取值范围正确的张量。
# 这种方法是可能的，在我们建立的模型范例中，对张量的值没有任何限制，
# 因为我们没有任何可能对超出范围的输入产生故障的操作。
#
# 贪婪搜索解码器(GreedySearchDecoder)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 回想一下，由于存在依赖于数据的控制流，我们编写了搜索模块的脚本。
# 在脚本化(scripting)的情况下，我们通过添加装饰器(decorator)并确保实现
# 符合脚本化规则来预先完成转换工作。
# 我们初始化脚本搜索程序，就像初始化一个非脚本变量(un-scripted variant)一样。
#

### Convert encoder model
# Create artificial inputs
test_seq = torch.LongTensor(MAX_LENGTH, 1).random_(0, voc.num_words)
test_seq_length = torch.LongTensor([test_seq.size()[0]])
# Trace the model
traced_encoder = torch.jit.trace(encoder, (test_seq, test_seq_length))

### Convert decoder model
# Create and generate artificial inputs
test_encoder_outputs, test_encoder_hidden = traced_encoder(test_seq, test_seq_length)
test_decoder_hidden = test_encoder_hidden[:decoder.n_layers]
test_decoder_input = torch.LongTensor(1, 1).random_(0, voc.num_words)
# Trace the model
traced_decoder = torch.jit.trace(decoder, (test_decoder_input, test_decoder_hidden, test_encoder_outputs))

### Initialize searcher module
scripted_searcher = GreedySearchDecoder(traced_encoder, traced_decoder, decoder.n_layers)


######################################################################
# 输出计算图
# ------------
#
# 现在我们的模型以Torch Script形式出现，我们可以打印每个模型的图表，
# 以确保我们适当地捕获了计算图。由于我们的 ``scripted_searcher`` 包含
# 我们的 ``traced_encoder`` 和 ``traced_decoder`` ，这些图表将会内联打印。
#

print('scripted_searcher graph:\n', scripted_searcher.graph)


######################################################################
# 运行评估
# --------------
#
# 最后，我们将使用Torch Script模型对Chatbot模型进行评估。如果转换正确，
# 则模型的行为将与其急切模式(eager-mode)表示中的行为完全相同。
#
# 默认情况下，我们计算几个常见的查询语句。
# 如果你想自己和机器人聊天，取消注释计算输入行.
#

# Evaluate examples
sentences = ["hello", "what's up?", "who are you?", "where am I?", "where are you from?"]
for s in sentences:
    evaluateExample(s, traced_encoder, traced_decoder, scripted_searcher, voc)

# 取消下面的注释来评估你的输入
#evaluateInput(traced_encoder, traced_decoder, scripted_searcher, voc)


######################################################################
# 保存模型
# ----------
#
# 现在我们已经成功地将模型转换为Torch Script，我们将序列化它，以便在非Python部署环境中使用。
# 要做到这一点，我们可以简单地保存我们的 ``scripted_searcher`` 模块，
# 因为这是针对聊天机器人(chatbot)模型运行推理的面向用户的界面。在保存Script模块时，
# 使用 script_module.save(PATH) 而不是 torch.save(model, PATH)。
#

scripted_searcher.save("scripted_chatbot.pth")
