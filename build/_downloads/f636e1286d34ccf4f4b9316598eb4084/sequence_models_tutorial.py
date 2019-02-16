# -*- coding: utf-8 -*-
r"""
序列模型和长短时记忆网络(LSTM)
===================================================

到目前为止，我们已经看到了各种各样的前馈网络(feed-forward networks)。
也就是说，根本不存在由网络维护的状态(state)。
这可能不是我们想要的行为。序列模型(Sequence models)是NLP的核心：
它们是在输入之间通过时间存在某种依赖关系的模型。
序列模型的经典例子是用于词性标注(part-of-speech tagging)的
隐马尔可夫模型(Hidden Markov Model)。
另一个例子是条件随机场(conditional random field)。

递归神经网络(recurrent neural network)是一种保持某种状态的网络。
例如，它的输出可以作为下一个输入的一部分使用，以便信息可以在序列通过网络时在序列中传播。
在LSTM的情况下，对于序列中的每个元素，都有相应的隐藏状态(:math:`h_t`)，
原则上可以包含来自序列中较早的任意点的信息。
我们可以利用隐藏状态来预测语言模型中的单词、词性标注(part-of-speech tags)以及无数其他事物。


Pytorch中的LSTM
~~~~~~~~~~~~~~~~~

在开始这个示例之前，请注意以下几点。Pytorch的LSTM期望它的所有输入都是3D张量。
这些张量的每个轴(axes)的语义很重要。第一个轴是序列本身，第二个轴索引batch中的样例，
以及第三个轴索引输入的元素。我们还没有讨论过mini-batching，所以让我们忽略这一点，
并假设我们总是只有1维在第二轴。如果我们想在"The cow jumped"这个句子上运行序列模型，
我们的输入应该看起来像这样：

.. math::


   \begin{bmatrix}
   \overbrace{q_\text{The}}^\text{row vector} \\
   q_\text{cow} \\
   q_\text{jumped}
   \end{bmatrix}

不过，请记住，还有一个额外的第二维,其size为1。

此外，您可以一次遍历一遍序列，在这种情况下，第一轴的size也是1。

让我们看一个快速的例子。
"""

# Author: Robert Guthrie

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

######################################################################

lstm = nn.LSTM(3, 3)  # Input dim is 3, output dim is 3
inputs = [torch.randn(1, 3) for _ in range(5)]  # make a sequence of length 5

# initialize the hidden state.
hidden = (torch.randn(1, 1, 3),
          torch.randn(1, 1, 3))
for i in inputs:
    # Step through the sequence one element at a time.
    # after each step, hidden contains the hidden state.
    out, hidden = lstm(i.view(1, 1, -1), hidden)

# alternatively, we can do the entire sequence all at once.
# the first value returned by LSTM is all of the hidden states throughout
# the sequence. the second is just the most recent hidden state
# (compare the last slice of "out" with "hidden" below, they are the same)
# The reason for this is that:
# "out" will give you access to all hidden states in the sequence
# "hidden" will allow you to continue the sequence and backpropagate,
# by passing it as an argument  to the lstm at a later time
# Add the extra 2nd dimension
inputs = torch.cat(inputs).view(len(inputs), 1, -1)
hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))  # clean out hidden state
out, hidden = lstm(inputs, hidden)
print(out)
print(hidden)


######################################################################
# 样例: 一种用于词性标注的LSTM
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 在这个小节中, 我们将使用 LSTM 来获得词性标注(part of speech tags)。
# 我们将不会使用Viterbi 或 Forward-Backward 或 其他任何类似的技术,
# 但是作为一个对读者稍微有挑战性的练习, 当你了解了这一切如何运转的时候
# 再考虑一下如何使用 Viterbi。 
#
# 模型如下: 假定我们的输入语句是 :math:`w_1, \dots, w_M`, 其中 :math:`w_i \in V`, 我们的词汇库。
# 另外, 假定 :math:`T` 是我们的标记集合, 以及 :math:`y_i` 是单词 :math:`w_i` 的标记。
# 把我们对单词 :math:`w_i` 的标记的预测记为 :math:`\hat{y}_i` 。
#
# 这是一个结构预测，模型，其中我们的输出是序列 :math:`\hat{y}_1, \dots, \hat{y}_M`,
# 其中 :math:`\hat{y}_i \in T` 。
# 
# 为了进行预测, 在句子上传递一个LSTM(pass an LSTM over the sentence)。 
# 在时间步(timestep) :math:`i` 的隐藏状态记为 :math:`h_i` 。
# 另外，给每个tag分配一个唯一的index (就像在词嵌入章节中的 word\_to\_ix 一样)。
# 然后，我们预测 :math:`\hat{y}_i` 的规则是：
#
# .. math::  \hat{y}_i = \text{argmax}_j \  (\log \text{Softmax}(Ah_i + b))_j
#
# 也就是说, 对 隐藏状态的仿射映射 取 对数软最大化(log softmax),
# 并且预测出的tag是这个向量中的最大值对应的tag。
# 请注意，这立即意味着 :math:`A` 的目标空间的维数为 :math:`|T|` 。
#
#
# 准备数据:

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]
word_to_ix = {}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
print(word_to_ix)
tag_to_ix = {"DET": 0, "NN": 1, "V": 2}

# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
EMBEDDING_DIM = 6
HIDDEN_DIM = 6

######################################################################
# 创建模型:


class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

######################################################################
# 训练模型:


model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
# Here we don't need to train, so the code is wrapped in torch.no_grad()
with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)
    print(tag_scores)

for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
    for sentence, tags in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Also, we need to clear out the hidden state of the LSTM,
        # detaching it from its history on the last instance.
        model.hidden = model.init_hidden()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)

        # Step 3. Run our forward pass.
        tag_scores = model(sentence_in)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()

# See what the scores are after training
with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)

    # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
    # for word i. The predicted tag is the maximum scoring tag.
    # Here, we can see the predicted sequence below is 0 1 2 0 1
    # since 0 is index of the maximum value of row 1,
    # 1 is the index of maximum value of row 2, etc.
    # Which is DET NOUN VERB DET NOUN, the correct sequence!
    print(tag_scores)


######################################################################
# 练习: 使用字符级特征增强LSTM语义标注
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 在上面的例子中，每个单词都有一个嵌入，作为序列模型的输入。
# 让我们用 从单词的字符派生出来的表示 来增强 单词嵌入。
# 我们希望这会有很大的帮助，因为词缀(affixes)之类的字符级信息对词性(part-of-speech)有很大的影响。
# 例如，带有词缀 *-ly* 的词在英语中几乎总是被标记为副词(adverbs)。
#
# 为了做到这一点, 令 :math:`c_w` 是单词 :math:`w` 的字符级表示(character-level representation)。
# 像之前一样，令 :math:`x_w` 是单词嵌入。 然后，我们序列模型的输入是 :math:`x_w` 和 :math:`c_w`
# 的串接(concatenation)。因此，如果 :math:`x_w` 有 5 个维度, 并且 :math:`c_w` 的纬度是 3 ,
# 那么我们的 LSTM 应该接受维数为8的输入。
#
# 为了获得字符级表示, 在一个单词的若干字符上做LSTM ,并且令 :math:`c_w` 是这个LSTM的最终隐藏状态。
# 
# 提示:
#
# * 你的新模型将会有两个LSTM。原来的LSTM输出POS标签分数(POS tag scores),
#   新的LSTM输出每个单词的字符级表示。
# * 为了在字符集上建一个序列模型, 你必须嵌入字符(embed characters)。
#   字符嵌入将会成为字符级LSTM的输入。
#
