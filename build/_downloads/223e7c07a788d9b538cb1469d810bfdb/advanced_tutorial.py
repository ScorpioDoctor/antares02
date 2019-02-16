# -*- coding: utf-8 -*-
r"""
高级专题: 做动态决策 和 Bi-LSTM CRF
======================================================

动态Vs静态深度学习工具包
--------------------------------------------

Pytorch 是一种 *动态* 神经网络工具箱。动态工具包的另一个例子是 `Dynet <https://github.com/clab/dynet>`__ 
(我提到这一点，因为Pytorch和Dynet是相似的。 如果您在Dynet中看到一个示例，它可能会帮助您在Python中实现它)。
相反的是 *静态* 工具包，包括Theano、Keras、TensorFlow等。其核心区别是:

* 在静态工具箱中，您只定义一次计算图，编译它，然后将实例流入它。
* 在动态工具箱中，为 *每个实例* 定义一个计算图。它从来不被编译，并且是实时执行的。

如果没有丰富的经验，就很难理解其中的不同之处。一个例子是假设我们想要构建一个深层成分解析器(deep constituent parser)。
假设我们的模型大致包括以下步骤：

* 我们自底向上构建树
* 标记根节点 (句子的单子)
* 从那里，使用神经网络和单词的嵌入，以找到形成成分的组合。
  每当您形成一个新的成分时，都要使用某种技术来获得该成分的嵌入。
  在这种情况下，我们的网络体系结构将完全依赖于输入语句。在“The green cat scratched the wall”这个句子中，在模型的某一点上，
  我们希望组合跨度(span) :math:`(i,j,r) = (1, 3, \text{NP})` (即NP成分跨越单词1到单词3，在本例中是"The green cat")。

然而，另一个句子可能是"Somewhere, the big fat cat scratched the wall" 。在这个句子中，我们要在某一点上形成成分 :math:`(2, 4, NP)` 。
我们想要形成的成分将取决于实例。如果我们只编译一次计算图，就像在静态工具箱中一样，那么编程这个深层成分解析器的逻辑将是非常困难或不可能的。
然而，在动态工具箱中，并不只有一个预定义的计算图。每个实例都可以有一个新的计算图，所以这根本就不是个问题。

动态工具包还具有更易于调试的优点，而且代码更类似于宿主语言(host language)(我的意思是，与Keras或Theano相比，
Pytorch和Dynet看起来更像真正的Python代码)。

Bi-LSTM 条件随机场(CRFs)的讨论
-------------------------------------------

对于本节，我们将看到一个完整的，复杂的例子：用于命名实体(named-entity)识别的Bi-LSTM条件随机场。上面的LSTM标记器对于词性标注来说通常是足够的，
但是像CRF这样的序列模型对于在NER上的强大性能来说是非常重要的。假设你熟悉条件随机场(CRF)。虽然这个名字听起来很吓人，所有的模型都是CRF，
只是LSTM为这些CRF模型提供了特征。不过，这仍然是一个高级模型，比本教程中的任何早期模型都要复杂得多。
如果你想跳过它，那也很好。看你是否准备好了，看看你能不能:

-  Write the recurrence for the viterbi variable at step i for tag k.
-  Modify the above recurrence to compute the forward variables instead.
-  Modify again the above recurrence to compute the forward variables in
   log-space (hint: log-sum-exp)

如果你能做这三件事，你应该能够理解下面的代码。回想一下CRF咋样计算条件概率的。让 :math:`y` 是标记序列，:math:`x` 是单词的输入序列。然后我们计算

.. math::  P(y|x) = \frac{\exp{(\text{Score}(x, y)})}{\sum_{y'} \exp{(\text{Score}(x, y')})}

其中，分数是通过定义一些 log potentials :math:`\log \psi_i(x,y)` 来确定的，这样就有下式：

.. math::  \text{Score}(x,y) = \sum_i \log \psi_i(x,y)

要使分区函数(partition function)易于处理，the potentials 必须只考虑局部特征。

在Bi-LSTM CRF中，我们定义了两种势(potentials)：发射势(emission)和跃迁势(transition)。
在索引 :math:`i` 处单词的emission来自Bi-LSTM在第 :math:`i` 步的隐藏状态。
transition scores 存储在一个 :math:`|T|x|T|` 矩阵 :math:`\textbf{P}` 中，其中 :math:`T` 是标记集(tag set)。
在我的实现中，:math:`\textbf{P}_{j,k}` 是从标记 :math:`k` 跃迁到标记 :math:`j` 的得分。因此:

.. math::  \text{Score}(x,y) = \sum_i \log \psi_\text{EMIT}(y_i \rightarrow x_i) + \log \psi_\text{TRANS}(y_{i-1} \rightarrow y_i)

.. math::  = \sum_i h_i[y_i] + \textbf{P}_{y_i, y_{i-1}}

其中在第二个表达式中, 我们认为标签被分配了唯一的非负索引。

如果你觉得上面的讨论过于简单, 你可以查看 `这个 <http://www.cs.columbia.edu/%7Emcollins/crf.pdf>`__ 
来自 Michael Collins 所写的关于 CRFs 的内容。

实现笔记
--------------------

下面的例子实现了对数空间中的前向算法来计算分区函数(partition function)，并实现了Viterbi算法来解码。
反向传播将自动为我们计算梯度。我们不需要手工做任何事。

这里的实现并没有进行优化。如果您了解正在发生的事情，您可能很快就会看到，
前向算法中的下一个标记的迭代其实更适合在一个大操作中完成的。
我想要代码更易读，所以并没有将其放在大操作中。
如果您想要进行相关的更改，可以将这个标记器(tagger)用于实际任务。
"""
# Author: Robert Guthrie

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)

#####################################################################
# 辅助函数使代码更具可读性。


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

#####################################################################
# 创建模型


class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

#####################################################################
# 运行训练


START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 5
HIDDEN_DIM = 4

# Make up some training data
training_data = [(
    "the wall street journal reported today that apple corporation made money".split(),
    "B I I I O O O B I O O".split()
), (
    "georgia tech is a university in georgia".split(),
    "B I O O O O B".split()
)]

word_to_ix = {}
for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}

model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

# Check predictions before training
with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
    print(model(precheck_sent))

# Make sure prepare_sequence from earlier in the LSTM section is loaded
for epoch in range(
        300):  # again, normally you would NOT do 300 epochs, it is toy data
    for sentence, tags in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is,
        # turn them into Tensors of word indices.
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)

        # Step 3. Run our forward pass.
        loss = model.neg_log_likelihood(sentence_in, targets)

        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss.backward()
        optimizer.step()

# Check predictions after training
with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    print(model(precheck_sent))
# We got it!


######################################################################
# 练习: 一个用于判别标记的新的损失函数
# --------------------------------------------------------
#
# 在解码时，我们没有必要创建一个计算图，因为我们不会从viterbi路径分数反向传播。
# 既然我们有它，试着训练标记器，其中损失函数是viterbi路径分数和gold-standard路径分数之间的差异。
# 应该清楚的是，这个函数是非负的，当预测的标记序列是正确的标记序列时是0。
# 这本质上是结构化感知器(*structured perceptron*)。
#
# 这个修改应该是短的，因为Viterbi和 score\_sentence 已经实现了。这是计算图的形状取决于训练实例的一个例子。
# 虽然我还没有尝试在静态工具箱中实现这一点，我认为这是可能的，但却要复杂得多。
#
# 选择一些真实数据并作比较!
#
