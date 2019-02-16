# -*- coding: utf-8 -*-
r"""
单词嵌入: 编码词汇语义
===========================================

单词嵌入(Word embeddings)是实数的密集向量，在你的词汇表中每个单词都会有一个整数对应。
在NLP中，特征几乎总是单词(word)！但是，你应该如何在计算机中表示一个单词呢？
您可以存储它的ascii字符表示，但这只告诉您单词是什么，它并不能说明它的含义。
更重要的是，你可以在什么意义上组合这些表示？我们经常希望从我们的神经网络中得到密集的输出，
其中输入是 :math:`|V|` 维的，其中 :math:`V` 是我们的词汇表，但是通常输出只有几维
(例如，如果我们只是预测几个标签的话)。我们如何从一个大的空间到一个较小的空间？

如果我们不使用 ascii representations, 而是使用 one-hot encoding 呢?
就是说, 我们表达单词 :math:`w` ， 通过

.. math::  \overbrace{\left[ 0, 0, \dots, 1, \dots, 0, 0 \right]}^\text{|V| elements}

其中 1 是一个惟一对应于 :math:`w` 的位置。 任何其他的word将会在其他的某个位置有 1 或者 0 。

这个表示法除了非常巨大这个明显的缺点之外还有一个巨大的缺点。它基本上把所有的词当作独立的实体，没有任何关系。
我们真正想要的是词语之间相似(*similarity*)的概念。为什么？让我们看看一个例子。

假设我们正在构建一个语言模型(language model)。假设我们在训练数据中看过这些句子：

* The mathematician ran to the store.
* The physicist ran to the store.
* The mathematician solved the open problem.

现在，假设我们得到了一个新的句子，这在我们的训练数据中是前所未见的：

* The physicist solved the open problem.

我们的语言模型在这句话上可能做得不错，但如果我们可以使用以下两个事实，难道不是更好吗：

* 我们看到数学家和物理学家在句子中扮演着同样的角色。或多或少的，他们有一个语义关系( semantic relation)。
* 我们看到数学家在这个新的没见过的句子中扮演着和我们现在看到的物理学家相同的角色。

然后推断物理学家在这个新的看不见的句子中是个很好的人选？这就是我们所说的相似的概念：
我们指的是语义相似性(*semantic similarity*)，而不仅仅是具有相似的表示法。这是一种与语言数据的稀疏性作斗争的技术，
它将我们所看到的和我们所没有的东西联系起来。这个例子当然依赖于一个基本的语言学假设：
在相似的语境中出现的词在语义上是相互关联的。这就是所谓的分布假说
( `distributional hypothesis <https://en.wikipedia.org/wiki/Distributional_semantics>`__)。


获得密集词嵌入
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

我们如何解决这个问题呢？也就是说，我们怎样才能在词中编码语义相似性呢？也许我们想出了一些语义属性。
例如，我们看到数学家和物理学家都能运行，所以也许我们给这些词一个高分，因为“is able to run”语义属性。
想想其他一些属性，想象一下你可能会在这些属性上得到一些共同的词汇。

如果每个属性都是一个维度，那么我们可以给每个单词一个向量，如下所示：

.. math::

    q_\text{mathematician} = \left[ \overbrace{2.3}^\text{can run},
   \overbrace{9.4}^\text{likes coffee}, \overbrace{-5.5}^\text{majored in Physics}, \dots \right]

.. math::

    q_\text{physicist} = \left[ \overbrace{2.5}^\text{can run},
   \overbrace{9.1}^\text{likes coffee}, \overbrace{6.4}^\text{majored in Physics}, \dots \right]

然后，我们可以通过这样做来衡量这些词之间的相似性：

.. math::  \text{Similarity}(\text{physicist}, \text{mathematician}) = q_\text{physicist} \cdot q_\text{mathematician}

虽然用长度来规范是比较常见的:

.. math::

    \text{Similarity}(\text{physicist}, \text{mathematician}) = \frac{q_\text{physicist} \cdot q_\text{mathematician}}
   {\| q_\text{\physicist} \| \| q_\text{mathematician} \|} = \cos (\phi)

其中 :math:`\phi` 是两个向量之间的角度。这样，非常相似的单词(嵌入指向相同方向的单词)将具有相似性 1。非常不同的词应该有相似度 -1 。

你可以把本节一开始讲到的稀疏的one-hot vectors看作是我们定义的这些新向量的一个特例，其中每个单词基本上都有相似度0，
我们给每个单词一些独特的语义属性。这些新的向量是密集的(*dense*)，也就是说，它们的条目通常是非零的.

但是，这些新的向量是一个很大的痛苦：你可以想出成千上万种不同的语义属性，这些属性可能与确定相似性有关，
你到底会如何设置不同属性的值呢？深度学习的核心思想是，由神经网络学习特征的表示，而不是要求程序员自己设计它们。
那么，为什么不让单词嵌入(word embedding)成为我们模型中的参数，然后在训练期间进行更新呢？这正是我们要做的。
我们将有一些潜在的语义属性(*latent semantic attributes*)，原则上网络可以学习。注意，嵌入(embeddings)这个词可能是不可解释的。
也就是说，尽管上面我们手工制作的向量，我们可以看到数学家和物理学家的相似之处在于他们都喜欢咖啡，
如果我们允许一个神经网络来学习嵌入，并且看到数学家和物理学家在第二维度中都有很大的值，
我们还不清楚这意味着什么。它们在某些潜在的语义维度上是相似的，但这对我们可能没有解释性。

总结一下, **词嵌入(word embeddings)是一个单词的语义的一种表示，
高效的编码了可能与手头的任务相关的语义信息**。 你也可以嵌入其他一些东西：
部分语音标记, 解析树, or 任何东西! 特征嵌入的思想是以具体领域为中心的。


Pytorch 中的词嵌入
~~~~~~~~~~~~~~~~~~~~~~~~~~

在我们开始一个有效的示例和练习之前，先简单介绍一下如何在Pytorch和一般的深度学习编程中使用嵌入(embeddings)。
就像我们在生成one-hot向量时为每个单词定义唯一索引一样，我们也需要在使用嵌入(embeddings)时为每个单词定义一个索引。
这些将是查找表的键(keys into a lookup table)。也就是说，嵌入被存储为一个 :math:`|V| \times D` 矩阵，
其中 :math:`D` 是嵌入的维数，使得具有索引 :math:`i` 的单词的嵌入被存储在矩阵的第 :math:`i` 行中。
在我的所有代码中，从单词到索引的映射是一个名为 word\_to\_ix 的字典。

允许您使用嵌入的module是 torch.nn.Embedding ，它包含两个参数：词汇表的大小和嵌入的维度。

要想在这个表中进行索引, 你必须适应 torch.LongTensor (因为索引是 integers, 而不是 floats).

"""

# Author: Robert Guthrie

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

######################################################################

word_to_ix = {"hello": 0, "world": 1}
embeds = nn.Embedding(2, 5)  # 2 words in vocab, 5 dimensional embeddings
lookup_tensor = torch.tensor([word_to_ix["hello"]], dtype=torch.long)
hello_embed = embeds(lookup_tensor)
print(hello_embed)


######################################################################
# 一个示例: N-Gram 语言模型
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 回想一下，在n-gram语言模型中，给定一单词序列 :math:`w` ，我们想要计算
#
# .. math::  P(w_i | w_{i-1}, w_{i-2}, \dots, w_{i-n+1} )
#
# 其中 :math:`w_i` 是序列的第 i 个单词。
#
# 在这个例子中, 我们将在一些训练样本上计算损失函数，并且使用反向传播(backpropagation)更新参数.
#

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10
# We will use Shakespeare Sonnet 2
test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()
# we should tokenize the input, but we will ignore that for now
# build a list of tuples.  Each tuple is ([ word_i-2, word_i-1 ], target word)
trigrams = [([test_sentence[i], test_sentence[i + 1]], test_sentence[i + 2])
            for i in range(len(test_sentence) - 2)]
# print the first 3, just so you can see what they look like
print(trigrams[:3])

vocab = set(test_sentence)
word_to_ix = {word: i for i, word in enumerate(vocab)}


class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


losses = []
loss_function = nn.NLLLoss()
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(10):
    total_loss = 0
    for context, target in trigrams:

        # Step 1. 准备要传递给模型的输入 (i.e, 把这些单词转化为整数索引并且封装在张量中)
        context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)

        # Step 2. 回想一下torch会 *累积* 梯度。 在传入一个新的样例之前，
        # 应该把之前那个样例产生的梯度清零
        model.zero_grad()

        # Step 3. 运行前向传递过程, 得到下一个单词的对数概率
        log_probs = model(context_idxs)

        # Step 4. 计算损失函数. (进一步, Torch 想要 目标单词封装在一个tensor中)
        loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))

        # Step 5. 执行反向传递并更新梯度
        loss.backward()
        optimizer.step()

        # 通过调用 tensor.item() 得到单元素张量中的Python数字
        total_loss += loss.item()
    losses.append(total_loss)
print(losses)  # 在训练数据上每一次迭代损失都会下降!


######################################################################
# 练习: 计算词嵌入: 连续词袋
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 连续词袋模型(CBOW)是NLP深度学习中常用的一种模式.它是一种模型，
# 它试图预测给定目标单词之前和之后的几个单词的上下文中的目标单词。
# 这与语言建模不同，因为CBOW不是序列化的，也不一定是概率的。通常，CBOW用于快速训练单词嵌入，
# 而这些嵌入用于初始化一些更复杂的模型的嵌入。通常，这被称为预训练嵌入(*pretraining embeddings*)。
# 它几乎总是能帮助性能提升几个百分点。
#
# CBOW 模型的形式化定义如下所示。 给定一个目标单词 :math:`w_i` 和 一个两边长度为 :math:`N` 的上下文窗口，
# :math:`w_{i-1}, \dots, w_{i-N}` 和 :math:`w_{i+1}, \dots, w_{i+N}`, :math:`C` 指向所有上下文单词集体，
# CBOW 试图去最小化
#
# .. math::  -\log p(w_i | C) = -\log \text{Softmax}(A(\sum_{w \in C} q_w) + b)
#
# 其中 :math:`q_w` 是单词 :math:`w` 的嵌入。
#
# 通过完善下面这个类在Pytorch中实现这个模型. 
# 
# 一些小建议:
#
# * 考虑一下你需要定义什么样的参数.
# * 确保你知道每个操作的期望输入和输出的张量的shape。如果你需要reshape的话，使用 .view() 。
#

CONTEXT_SIZE = 2  # 2 words to the left, 2 to the right
raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()

# By deriving a set from `raw_text`, we deduplicate the array
vocab = set(raw_text)
vocab_size = len(vocab)

word_to_ix = {word: i for i, word in enumerate(vocab)}
data = []
for i in range(2, len(raw_text) - 2):
    context = [raw_text[i - 2], raw_text[i - 1],
               raw_text[i + 1], raw_text[i + 2]]
    target = raw_text[i]
    data.append((context, target))
print(data[:5])


class CBOW(nn.Module):

    def __init__(self):
        pass

    def forward(self, inputs):
        pass

# 创建你的模型并训练。这里的一些函数可以帮你准备好你的module所需要的数据 


def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long)


make_context_vector(data[0][0], word_to_ix)  # example
