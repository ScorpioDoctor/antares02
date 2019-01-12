# -*- coding: utf-8 -*-
r"""
使用 PyTorch 进行 深度学习
**************************
翻译者： http://www.studyai.com/antares

深度学习构建块: 仿射映射, 非线性单元 和 目标函数
==========================================================================

深度学习包括以聪明的方式组合线性和非线性。非线性的引入使模型变得很强大。
在本节中，我们将使用这些核心组件，构造一个目标函数，并查看模型是如何训练的。


仿射映射
~~~~~~~~~~~

深度学习的核心工作之一是仿射映射(affine map)，它是一个函数 :math:`f(x)` ，其中

.. math::  f(x) = Ax + b

对矩阵 :math:`A` 和向量 :math:`x, b`. 这里我们要学习的参数就是 :math:`A` 和 :math:`b`. 
通常, :math:`b` 被称之为 偏置项(the *bias* term)。


PyTorch和大多数其他深度学习框架所做的事情与传统的线性代数略有不同。它映射输入的行而不是列。
也就是说，下面第 :math:`i` 行的输出是输入的第 :math:`i` 行的映射，再加上偏置项。看下面的例子。

"""

# Author: Robert Guthrie

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)


######################################################################

lin = nn.Linear(5, 3)  # maps from R^5 to R^3, parameters A, b
# data is 2x5.  A maps from 5 to 3... can we map "data" under A?
data = torch.randn(2, 5)
print(lin(data))  # yes


######################################################################
# 非线性单元
# ~~~~~~~~~~~~~~~
#
# 首先, 注意以下事实, 它告诉我们为什么需要非线性单元。假定 我们有两个仿射映射：
# :math:`f(x) = Ax + b` 和 :math:`g(x) = Cx + d` 。 那么 :math:`f(g(x))` 是什么呢？
#
# .. math::  f(g(x)) = A(Cx + d) + b = ACx + (Ad + b)
#
# :math:`AC` 是一个矩阵 而 :math:`Ad + b` 是一个向量, 因此我们看到组合仿射映射之后的结果还是仿射映射。
#
# 从这里，你可以看到，如果你想要你的神经网络是很多仿射变换构成的长链条，这并没有为你的模型增加新的能力，
# 你的模型，最终只是做单个仿射映射而已。
#
# 如果我们在仿射层之间引入非线性，情况就不再是这样了，我们可以建立更强大的模型。
#
# 已经有一些核心的非线性单元: :math:`\tanh(x), \sigma(x), \text{ReLU}(x)` 是最常见的。
# 你或许会质疑: "为什么非得是这些非线性函数? 我可以想到的其他形式的非线性函数多如牛毛"。
# 这主要是因为它们的梯度非常容易计算，而且梯度计算在学习中有着至关重要的作用。
# 比如说：
#
# .. math::  \frac{d\sigma}{dx} = \sigma(x)(1 - \sigma(x))
#
# 注意：虽然你在AI类的介绍中学到了一些神经网络，其中 :math:`\sigma(x)` 是默认的非线性单元，
# 但在实践中人们通常回避它。这是因为梯度随着参数绝对值的增长而迅速消失。
# 小梯度意味着很难学习。大多数人默认为tanh或relu。
#

# 在 pytorch 中, 大多数非线性单元是在 torch.nn.functional (我们将它导入为F)
# 注意，非线性单元通常没有仿射映射那样的可学习参数。也就是说，他们没有在训练中更新的权重。
data = torch.randn(2, 2)
print(data)
print(F.relu(data))


######################################################################
# 软最大化 和 概率分布
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 函数 :math:`\text{Softmax}(x)` 也是一个非线性函数，但它的特殊之处在于它通常是网络中的最后一次运算。
# 这是因为它接受一个实数向量，并返回一个概率分布。
# 其定义如下。设 :math:`x` 是实数向量(正的，负的，随便什么，没有约束)。
# 那么 :math:`\text{Softmax}(x)` 的第 i 个分量是:
#
# .. math::  \frac{\exp(x_i)}{\sum_j \exp(x_j)}
#
# 应该清楚的是，输出是一个概率分布：每个元素都是非负的，所有分量之和等于 1。
#
# 你也可以认为它只是对输入向量进行按元素的指数运算，使一切非负，然后除以归一化常数。
#

# Softmax 也在 torch.nn.functional 中
data = torch.randn(5)
print(data)
print(F.softmax(data, dim=0))
print(F.softmax(data, dim=0).sum())  # Sums to 1 because it is a distribution!
print(F.log_softmax(data, dim=0))  # theres also log_softmax


######################################################################
# 目标函数
# ~~~~~~~~~~~~~~~~~~~
#
# 目标函数是您的网络正在被训练以最小化的函数(在这种情况下，它通常被称为损失函数 
# *loss function* 或代价(成本)函数 *cost function*)。
# 首先选择一个训练实例，通过你的神经网络运行它，然后计算输出的损失。
# 然后利用损失函数的导数对模型的参数进行更新。直觉地说，如果你的模型对它的答案完全有信心，
# 而且它的答案是错误的，那么你的损失就会很大。如果它对它的答案很有信心，
# 而且它的答案是正确的，那么损失就会很小。
#
# 在你的训练样例上最小化损失函数的思想是，希望您的网络能够很好地泛化(generalize)，
# 并在开发集、测试集或生产中的未见过的样例中损失较小。
# 一个典型的损失函数是负对数似然损失(*negative log likelihood loss*)，
# 这是多类分类的一个非常常见的目标。对于有监督的多类分类，
# 这意味着训练网络最小化正确输出的负对数概率(或等效地，最大化正确输出的对数概率)。
#


######################################################################
# 优化和训练
# =========================
#
# 那么，我们能为一个实例计算一个损失函数吗？那我们该怎么办？我们早些时候看到张量知道如何计算该张量
# 相对于那些计算出它的变量的梯度。既然我们的损失是张量，我们就可以计算所有用于计算它的参数的梯度！
# 然后我们可以执行标准梯度更新。设 :math:`\theta` 为我们的参数，:math:`L(\theta)`  为损失函数，
# :math:`\eta` 为正的学习速率.然后:
#
# .. math::  \theta^{(t+1)} = \theta^{(t)} - \eta \nabla_\theta L(\theta)
#
# 有大量的算法和积极的研究试图做的不仅仅是这个普通的梯度更新。许多人试图根据训练时段发生的情况来改变学习速度。
# 除非您真正感兴趣，否则您不需要担心这些算法具体做了什么。Torch在 Torch.optim 包中提供了许多学习速率调节策略，而且它们都是完全透明的。
# 使用最简单的梯度更新和更复杂的算法是一样的。尝试不同的更新算法和更新算法的不同参数
# (比如不同的初始学习速率)对于优化网络的性能非常重要。
# 通常，用Adam或RMSProp这样的优化器代替普通的SGD会显着地提高性能。
#


######################################################################
# 在 PyTorch 中创建神经网络组件
# ======================================
#
# 在我们继续关注NLP之前，让我们做一个带注释的例子，在PyTorch中使用仿射映射和非线性来构建网络。
# 我们还将看到如何计算损失函数，使用PyTorch的负对数似然，并通过反向传播更新参数。
#
# 所有网络组件都应该继承nn.Module并覆盖forward()方法。嗯，没错，就是这样一个套路。
# 从nn.Module继承可以为组件提供很多功能。例如，它可以跟踪它的可训练参数，
# 您可以使用 ``.to(device)`` 方法在CPU和GPU之间交换它，
# 其中设备可以是CPU设备 ``torch.device("cpu")`` 或CUDA设备 ``torch.device("cuda:0")`` 。
#
# 让我们编写一个带标注的示例网络，它接受稀疏的词袋表示(sparse bag-of-words representation)，
# 并在两个标签上输出概率分布：“英语”和“西班牙语”。这个模型只是个Logistic回归模型。
#


######################################################################
# 案例: Logistic回归词袋分类器
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 我们的模型将把稀疏词袋(BoW)表示映射到标签上的概率。我们为词汇库(vocab)的单词分配一个索引(index)。
# 例如，假设我们的整个词汇是两个单词“hello”和“world”，索引分别为0和1。
# "hello hello hello hello" 这个句子的BoW向量是
#
# .. math::  \left[ 4, 0 \right]
#
# 对于句子 "hello world world hello" 的 BoW 向量是
#
# .. math::  \left[ 2, 2 \right]
#
# etc. 通用意义上说, 它是
#
# .. math::  \left[ \text{Count}(\text{hello}), \text{Count}(\text{world}) \right]
#
# 把这个BoW向量记为 :math:`x` 。 那么我们网络的输出是:
#
# .. math::  \log \text{Softmax}(Ax + b)
#
# 那就是说, 我们传递输入使其通过一个仿射映射，然后做对数软最大化(log softmax).
#

data = [("me gusta comer en la cafeteria".split(), "SPANISH"),
        ("Give it to me".split(), "ENGLISH"),
        ("No creo que sea una buena idea".split(), "SPANISH"),
        ("No it is not a good idea to get lost at sea".split(), "ENGLISH")]

test_data = [("Yo creo que si".split(), "SPANISH"),
             ("it is lost on me".split(), "ENGLISH")]

# word_to_ix maps each word in the vocab to a unique integer, which will be its
# index into the Bag of words vector
word_to_ix = {}
for sent, _ in data + test_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
print(word_to_ix)

VOCAB_SIZE = len(word_to_ix)
NUM_LABELS = 2


class BoWClassifier(nn.Module):  # inheriting from nn.Module!

    def __init__(self, num_labels, vocab_size):
        # calls the init function of nn.Module.  Dont get confused by syntax,
        # just always do it in an nn.Module
        super(BoWClassifier, self).__init__()

        # Define the parameters that you will need.  In this case, we need A and b,
        # the parameters of the affine mapping.
        # Torch defines nn.Linear(), which provides the affine map.
        # Make sure you understand why the input dimension is vocab_size
        # and the output is num_labels!
        self.linear = nn.Linear(vocab_size, num_labels)

        # NOTE! The non-linearity log softmax does not have parameters! So we don't need
        # to worry about that here

    def forward(self, bow_vec):
        # Pass the input through the linear layer,
        # then pass that through log_softmax.
        # Many non-linearities and other functions are in torch.nn.functional
        return F.log_softmax(self.linear(bow_vec), dim=1)


def make_bow_vector(sentence, word_to_ix):
    vec = torch.zeros(len(word_to_ix))
    for word in sentence:
        vec[word_to_ix[word]] += 1
    return vec.view(1, -1)


def make_target(label, label_to_ix):
    return torch.LongTensor([label_to_ix[label]])


model = BoWClassifier(NUM_LABELS, VOCAB_SIZE)

# 模型知道它的参数.  下面的第一个输出是A, 第二个输出是 b。
# 不论何时，你给一个module的 __init__() 函数中的类变量分配一个组件
# 就像这一句做的这样：self.linear = nn.Linear(...)
# 然后通过一些 Python的魔法函数，你的module(本例中的 BoWClassifier )将会存储关于 nn.Linear 的参数的知识
for param in model.parameters():
    print(param)

# 要运行这个模型, 传递一个 BoW vector
# 这里我们先不考虑训练, 因此代码被封装在 torch.no_grad() 中：
with torch.no_grad():
    sample = data[0]
    bow_vector = make_bow_vector(sample[0], word_to_ix)
    log_probs = model(bow_vector)
    print(log_probs)


######################################################################
# 以上哪个值对应于英语的对数概率(log probability)，哪个值对应于西班牙语？我们从来没有定义过它，
# 但是如果我们想训练它的话，我们就需要这样做。
#

label_to_ix = {"SPANISH": 0, "ENGLISH": 1}


######################################################################
# 所以让我们训练吧！为此，我们把样本实例传给网络获取输出的对数概率，计算损失函数的梯度，然后用梯度更新一步参数。
# 损失函数由PyTorch中的nn包提供。nn.nLLoss() 是我们想要的负对数似然损失(negative log likelihood loss)。
# PyTorch中的torch.optim包中定义了优化函数。在这里，我们将只使用SGD。
#
# 注意，NLLoss的 *input*  是对数概率的向量还有目标标签向量。它不计算我们的对数概率(log probabilities)。
# 这就是为什么我们网络的最后一层是 log softmax。损失函数nn.CrossEntroyLoss()与NLLoss()相同，
# 只是它为您做了log softmax。
#

# 在我们训练之前在测试数据上运行一下模型, 这么做是想把训练前后模型在测试数据上的输出做个对比
with torch.no_grad():
    for instance, label in test_data:
        bow_vec = make_bow_vector(instance, word_to_ix)
        log_probs = model(bow_vec)
        print(log_probs)

# Print the matrix column corresponding to "creo"
print(next(model.parameters())[:, word_to_ix["creo"]])

loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 通常，您要传递好几次训练数据。100次epoch比在实际数据集上的epoch大得多，
# 但实际数据集有两个以上的实例。通常，在5至30个epoch之间是合理的。
for epoch in range(100):
    for instance, label in data:
        # Step 1. 请记住，PyTorch累积了梯度。我们需要在每个样例(或batch)之前清除它们
        model.zero_grad()

        # Step 2. 产生我们的BOW向量，同时，我们必须将目标装在一个整数张量中。
        # 例如，如果目标是西班牙语，那么我们包装整数0。然后，
        # 损失函数知道对数概率向量的第0元素是与西班牙语对应的对数概率。
        bow_vec = make_bow_vector(instance, word_to_ix)
        target = make_target(label, label_to_ix)

        # Step 3. 运行我们的前向传递过程.
        log_probs = model(bow_vec)

        # Step 4. 计算 loss, gradients, 并通过调用 optimizer.step() 更新参数
        loss = loss_function(log_probs, target)
        loss.backward()
        optimizer.step()

with torch.no_grad():
    for instance, label in test_data:
        bow_vec = make_bow_vector(instance, word_to_ix)
        log_probs = model(bow_vec)
        print(log_probs)

# Index corresponding to Spanish goes up, English goes down!
print(next(model.parameters())[:, word_to_ix["creo"]])


######################################################################
# 我们得到了正确的答案！您可以看到，在第一个样例中西班牙语的对数概率要高得多，
# 而在第二个测试数据中英语的对数概率要高得多，这是应该的。
#
# 现在，您将看到如何制作PyTorch组件，通过它传递一些数据并进行梯度更新。
# 我们准备深入挖掘深度NLP所能提供的内容。
#
