# -*- coding: utf-8 -*-
"""
`torch.nn` 究竟是神马东东？
============================
**翻译者:** `Antares <http://wwww.studyai.com/antares>`_
"""
###############################################################################
# 我们建议将本教程作为notebook运行，而不是脚本。要下载notebook(.ipynb)文件，
# 请到本文档的最后面 。
#
# PyTorch提供了精心设计的模块和类 `torch.nn <https://pytorch.org/docs/stable/nn.html>`_  、
# `torch.optim <https://pytorch.org/docs/stable/optim.html>`_ 、
# `Dataset <https://pytorch.org/docs/stable/data.html?highlight=dataset#torch.utils.data.Dataset>`_ 
# 和 `DataLoader <https://pytorch.org/docs/stable/data.html?highlight=dataloader#torch.utils.data.DataLoader>`_，
# 以帮助您创建和训练神经网络。为了充分利用他们的力量，并为你的问题定制他们，你需要真正理解他们在做什么。
# 为了加深这种理解，我们将首先在MNIST数据集上训练基本神经网络，而不使用这些模型的任何特性；
# 我们最初只使用最基本的PyTorch张量功能。然后，我们将依次递增地从 ``torch.nn``, ``torch.optim``, ``Dataset`` 或 ``DataLoader``
# 中添加一个特性，准确地显示每个片段的功能，以及它如何使代码更简洁或更灵活
#
# **本教程假设您已经安装了PyTorch，并且熟悉张量操作的基础知识。** 
# (如果您熟悉Numpy数组操作，您会发现这里使用的PyTorch张量操作几乎相同)。
# 
# MNIST 数据设置
# ----------------
#
# 我们将使用经典的 `MNIST <http://deeplearning.net/data/mnist/>`_  数据集，
# 它由手写数字的黑白图像(0到9之间)组成。
#
# 我们将使用 `pathlib <https://docs.python.org/3/library/pathlib.html>`_ 
# 来处理路径(Python 3标准库的一部分)，并使用 `requests <http://docs.python-requests.org/en/master/>`_ 下载数据集。
# 我们只会在使用它们时导入模块，这样您就可以准确地看到在每一点上使用的是什么。

from pathlib import Path
import requests

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "http://deeplearning.net/data/mnist/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).open("wb").write(content)

###############################################################################
# 此数据集为numpy数组格式，并使用 pickle (一种用于序列化数据的python特定格式)存储。

import pickle
import gzip

with gzip.open(PATH / FILENAME, "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

###############################################################################
# 每个图像为28x28，并且存储为长度为784的扁平行(=28x28)。
# 让我们看一看，我们需要先把它重塑到2d。

from matplotlib import pyplot
import numpy as np

pyplot.imshow(x_train[0].reshape((28, 28)), cmap="gray")
print(x_train.shape)

###############################################################################
# PyTorch 使用 ``torch.tensor``, 而不是 numpy arrays, 因此我们需要转换我们的数据。

import torch

x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)
n, c = x_train.shape
x_train, x_train.shape, y_train.min(), y_train.max()
print(x_train, y_train)
print(x_train.shape)
print(y_train.min(), y_train.max())

###############################################################################
# 从零开始搭建网络 (不用 torch.nn)
# ---------------------------------------------
#
# 首先，我们只使用PyTorch张量操作来创建一个模型。我们假设你已经熟悉神经网络的基本知识了。
#
# PyTorch提供了创建随机或零填充张量的方法，我们将使用这种方法为一个简单的线性模型创建权重和偏置。
# 这些只是常规张量，它带有一个非常特殊的附加功能：我们告诉PyTorch，它们需要一个梯度。
# 这使得PyTorch记录了在张量上所做的所有操作，这样它就可以在反向传播过程中自动计算梯度了！
#
# 对于权重，我们在初始化 **之后** 设置 ``requires_grad``，因为我们不希望该步骤包含在梯度中。
# (注意，PyTorch中尾随的 ``_`` 表示操作是原位(in-place)执行的。)
#
# .. note:: 我们使用这篇论文中的方法 来初始化权重：
#    `Xavier initialisation <http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf>`_
#    (by multiplying with 1/sqrt(n)).

import math

weights = torch.randn(784, 10) / math.sqrt(784)
weights.requires_grad_()
bias = torch.zeros(10, requires_grad=True)

###############################################################################
# 由于PyTorch自动计算梯度的能力，我们可以使用任何标准Python函数(或可调用对象)作为模型！
# 所以让我们写一个简单的矩阵乘法和广播加法来创建一个简单的线性模型。我们还需要一个激活函数，
# 所以我们将编写 `log_softmax` 并使用它。记住：虽然PyTorch提供了许多预先编写的损失函数、激活函数等等，
# 但是您可以使用普通python轻松地编写自己的函数。PyTorch甚至会自动为您的函数创建快速GPU或矢量化CPU代码。

def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)

def model(xb):
    return log_softmax(xb @ weights + bias)

###############################################################################
# 在上面，@代表点积操作。我们将在一批数据(本例中为64幅图像)上调用我们的函数。这是一个
# 前向传递(*forward pass*)。
# 注意，我们的预测在这个阶段不会比随机的好，因为我们从随机权重开始。

bs = 64  # batch size

xb = x_train[0:bs]  # a mini-batch from x
preds = model(xb)  # predictions
preds[0], preds.shape
print(preds[0], preds.shape)

###############################################################################
# 正如你所看到的，``preds`` 张量不仅包含张量值，还包含一个梯度函数。
# 我们稍后会用这个来做反向传播(backprop) 。
#
# 让我们实现negative log-likelihood作为损失函数(同样，我们只需要使用标准Python)：

def nll(input, target):
    return -input[range(target.shape[0]), target].mean()

loss_func = nll

###############################################################################
# 让我们用我们的随机模型来检验我们的损失，这样我们就可以看到我们在反向传播后是否有所改善。

yb = y_train[0:bs]
print(loss_func(preds, yb))


###############################################################################
# 让我们还实现一个函数来计算我们的模型的准确率。对于每个预测，如果具有最大值的索引与目标值匹配，则预测是正确的。

def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()

###############################################################################
# 让我们来检查一下我们的随机模型的准确性，这样我们就可以看到我们的准确率是否随着损失的增加而提高。

print(accuracy(preds, yb))

###############################################################################
# 我们现在可以运行一个训练循环。对于每次迭代，我们将：
#
# - 选择数据的 一个 mini-batch (of size ``bs``)
# - 使用模型做预测
# - 计算损失
# - ``loss.backward()`` 更新模型的梯度, 在这个例子中是 ``weights`` 和 ``bias``.
#
# 我们现在使用这些梯度来更新权重和偏置。我们在 ``torch.no_grad()`` 上下文管理器中这样做，
# 因为我们不希望在下一次计算梯度时记录这些操作。您可以在 `这里 <https://pytorch.org/docs/stable/notes/autograd.html>`_.
# 阅读有关PyTorch的Autograd记录操作的更多信息。
#
# 然后，我们将梯度设置为零，以便为下一个循环做好准备。否则，我们的梯度将记录已经发生的
# 所有操作的运行记录(i.e. ``loss.backward()`` 将梯度 *加* 到已经存储的梯度中，而不是替换它们)。
#
# .. tip:: 您可以使用标准python调试器逐步遍历PyTorch代码，允许您在每一步检查各种变量值。
#    在下面取消注释  ``set_trace()`` 来尝试它。
#

from IPython.core.debugger import set_trace

lr = 0.5  # learning rate
epochs = 2  # how many epochs to train for

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        #         set_trace()
        start_i = i * bs
        end_i = start_i + bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        with torch.no_grad():
            weights -= weights.grad * lr
            bias -= bias.grad * lr
            weights.grad.zero_()
            bias.grad.zero_()

###############################################################################
# 就是这样：我们已经完全从零开始创建和训练了一个最小的神经网络(在这种情况下，
# 是一个logistic regression，因为我们没有隐藏的层)！
#
# 让我们检查损失和准确性，并比较我们之前得到的。我们预计损失将有所减少，
# 准确性将有所提高，而且他们已经做到了。

print(loss_func(model(xb), yb), accuracy(model(xb), yb))

###############################################################################
# 使用 torch.nn.functional
# ------------------------------
#
# 现在我们将重构我们的代码，这样它就可以像以前那样做了，只是我们将开始利用PyTorch
# 的 ``nn`` 包中的类来使它更加简洁和灵活。
# 从这里开始的每一步，我们都应该将代码变成一个或多个：更短、更容易理解和/或更灵活。
#
# 第一个也是最简单的步骤是将手写的激活和损失函数替换为 ``torch.nn.function`` 
# (通常按照约定导入命名空间F)中的函数，从而缩短代码。
# 此模块包含 ``torch.nn`` 库中的所有函数(而库的其他部分则包含类)。
# 除了广泛的损失和激活函数之外，您还可以在这里找到一些创建神经网络的方便函数，
# 例如池化函数。(还有一些函数用于进行卷积、线性层等，但正如我们将要看到的那样，
# 通常使用库的其他部分来更好地处理这些函数。)
# 
# 如果您使用的是negative log likelihood loss和log softmax activation，
# 则Pytorch提供了一个将两者结合在一起的函数 ``F.cross_entropy`` 。
# 所以我们甚至可以从我们上面的模型中删除激活函数。

import torch.nn.functional as F

loss_func = F.cross_entropy

def model(xb):
    return xb @ weights + bias

###############################################################################
# 注意，我们不再在 ``model``  函数中调用 ``log_softmax`` 。 
# 现在让我们确认我们的损失和准确性与以前一样：

print(loss_func(model(xb), yb), accuracy(model(xb), yb))

###############################################################################
# 使用 nn.Module 重构
# -----------------------------
# 接下来，我们将使用 ``nn.Module`` 和 ``nn.Parameter`` 来进行更清晰、更简洁的训练循环。
# 我们创建 ``nn.Module`` 的子类(它本身是一个类，能够跟踪状态) 。
# 在这个案例中，我们想要创建一个类来为下一步保存我们的权重、偏置和方法 。
# ``nn.Module`` 有许多属性和方法(例如 ``.parameters()`` 和 ``.zero_grad()`` ，
# 我们将使用这些属性和方法。
#
# .. note:: ``nn.Module`` (大写 M)  是PyTorch特有的概念，是我们将要经常使用的一个类。
#    不可将 ``nn.Module`` 与Python的 `module <https://docs.python.org/3/tutorial/modules.html>`_ 
#    概念(小写m)混淆，后者是可以导入的Python代码文件。

from torch import nn

class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(784, 10) / math.sqrt(784))
        self.bias = nn.Parameter(torch.zeros(10))

    def forward(self, xb):
        return xb @ self.weights + self.bias

###############################################################################
# 由于我们现在使用的是一个对象，而不是仅仅使用一个函数，所以我们首先必须实例化我们的模型：

model = Mnist_Logistic()

###############################################################################
# 现在我们可以和以前一样计算损失了。注意，``nn.Module`` 对象被当作函数来使用(即它们是可调用的)，
# 但是在幕后Pytorch会自动调用我们的 ``forward`` 方法。

print(loss_func(model(xb), yb))

###############################################################################
# 以前，在我们的训练循环中，我们必须按名称更新每个参数的值，并手动将每个参数的梯度分别清零，如下所示：
# ::
#   with torch.no_grad():
#       weights -= weights.grad * lr
#       bias -= bias.grad * lr
#       weights.grad.zero_()
#       bias.grad.zero_()
#
#
# 现在，我们可以利用 model.parameters() 和 model.zero_grad() 
# (它们都是由PyTorch为 ``nn.Module`` 定义的)使这些步骤更简洁，
# 更不容易忘记一些参数，特别是如果我们有一个更复杂的模型:
# ::
#   with torch.no_grad():
#       for p in model.parameters(): p -= p.grad * lr
#       model.zero_grad()
#
#
# 我们将把我们的小训练循环封装在一个 ``fit`` 函数中，这样我们可以在以后再运行它。

def fit():
    for epoch in range(epochs):
        for i in range((n - 1) // bs + 1):
            start_i = i * bs
            end_i = start_i + bs
            xb = x_train[start_i:end_i]
            yb = y_train[start_i:end_i]
            pred = model(xb)
            loss = loss_func(pred, yb)

            loss.backward()
            with torch.no_grad():
                for p in model.parameters():
                    p -= p.grad * lr
                model.zero_grad()

fit()

###############################################################################
# 让我们再检查一下我们的损失是否已经减少了:

print(loss_func(model(xb), yb))

###############################################################################
# 使用 nn.Linear 重构
# -------------------------
#
# 我们继续重构我们的代码。与手动定义和初始化 ``self.weights`` 和 ``self.bias`` 
# 以及计算 ``xb @ self.weights + self.bias`` 不同，
# 我们将使用 Pytorch 的类 `nn.Linear <https://pytorch.org/docs/stable/nn.html#linear-layers>`_ 
# 来构建线性层，将为我们完成所有这些工作。 Pytorch 有许多类型的预定义层，
# 它们可以极大地简化我们的代码，而且通常也会使它更快。

class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(784, 10)

    def forward(self, xb):
        return self.lin(xb)

###############################################################################
# 我们实例化我们的模型并以与以前相同的方式计算损失:

model = Mnist_Logistic()
print(loss_func(model(xb), yb))

###############################################################################
# 我们仍然可以像以前一样使用同样的 ``fit`` 方法。

fit()

print(loss_func(model(xb), yb))

###############################################################################
# 使用 optim 重构
# ------------------------------
#
# Pytorch 还有一个包含各种优化算法的包 ``torch.optim`` 。
# 我们可以使用优化器中的 ``step`` 方法向前迈出一步，而不是手动更新每个参数。
#
# 这将让我们取代以前手工编写的优化步骤:
# ::
#   with torch.no_grad():
#       for p in model.parameters(): p -= p.grad * lr
#       model.zero_grad()
#
# 所以我们的代码变成了这样:
# ::
#   opt.step()
#   opt.zero_grad()
#
# (``optim.zero_grad()`` 将梯度重置为0，我们需要在计算下一个minibatch的梯度之前调用它。)

from torch import optim

###############################################################################
# 我们将定义一个小函数来创建我们的模型和优化器，以便将来可以重用它。

def get_model():
    model = Mnist_Logistic()
    return model, optim.SGD(model.parameters(), lr=lr)

model, opt = get_model()
print(loss_func(model(xb), yb))

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        start_i = i * bs
        end_i = start_i + bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

print(loss_func(model(xb), yb))

###############################################################################
# 使用 Dataset 重构
# ------------------------------
#
# PyTorch 有一个抽象 ``Dataset`` 类。Dataset可以是任何具有 ``__len__`` 函数(由Python的标准len函数调用)
# 和 ``__getitem__`` 函数作为索引的方法的任何数据集。
# `此教程 <https://pytorch.org/tutorials/beginner/data_loading_tutorial.html>`_ 
# 将介绍如何将自定义的 ``FacialLandmarkDataset`` 类创建为 ``Dataset`` 的子类。
#
# PyTorch 的 `TensorDataset <https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#TensorDataset>`_
# 是一个封装了tensors的 Dataset 。 通过定义长度和索引方式，这也为我们提供了沿着张量的第一维迭代、索引和切片的方法。
# 这将使我们在训练过程中更容易访问同一行中的独立变量和因变量。

from torch.utils.data import TensorDataset

###############################################################################
# 包括 ``x_train`` 和 ``y_train`` 都可以被组合到单个的 ``TensorDataset``,
# 这样会使迭代和索引变得更容易。

train_ds = TensorDataset(x_train, y_train)

###############################################################################
# 以前，我们必须分别迭代 x 和y 获得 minibatches:
# ::
#     xb = x_train[start_i:end_i]
#     yb = y_train[start_i:end_i]
#
#
# 现在我们只需要一步就可以搞定:
# ::
#     xb,yb = train_ds[i*bs : i*bs+bs]
#

model, opt = get_model()

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        xb, yb = train_ds[i * bs: i * bs + bs]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

print(loss_func(model(xb), yb))

###############################################################################
# 使用 DataLoader 重构
# ------------------------------
#
# Pytorch 的 ``DataLoader`` 负责管理 batches 。您可以从任何 ``Dataset`` 中创建 ``DataLoader`` 。
# ``DataLoader`` 使遍历批处理变得更容易。 ``DataLoader`` 不需要使用 ``train_ds[i*bs : i*bs+bs]``  ，
# 而是自动给我们每个 minibatch 。

from torch.utils.data import DataLoader

train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs)

###############################################################################
# 以前，我们的循环迭代 batches (xb, yb)，如下所示 :
# ::
#       for i in range((n-1)//bs + 1):
#           xb,yb = train_ds[i*bs : i*bs+bs]
#           pred = model(xb)
#
# 现在，我们的循环要干净得多，因为 (xb, yb) 是从数据加载器自动加载的:
# ::
#       for xb,yb in train_dl:
#           pred = model(xb)

model, opt = get_model()

for epoch in range(epochs):
    for xb, yb in train_dl:
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

print(loss_func(model(xb), yb))

###############################################################################
# 感谢 Pytorch 的 ``nn.Module``, ``nn.Parameter``, ``Dataset``, 和 ``DataLoader``,
# 我们的training loop现在大大缩小，更容易理解。现在让我们尝试在实践中添加创建有效模型所需的基本特性。
#
# 添加验证
# -----------------------
#
# 在第一节中，我们只是试图建立一个合理的训练循环，以便在我们的训练数据上使用。
# 在现实中，您也应该有一个验证集，以确定您是否过拟合了。
#
# 调整训练数据对于防止批次和过度拟合之间的相关性是很重要的
# (`important <https://www.quora.com/Does-the-order-of-training-data-matter-when-training-neural-networks>`_)。
# 另一方面，无论我们是否对验证集洗牌，验证损失都是相同的。
# 由于洗牌需要额外的时间，因此对验证数据进行洗牌是没有意义的。
#
# 对于验证集，我们将使用一个批处理大小，它是训练集的两倍大。这是因为验证集不需要反向传播，
# 因此占用的内存更少(不需要存储梯度)。我们利用这一点使用更大的批次大小，并更快地计算损失。

train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)

valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_ds, batch_size=bs * 2)

###############################################################################
# 我们将在每个回合(epoch)结束时计算和打印验证损失(validation loss)。
#
# (请注意，我们总是在训练前调用  ``model.train()`` ，在推理之前调用 ``model.eval()`` ，
# 因为这些调用被 ``nn.BatchNorm2d`` 和 ``nn.Dropout`` 等层使用，以确保这些不同阶段的适当行为。)

model, opt = get_model()

for epoch in range(epochs):
    model.train()
    for xb, yb in train_dl:
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

    model.eval()
    with torch.no_grad():
        valid_loss = sum(loss_func(model(xb), yb) for xb, yb in valid_dl)

    print(epoch, valid_loss / len(valid_dl))

###############################################################################
# 创建 fit() 和 get_data()
# ----------------------------------
#
# 我们现在要做一些我们自己的重构。由于我们经历了两次计算训练集和验证集的损失的类似过程，
# 所以让我们将其转化为它自己的函数-``loss_batch`` ，它计算一个批次的损失。
#
# 我们为训练集传递一个优化器，并使用它来执行反向传播。对于验证集，不会通过优化器，
# 因此该方法不会执行反向传播。

def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)

###############################################################################
# ``fit`` 运行必要的操作来训练我们的模型，并计算每个epoch的训练和验证损失。

import numpy as np

def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        print(epoch, val_loss)

###############################################################################
# ``get_data`` 返回训练和验证集的数据加载器。


def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )

###############################################################################
# 现在，我们获得数据加载器和拟合模型的整个过程可以在3行代码中运行：

train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
model, opt = get_model()
fit(epochs, model, loss_func, opt, train_dl, valid_dl)

###############################################################################
# 您可以使用这些基本的3行代码来训练各种各样的模型。
# 让我们看看是否可以用它们来训练一个卷积神经网络(CNN)！
#
# 切换到 CNN
# -------------
#
# 我们现在要建立三个卷积层的神经网络。因为上一节中的函数没有任何关于模型形式的假设，
# 所以我们可以使用它们来训练CNN，而不需要任何修改。
#
# 我们将使用Pytorch的预定义的 `Conv2d <https://pytorch.org/docs/stable/nn.html#torch.nn.Conv2d>`_ 
# 类作为我们的卷积层。我们定义了一个包含三个卷积层的CNN。
# 每个卷积后面跟着一个ReLU。最后，我们执行一个平均池化。
# (请注意， ``view`` 是PyTorch版本的Numpy的 ``reshape`` )

class Mnist_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1)

    def forward(self, xb):
        xb = xb.view(-1, 1, 28, 28)
        xb = F.relu(self.conv1(xb))
        xb = F.relu(self.conv2(xb))
        xb = F.relu(self.conv3(xb))
        xb = F.avg_pool2d(xb, 4)
        return xb.view(-1, xb.size(1))

lr = 0.1

###############################################################################
# 动量(`Momentum <http://cs231n.github.io/neural-networks-3/#sgd>`_)是随机梯度下降算法的一种变体，
# 它会考虑到以前的更新，通常导致更快的训练。

model = Mnist_CNN()
opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

fit(epochs, model, loss_func, opt, train_dl, valid_dl)

###############################################################################
# nn.Sequential
# ------------------------
#
# ``torch.nn`` 还有一个方便的类，我们可以使用它来简单地编写代码: `Sequential <https://pytorch.org/docs/stable/nn.html#torch.nn.Sequential>`_ 。
# ``Sequential`` 对象以顺序的方式运行包含在其中的每个modules。这是一种更简单的编写神经网络的方法。
# 为了利用这一点，我们需要能够轻松地从给定的函数中定义 **custom layer** 。例如，PyTorch没有 `view` 层，
# 我们需要为我们的网络创建一个 `view` 层。``Lambda`` 将创建一个层，然后我们在用 ``Sequential`` 定义网络的时候就可以使用它。

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


def preprocess(x):
    return x.view(-1, 1, 28, 28)

###############################################################################
# 使用 ``Sequential`` 创建的模型是很简单滴:

model = nn.Sequential(
    Lambda(preprocess),
    nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.AvgPool2d(4),
    Lambda(lambda x: x.view(x.size(0), -1)),
)

opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

fit(epochs, model, loss_func, opt, train_dl, valid_dl)

###############################################################################
# 封装 DataLoader
# -----------------------------
#
# 我们的CNN相当简洁，但它只适用于MNIST，因为:
#  - 它假设输入是一个 28*28 长的向量。
#  - 它假设最后的CNN网格大小为4*4(因为这是我们使用的平均池化核的size)。
#
# 让我们去掉这两个假设，我们的模型适用于任何2d的单通道图像。首先，我们可以删除初始 Lambda 层，
# 但是将数据预处理移到一个生成器中:

def preprocess(x, y):
    return x.view(-1, 1, 28, 28), y


class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))

train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
train_dl = WrappedDataLoader(train_dl, preprocess)
valid_dl = WrappedDataLoader(valid_dl, preprocess)

###############################################################################
# 接下来，我们可以将 ``nn.AvgPool2d`` 替换为 ``nn.AdaptiveAvgPool2d`` ，
# 这允许我们定义我们想要的 *output* tensor 的大小，
# 而不是我们所拥有的 *input* tensor。因此，我们的模型将适用于任何大小的输入。

model = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    Lambda(lambda x: x.view(x.size(0), -1)),
)

opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

###############################################################################
# 让我们的代码跑起来试试看:

fit(epochs, model, loss_func, opt, train_dl, valid_dl)

###############################################################################
# 使用 GPU
# ---------------
#
# 如果您有一个CUDA功能的GPU，那么您可以使用它来加速您的代码。
# 首先，检查您的GPU是否正在Pytorch中工作：

print(torch.cuda.is_available())

###############################################################################
# 然后为它创建一个设备对象（device object）： 

dev = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

###############################################################################
# 让我们更新 ``preprocess`` 把 batches 移动到 GPU 上:


def preprocess(x, y):
    return x.view(-1, 1, 28, 28).to(dev), y.to(dev)


train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
train_dl = WrappedDataLoader(train_dl, preprocess)
valid_dl = WrappedDataLoader(valid_dl, preprocess)

###############################################################################
# 最后, 我们可以把我们的 model 也移动到 GPU 上。

model.to(dev)
opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

###############################################################################
# 现在你应该能发现它跑得更快了:

fit(epochs, model, loss_func, opt, train_dl, valid_dl)

###############################################################################
# 完结时的思考
# -----------------
#
# 现在我们有了一个通用的 data pipeline 和 training loop，您可以使用它来训练许多类型的模型。
# 要了解模型现在可以进行多简单的训练，请看一下 `mnist_sample` notebook 。
#
# 当然，您需要添加很多东西，例如数据增强、超参数调优、监视训练、迁移学习等等。
#
# 在本教程开始时，我们承诺将通过示例解释 ``torch.nn``, ``torch.optim``, ``Dataset``, 和 ``DataLoader`` 。
# 让我们总结一下我们所看到的：
#
#  - **torch.nn**
#
#    + ``Module``: 创建一个可调用对象，它的行为像一个函数，但也可以包含状态(如神经网络层权重)。
#      它知道它包含什么 ``Parameter`` ，并且可以对它们的所有梯度进行零化，循环遍历它们以进行权重更新，等等。
#    + ``Parameter``: 一个张量的封装器，它告诉 ``Module`` 它有需要在反向传播过程中更新的权重。
#      只更新具有 `requires_grad` 属性的张量。
#    + ``functional``: 一个module(通常按约定导入到 ``F`` 命名空间)，它包含激活函数、损失函数等，以及层的无状态版本，如卷积层和线性层。
#  - ``torch.optim``: 包含诸如 ``SGD`` 之类的优化器，这些优化器在反向传递期间更新参数的权重
#  - ``Dataset``: 具有 ``__len__`` 和 ``__getitem__`` 的对象的抽象接口，包括Pytorch提供的类，如 ``TensorDataset``
#  - ``DataLoader``: 获取任意 ``Dataset`` 并创建返回批量数据的迭代器。 
