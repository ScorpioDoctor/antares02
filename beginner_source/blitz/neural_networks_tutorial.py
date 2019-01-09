# -*- coding: utf-8 -*-
"""
神经网络
===============
**翻译者**: `Antares博士 <http://www.studyai.com/antares>`_

神经网络(Neural networks)可以使用 ``torch.nn`` package 构建。

现在你已经了解了自动梯度 ``autograd``, ``nn`` 依赖于 
``autograd`` 来定义模型并对它们求微分。
一个 ``nn.Module`` 包含若干 layers, 和一个方法 ``forward(input)``,
该方法返回 ``output`` 。

举个栗子, 请看下面这个用来分类手写数字的网络:

.. figure:: /_static/img/mnist.png
   :alt: convnet

   convnet

它是一个简单的前馈网络(feed-forward network)。 它接受输入(input)，并把它们一个层接着一个层的往前传递，
最后给出 输出(output)。

一个神经网络的典型训练步骤包括下面这几步:

- 定义具有可学习参数(weights)的神经网络
- 在输入的一个数据集上进行迭代
- 沿着网络处理输入
- 计算损失 (度量 网络的输出 离 我们期望的正确输出 还有多远)
- 把梯度反向传播到网络的参数
- 更新网络权重, 典型的是使用一个简单的更新规则:  ``weight = weight - learning_rate * gradient``

定义网络
------------------

让我们定义一个网络吧:
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 输入图像 1 个通道, 6 个输出通道, 5x5 方形卷积核
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 一个线性映射: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 最大池化的 窗口大小(2, 2)
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # 如果池化窗口是方形的，你只需要指定单个数字
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # 除了 batch 维 的所有纬度
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)

########################################################################
# 你仅仅需要定义一个 ``forward`` 函数, 并且 ``backward`` 函数
# (在其中梯度被计算出来) 是使用 ``autograd`` 为你自动定义的。
# 你可以在 ``forward`` 函数中使用任意的 Tensor 运算/操作。
#
# 模型的可学习参数通过 ``net.parameters()`` 来获取。

params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight

########################################################################
# 让我们尝试一个 32x32 的随机输入
# 注意: 这个网络(LeNet)期望的输入尺寸是 32x32。 为了把这个网络用于
# MNIST 数据集, 请将数据集中的图像尺寸缩放到 32x32.

input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)


########################################################################
# 将所有参数的梯度缓存(gradient buffers)置 0, 并使用随机梯度反向传播:
net.zero_grad()
out.backward(torch.randn(1, 10))

########################################################################
# .. note::
#
#     ``torch.nn`` 仅支持 mini-batches。 整个 ``torch.nn``
#     package 仅支持以样本的 mini-batch 作为输入，而不支持单个样本作为输入。
#
#     例如, ``nn.Conv2d`` 将接受一个shape为 ``nSamples x nChannels x Height x Width`` 
#     的 4D Tensor 作为输入。
#
#     如果你有一个单样本, 请使用 ``input.unsqueeze(0)`` 为其添加一个虚构的batch纬度。
#
# 在进一步处理之前, 让我们回顾一下目前为止你用到的所有类 吧。
#
# **总结:**
#   -  ``torch.Tensor`` - 一个 *多维数组* , 支持自动微分(autograd)
#      操作 比如 ``backward()`` 。 它还 *持有梯度* w.r.t. the tensor。
#   -  ``nn.Module`` - 神经网络模块。*封装参数的便捷方式* ,带有一些帮助函数用于将模型迁移到GPU上, 模型的导出加载,等等。
#   -  ``nn.Parameter`` - 一种Tensor,是被作为参数自动注册的,当把它作为一个属性(attribute)分配给一个 ``Module`` 的时候。
#   -  ``autograd.Function`` - 实现了 *一个自动微分运算(autograd operation) 的 前向和反向定义(forward and backward definitions)*。 
#      每一个 ``Tensor`` 运算/操作, 创建至少一个单个的 ``Function`` 节点，该节点连接着那些创建了一个 ``Tensor`` 的函数以及*编码它的历史*的函数。
#
# **当目前为止, 我们学习了:**
#   -  定义一个神经网络
#   -  处理输入，调用 backward 。
#
# **剩余的步骤:**
#   -  计算损失
#   -  更新网络权重
#
# 损失函数
# -------------
# 一个损失函数接受 (output, target) 作为输入，然后计算一个估计网络输出离我们的期望输出还有多远的评估值。
#
# 在 nn package 里面有各种不同形式的损失函数(`loss functions <https://pytorch.org/docs/nn.html#loss-functions>`_) 。
# 一个简单的损失函数是: ``nn.MSELoss`` ，计算损失函数的输入与目标值之间的平均平方误差(mean-squared error)。
#
# 举个栗子:

output = net(input)
target = torch.randn(10)  # 一个虚拟的目标值, 为了举例子，不要太在意
target = target.view(1, -1)  # 使其具有与输出相同的shape
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)


########################################################################
# 现在，如果你使用 ``loss`` 的 ``.grad_fn`` 属性向后跟踪 ``loss`` ，将看到如下所示的计算图:
#
# ::
#
#     input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
#           -> view -> linear -> relu -> linear -> relu -> linear
#           -> MSELoss
#           -> loss
#
# 因此, 当我们调用 ``loss.backward()`` 时, 整个的计算图被求了相对于loss的微分(the whole graph is differentiated
# w.r.t. the loss), 并且计算图中所有的 Tensors 只要其满足 ``requires_grad=True`` 就会有它们自己的用梯度累积起来的 ``.grad`` Tensor。
#
# 为了说明, 让我们向后跟踪几步吧:

print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU


########################################################################
# 反向传播
# --------
# 为了反向传播误差，我们要做的所有的事就是去调用 ``loss.backward()`` 。
# 但是在此之前，你必须先将已经存在的梯度清除，否则梯度会被累加到上一批次迭代时产生的旧的梯度上。
#
#
# 现在我们可以调用 ``loss.backward()``, 然后我们看看 
# conv1 的 bias gradients 在backward之前和之后的变化。


net.zero_grad()     # 将所有参数的梯度缓存(gradient buffers)置零

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)


########################################################################
# 现在我们已经知道如何使用损失函数啦.
#
# **接着阅读:**
#
#   神经网络包(nn package)包含各种模块和损失函数，
#   构成深度神经网络的构建块，有文档的完整列表在(`这里 <https://pytorch.org/docs/nn>`_)。
#
# **剩下要学习的唯一的事情就是:**
#
#   - 更新网络权重
#
# 更新权重
# ------------------
# 实践中使用的最简单的更新规则是随机梯度下降(Stochastic Gradient Descent, SGD):
#
#      ``weight = weight - learning_rate * gradient``
#
# 我们可以用简单的Python代码来实现这个优化过程:
#
# .. code:: python
#
#     learning_rate = 0.01
#     for f in net.parameters():
#         f.data.sub_(f.grad.data * learning_rate)
#
# 然而，在使用神经网络时，您需要使用各种不同的更新规则，如SGD、Nesterov-SGD、ADAM、RMSProp等。
# 为此，我们构建了一个小包: ``torch.optim`` ，它实现了所有这些方法。使用它非常简单：

import torch.optim as optim

# 创建你的优化器
optimizer = optim.SGD(net.parameters(), lr=0.01)

# 在你的每一次训练回环(training loop)中:
optimizer.zero_grad()   # 将梯度缓存置零
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # 执行更新


###############################################################
# .. Note::
#
#       观察如何使用 ``optimizer.zero_grad()`` 手动将梯度缓存设置为零。
#       这是因为梯度是按反向传播小节中解释的那样累积的。
#
#
