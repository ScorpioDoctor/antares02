# -*- coding: utf-8 -*-
r"""
PyTorch的简单介绍
***********************
翻译者：www.studyai.com/antares

Torch 的张量库的介绍
======================================

所有的深度学习都是关于张量的计算，它是矩阵的推广，可以在多个维度上索引。
我们将在以后深入了解这意味着什么。首先，让我们看看我们能用张量做些什么。
"""
# Author: Robert Guthrie

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)


######################################################################
# 创建张量
# ~~~~~~~~~~~~~~~~
#
# 可以使用 torch.Tensor() 函数 从Python的list创建张量
#

# torch.tensor(data) 使用给定的数据创建一个 torch.Tensor 对象
V_data = [1., 2., 3.]
V = torch.tensor(V_data)
print(V)

# 创建一个矩阵
M_data = [[1., 2., 3.], [4., 5., 6]]
M = torch.tensor(M_data)
print(M)

# 创建一个尺寸为 2x2x2 的 3D tensor。
T_data = [[[1., 2.], [3., 4.]],
          [[5., 6.], [7., 8.]]]
T = torch.tensor(T_data)
print(T)


######################################################################
# 什么是三维张量？像这样想一想。如果你有一个向量，索引到向量会给你一个标量。
# 如果你有一个矩阵，索引到矩阵给你一个向量。
# 如果你有一个三维张量，那么索引到张量给你一个矩阵！
#
# 关于术语的注意事项：当我在本教程中说“张量”时，它指的是任何一个torch.Tensor对象。
# 矩阵和向量是torch.Tensor的特例，它们的维数分别为1和2。当我谈论三维张量时，我会明确地使用“三维张量(3D tensor)”这个词。
#

# Index into V and get a scalar (0 dimensional tensor)
print(V[0])
# Get a Python number from it
print(V[0].item())

# Index into M and get a vector
print(M[0])

# Index into T and get a matrix
print(T[0])


######################################################################
# 您还可以创建其他数据类型的张量。如您所见，默认情况是Float。
# 若要创建整数类型的张量，请尝试 torch.LongTensor()。
# 查看文档以获得更多的数据类型，但Float和Long将是最常见的。
# 


######################################################################
# 你可以用随机数据创建一个张量，提供维数,用 torch.randn() 创建。
#

x = torch.randn((3, 4, 5))
print(x)


######################################################################
# 张量的操作
# ~~~~~~~~~~~~~~~~~~~~~~~
#
# 你可以按你期望的方式对张量进行操作。

x = torch.tensor([1., 2., 3.])
y = torch.tensor([4., 5., 6.])
z = x + y
print(z)


######################################################################
# 有关可供您使用的大量操作的完整列表，请 `参阅文档 <https://pytorch.org/docs/torch.html>`__。
# 它们不仅仅包括数学运算。
#
# 我们稍后将使用的一个有用的操作是串接(concatenation)。
#

# 默认情况下，它沿着第一个轴(axis)进行串接 (concatenates rows)
x_1 = torch.randn(2, 5)
y_1 = torch.randn(3, 5)
z_1 = torch.cat([x_1, y_1])
print(z_1)

# Concatenate columns:
x_2 = torch.randn(2, 3)
y_2 = torch.randn(2, 5)
# 第二个参数用于指定要沿着哪个轴(axis)进行串接
z_2 = torch.cat([x_2, y_2], 1)
print(z_2)

# 如果你的张量不兼容，torch会抱怨。取消注释以查看错误
# torch.cat([x_1, x_2])


######################################################################
# 重塑(reshape)张量
# ~~~~~~~~~~~~~~~~~
#
# 使用.view()方法重新塑造张量(reshape a tensor)。由于许多神经网络组件期望的输入具有一定的形状(shape)，
# 这个方法(reshape, .view()方法)得到了大量的应用。通常，在将数据传递给组件之前，您需要进行整形(reshape)。
#

x = torch.randn(2, 3, 4)
print(x)
print(x.view(2, 12))  # Reshape to 2 rows, 12 columns
# Same as above.  如果其中一个维度是 -1, 它的size将会被推断出来
print(x.view(2, -1))


######################################################################
# 计算图和自动微分
# ================================================
#
# 
# 计算图的概念对于高效的深度学习编程至关重要，因为它允许您不必自己编写反向传播梯度。
# 计算图只是对如何组合数据以给出输出的一种规范。由于计算图完全指定了涉及哪些操作的参数，
# 所以它包含了足够的信息来计算导数。这可能听起来很模糊，所以让我们看看使用 基础标志
# ``requires_grad`` 将会发生什么。
#
# 首先，从程序员的角度思考。我们在上面创建的 torch.Tensor 对象中存储的是什么？
# 很明显，数据和形状，也许还有其他一些东西。但是当我们把两个张量相加时，
# 我们得到了一个输出张量。这个输出张量只知道它的数据和形状。
# 它不知道它是另外两个张量的总和(它可能是从文件中读取的，也可能是其他操作的结果，等等)。
#
# 如果 ``requires_grad=True``, Tensor 对象就可以跟踪它自己是如何被创建出来的。让我们看看具体的代码吧。
#

# Tensor factory methods have a ``requires_grad`` flag
x = torch.tensor([1., 2., 3], requires_grad=True)

# With requires_grad=True, you can still do all the operations you previously
# could
y = torch.tensor([4., 5., 6], requires_grad=True)
z = x + y
print(z)

# BUT z knows something extra.
print(z.grad_fn)


######################################################################
# 所以张量知道是什么创造了它们。Z 知道它不是从文件中读取的，它不是乘法或指数之类的结果。
# 如果你继续跟踪 z.grad_fn，你会在  x和 y 发现自己。
#
# 但这如何帮助我们计算梯度呢？
#

# Lets sum up all the entries in z
s = z.sum()
print(s)
print(s.grad_fn)


######################################################################
# 那么，这个和(sum)相对于x的第一个分量的导数是什么呢？在数学方面，我们想:
#
# .. math::
#
#    \frac{\partial s}{\partial x_0}
#
#
#
# 好吧, s 知道它是被作为张量 z 的和所创建的。z 知道它是 x + y 的和，因此：
#
# .. math::  s = \overbrace{x_0 + y_0}^\text{$z_0$} + \overbrace{x_1 + y_1}^\text{$z_1$} + \overbrace{x_2 + y_2}^\text{$z_2$}
#
# 因此 s 包含了足够的信息去计算出我们想要的导数就是 1 !!
#
# 当然，这掩盖了如何实际计算导数的挑战。这里的要点是，s携带了足够多的信息，因此可以计算它。
# 在现实中，Pytorch 的开发人员对sum()和+操作进行编程，以了解如何计算它们的梯度，
# 并运行反向传播算法。对该算法的深入讨论超出了本教程的范围.
#


######################################################################
# 让Pytorch计算梯度，看看我们是对的：(注意，如果您多次运行这个块，梯度就会增加。
# 这是因为Pytorch将梯度 **累加** 到 .grad 属性中，因为对于许多模型来说，这非常方便。)
#

# 调用任意变量的 .backward() 将会执行反向传播(backprop), 从该变量开始.
s.backward()
print(x.grad)


######################################################################
# 理解下面这个模块中正在发生的事情对于成为一名成功的深度学习的程序员来说是至关重要的。
#

x = torch.randn(2, 2)
y = torch.randn(2, 2)
# 默认情况下，用户创建的 Tensors 的 ``requires_grad=False``
print(x.requires_grad, y.requires_grad)
z = x + y
# So you can't backprop through z
print(z.grad_fn)

# ``.requires_grad_( ... )`` 会原位改变一个已经存在的张量的 ``requires_grad`` 标记
# The input flag defaults to ``True`` if not given.
x = x.requires_grad_()
y = y.requires_grad_()
# z 包含了足够的信息来计算梯度, as we saw above
z = x + y
print(z.grad_fn)
# If any input to an operation has ``requires_grad=True``, so will the output
print(z.requires_grad)

# 现在 z 拥有它自己和x以及y相关联的计算历史，
# 我们可以只接受它的值, 而把它从它的历史中 **detach** 出来吗?
new_z = z.detach()

# ... 那么 new_z 有信息可以反向传播到 x 和 y 吗?  **没有**
print(new_z.grad_fn)
# 那么为啥会这样的呢? ``z.detach()`` 返回一个与 ``z`` 共享存储空间的张量，但是把 ``z`` 上的
# 计算历史全忘记了(扔了)。 new_z 根本不知道它是如何被计算出来的。
# 从本质上说，我们已经把张量从过去的历史中剥离出来了。

###############################################################
# 还可以通过将代码块包装在 ``with torch.no_grad():`` 中来阻止autograd在
# ``.requires_grad``=True 中跟踪张量上的历史记录：
print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
	print((x ** 2).requires_grad)


