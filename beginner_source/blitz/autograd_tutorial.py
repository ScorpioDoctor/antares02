# -*- coding: utf-8 -*-
"""
Autograd: 自动微分
===================================
**翻译者**: `Antares博士 <http://www.studyai.com/antares>`_

在PyTorch的所有神经网络中，核心是 ``autograd`` 包。让我们先简单介绍一下，
然后我们将开始训练我们的第一个神经网络。


``autograd`` package 为张量上的所有操作提供自动微分(automatic differentiation)。
它是一个按运行定义的框架(define-by-run framework)，
这意味着您的后端(backprop)由您的代码运行方式来定义，并且每个迭代都可能是不同的。

让我们用更简单的术语来看这一点，并举几个例子。

张量(Tensor)
--------------

``torch.Tensor`` 是此package的核心类。 如果你将它的属性 ``.requires_grad`` 设置为 ``True``, 
它就开始跟踪在它上面的所有运算操作。当你完成计算时你可以调用 ``.backward()`` ，
这会使得所有的梯度都被自动计算出来。对于这个tensor的梯度将会被累加到 ``.grad`` 属性中去。

如果想要阻止一个tensor不去跟踪历史(tracking history), 你可以调用 ``.detach()`` 方法
把它从计算历史中分离出来, 并且会阻止进一步的计算被跟踪。 

若要防止跟踪历史记录(并使用内存)，还可以把代码块封装在with语句 ``with torch.no_grad():`` 中。 
这在评估模型时特别有用，因为模型可能具有可训练的参数(`requires_grad=True`)。但是在评估模型的
时候不需要计算梯度，而且我们不想把模型的这些可训练参数设置为 `requires_grad=False` ，那么封装在
with语句 ``with torch.no_grad():`` 中是很赞的。

还有一个类对于实现自动微分至关重要，那就是--- ``Function`` 。

``Tensor`` 和 ``Function`` 是内部相互联系的，并建立了一个无环图(acyclic graph)，它编码了一个完整的计算历史。
每个tensor都有一个 ``.grad_fn`` 属性，它引用了创建了 ``Tensor`` 的 ``Function`` 。
(除了由用户创建的 Tensors -它们的 ``grad_fn is None``)。

如果要计算导数(derivatives)，可以在 ``Tensor`` 上调用 ``.backward()`` 。 
如果 ``Tensor`` 是一个标量(scalar) (i.e. 它里面只持有一个元素的数据), 
那么你不需要为 ``backward()`` 方法传递任何参数。然而，如果 ``Tensor`` 有更多的元素，那么
你需要指定一个 ``gradient`` 参数，其必须是一个shape相匹配的 tensor 。
"""

import torch

###############################################################
# 创建一个 tensor 并设置 requires_grad=True 来跟踪这个tensor上的计算
x = torch.ones(2, 2, requires_grad=True)
print(x)

###############################################################
# 对 tensor 做运算:
y = x + 2
print(y)

###############################################################
# ``y`` 作为加法运算的结果被创建了出来, 因此它有一个 ``grad_fn``.
print(y.grad_fn)

###############################################################
# 在张量  ``y`` 上做更多运算操作
z = y * y * 3
out = z.mean()

print(z, out)

################################################################
# ``.requires_grad_( ... )`` 可以原位(in-place)修改一个已经存在的 
# Tensor 的 ``requires_grad`` 标志位。
# 如果没有给定， 输入的标志位默认是 ``False`` 。
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)
print(b.requires_grad)

###############################################################
# 梯度(Gradient)
# --------------------
# 现在我们开始反向传播啦
# 因为 ``out`` 包含一个单个的标量, ``out.backward()`` 
# 是等价于 ``out.backward(torch.tensor(1.))`` 的。

out.backward()

###############################################################
# 输出梯度 d(out)/dx
#

print(x.grad)

###############################################################
# 你应该得到了一个 ``4.5`` 的2x2矩阵。 我们把 ``out`` 称为
# *Tensor* “:math:`o`” 。
# 我们有这样一个式子成立 :math:`o = \frac{1}{4}\sum_i z_i`,
# :math:`z_i = 3(x_i+2)^2` 和 :math:`z_i\bigr\rvert_{x_i=1} = 27`.
# 因此,
# :math:`\frac{\partial o}{\partial x_i} = \frac{3}{2}(x_i+2)`, 因此
# :math:`\frac{\partial o}{\partial x_i}\bigr\rvert_{x_i=1} = \frac{9}{2} = 4.5`.

###############################################################
# 数学上, 如果你有一个向量值函数(vector valued function) :math:`\vec{y}=f(\vec{x})`,
# 那么 :math:`\vec{y}` 相对于 :math:`\vec{x}` 的梯度
# 是一个雅克比矩阵(Jacobian matrix) :
#
# .. math::
#   J=\left(\begin{array}{ccc}
#    \frac{\partial y_{1}}{\partial x_{1}} & \cdots & \frac{\partial y_{m}}{\partial x_{1}}\\
#    \vdots & \ddots & \vdots\\
#    \frac{\partial y_{1}}{\partial x_{n}} & \cdots & \frac{\partial y_{m}}{\partial x_{n}}
#    \end{array}\right)
#
# 广义上说, ``torch.autograd`` 是一个用来计算雅克比向量乘积(Jacobian-vector product)的引擎。
# 这就是说, 给定任意的向量 
# :math:`v=\left(\begin{array}{cccc} v_{1} & v_{2} & \cdots & v_{m}\end{array}\right)^{T}`,
# 计算乘积 :math:`J\cdot v` 。 如果 :math:`v` 恰好是一个标量函数 :math:`l=g\left(\vec{y}\right)` 的梯度，
# 即,
# :math:`v=\left(\begin{array}{ccc}\frac{\partial l}{\partial y_{1}} & \cdots & \frac{\partial l}{\partial y_{m}}\end{array}\right)^{T}`,
# 那么根据链式法则, 雅克比向量乘积 就是 :math:`l` 相对于 :math:`\vec{x}` 的梯度 :
#
# .. math::
#   J\cdot v=\left(\begin{array}{ccc}
#    \frac{\partial y_{1}}{\partial x_{1}} & \cdots & \frac{\partial y_{m}}{\partial x_{1}}\\
#    \vdots & \ddots & \vdots\\
#    \frac{\partial y_{1}}{\partial x_{n}} & \cdots & \frac{\partial y_{m}}{\partial x_{n}}
#    \end{array}\right)\left(\begin{array}{c}
#    \frac{\partial l}{\partial y_{1}}\\
#    \vdots\\
#    \frac{\partial l}{\partial y_{m}}
#    \end{array}\right)=\left(\begin{array}{c}
#    \frac{\partial l}{\partial x_{1}}\\
#    \vdots\\
#    \frac{\partial l}{\partial x_{n}}
#    \end{array}\right)
#
# 雅克比向量乘积的这个特点使得 将外部梯度输入到一个具有非标量输出的模型中去 变得非常方便。

###############################################################
# 现在呢 我就来看看 雅克比向量乘积(Jacobian-vector product) 的一个例子:

x = torch.randn(3, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print(y)

###############################################################
# 现在 ``y`` 不再是一个标量啦。 ``torch.autograd``
# 不能直接计算出整个雅可比矩阵, 但如果我们只想要雅可比向量积(Jacobian-vector product), 
# 只需要简单的传递一个向量到函数 ``backward`` 的参数中去:
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)

print(x.grad)

###############################################################
# 您还可以通过将代码块包装在下面的 ``with torch.no_grad()`` 代码块中，
# 从而停止使用autograd来跟踪状态为 ``.requires_grad=True`` 的 tensors 上的历史记录:
print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
	print((x ** 2).requires_grad)

###############################################################
# **后续阅读:**
#
# ``autograd`` 和 ``Function`` 的文档在 
# https://pytorch.org/docs/autograd
