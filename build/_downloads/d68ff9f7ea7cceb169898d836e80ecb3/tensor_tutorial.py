# -*- coding: utf-8 -*-
"""
PyTorch是什么?
================
**翻译者**: `Antares博士 <http://www.studyai.com/antares>`_

这是一个基于Python的科学计算包，针对两组受众:

-  想要替换NumPy从而使用GPU的计算能力
-  提供最大灵活性和速度的深度学习研究平台

开始走起
---------------

Tensors
^^^^^^^

Tensors 类似于NumPy的 ndarrays, 另外，它还可以在GPU上使用加速计算。
"""

from __future__ import print_function
import torch

###############################################################
# 构建一个 5x3 矩阵, 未初始化:

x = torch.empty(5, 3)
print(x)


###############################################################
# 构建一个随机初始化的矩阵:

x = torch.rand(5, 3)
print(x)


###############################################################
# 创建一个矩阵，用 0 填充，数据类型为 long:

x = torch.zeros(5, 3, dtype=torch.long)
print(x)


###############################################################
# 直接从已有的数据(不是Tensor类型)创建一个矩阵:

x = torch.tensor([5.5, 3])
print(x)

###############################################################
# 或者基于一个已有的tensor创建一个新的tensor。这类方法将会重用
# 输入tensor的属性, e.g. dtype, 除非用户提供了新的属性值

x = x.new_ones(5, 3, dtype=torch.int)      # new_* 方法需要接受 sizes 参数
print(x)

x = torch.randn_like(x, dtype=torch.float)    # 覆盖上面的 x 的 dtype!
print(x)                                      # 结果有相同的 size


###############################################################
# 获得张量的 size:

print(x.size())

###############################################################
# .. note::
#     ``torch.Size`` 事实上是个元祖(tuple),因此它支持所有的元祖操作
#
# 操作
# ^^^^^^^^^^
# 张量的运算有多种语法。在下面的示例中，我们将查看加法运算，减法运算以此为例。
#
# 加法: 语法 1
y = torch.rand(5, 3)
print(x + y)


###############################################################
# 加法: 语法 2

print(torch.add(x, y))


###############################################################
# 加法: 提供一个输出张量作为参数
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)


###############################################################
# 加法: 原位操作(in-place)

# 把 x 加到 y 上
y.add_(x)
print(y)


###############################################################
# .. note::
#     任何原位修改一个张量的运算操作都带有下划线后缀: ``_`` 。
#     比如: ``x.copy_(y)``, ``x.t_()``, 将会改变 ``x``.
#
# 您可以使用标准的NumPy类索引(standard NumPy-like indexing) 以及它的所有一切花哨的索引技巧!

print(x[:, 1])


###############################################################
# 调整大小(Resizing): 如果要调整张量的大小/形状(resize/reshape tensor)，可以使用 ``torch.view``:
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # 这个 size -1 是可以根据其他维的维数和总的维数推断出来的
print(x.size(), y.size(), z.size())


###############################################################
# 如果你有一个单元素张量(one element tensor), 可以使用 ``.item()`` 来取得里面的值，将其作为普通的Python number。
x = torch.randn(1)
print(x)
print(x.item())


###############################################################
# **稍后阅读:**
#
#
#   100+ Tensor 运算操作, 包括 transposing, indexing, slicing,
#   mathematical operations, linear algebra, random numbers, etc.,
#   点击下面的链接查看 `here <https://pytorch.org/docs/torch>`_.
#
# NumPy 桥接
# ------------
#
# 把Torch Tensor转换成NumPy array是很easy的，反之亦然。
#
# Torch Tensor 和 NumPy array 将会共享底层内存位置, 并且 修改了一个后另个也会被改变.
#
# 把 Torch Tensor 转换为一个 NumPy Array
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

a = torch.ones(5)
print(a)


###############################################################
#

b = a.numpy()
print(b)

###############################################################
# 请看下面 numpy array 的数值是如何改变的.

a.add_(1)
print(a)
print(b)



###############################################################
# 把 NumPy Array 转为 Torch Tensor
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 请看如何自动的把 np array 转变为 Torch Tensor:

import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)


###############################################################
# 所有在CPU 上的张量除了CharTensor 之外都支持与Numpy之间的相互转换。
#
# CUDA 张量
# ------------
#
# 可以使用 ``.to`` 方法把张量移动到任意的设备。

# 让我们运行下面这段代码仅仅当 CUDA 可用的时候。
# 我们将使用 ``torch.device`` 对象把张量在CPU和GPU之间移进来移出去。
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!

#########################################################################
# .. rst-class:: sphx-glr-script-out
# 
#  Out:
# 
#  .. code-block:: none
# 
#     tensor([0.8812], device='cuda:0')
#     tensor([0.8812], dtype=torch.float64)
# 