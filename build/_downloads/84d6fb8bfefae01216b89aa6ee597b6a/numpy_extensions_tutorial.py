# -*- coding: utf-8 -*-
"""
使用 numpy 和 scipy 创建扩展
=========================================
**翻译者**: `Antares博士 <http://www.studyai.com/antares>`_

在本教程中, 我们要完成两个任务:

1. 创建一个没有参数的神经网络:

    -  调用 **numpy** 作为其实现的一部分

2. 创建一个有可学习参数的神经网络:

    -  调用 **SciPy** 作为其实现的一部分
"""

import torch
from torch.autograd import Function

###############################################################
# 无参数的示例
# ----------------------
#
# 这一层没有做任何有用的或数学上正确的事情。
#
# 被恰当地命名为  BadFFTFunction
#
# **Layer 实现**

from numpy.fft import rfft2, irfft2


class BadFFTFunction(Function):

    def forward(self, input):
        numpy_input = input.detach().numpy()
        result = abs(rfft2(numpy_input))
        return input.new(result)

    def backward(self, grad_output):
        numpy_go = grad_output.numpy()
        result = irfft2(numpy_go)
        return grad_output.new(result)

# 由于该层没有任何参数，所以我们可以简单地将其声明为函数，而不是 nn.Module 类。


def incorrect_fft(input):
    return BadFFTFunction()(input)

###############################################################
# **怎么使用自己创造的Layers:**

input = torch.randn(8, 8, requires_grad=True)
result = incorrect_fft(input)
print(result)
result.backward(torch.randn(result.size()))
print(input)

###############################################################
# 参数化的示例
# --------------------
#
# 在深度学习的文献中, 这个层被含糊的称为卷积层而实际上的操作是交叉互相关(cross-correlation)
# (卷积和交叉互相关的唯一区别是 做卷积的时候滤波器核会被反转，而交叉互相关则不要反转滤波器核)。
#
# 具有可学习权值的层的实现，其中互相关有一个表示权重的滤波器核。
#
# 反向传递计算损失相对于输入的梯度和损失相对于滤波器的梯度。
# 

from numpy import flip
import numpy as np
from scipy.signal import convolve2d, correlate2d
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


class ScipyConv2dFunction(Function):
    @staticmethod
    def forward(ctx, input, filter, bias):
        # detach so we can cast to NumPy
        input, filter, bias = input.detach(), filter.detach(), bias.detach()
        result = correlate2d(input.numpy(), filter.numpy(), mode='valid')
        result += bias.numpy()
        ctx.save_for_backward(input, filter, bias)
        return torch.as_tensor(result, dtype=input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.detach()
        input, filter, bias = ctx.saved_tensors
        grad_output = grad_output.numpy()
        grad_bias = np.sum(grad_output, keepdims=True)
        grad_input = convolve2d(grad_output, filter.numpy(), mode='full')
        # the previous line can be expressed equivalently as:
        # grad_input = correlate2d(grad_output, flip(flip(filter.numpy(), axis=0), axis=1), mode='full')
        grad_filter = correlate2d(input.numpy(), grad_output, mode='valid')
        return torch.from_numpy(grad_input), torch.from_numpy(grad_filter).to(torch.float), torch.from_numpy(grad_bias).to(torch.float)


class ScipyConv2d(Module):
    def __init__(self, filter_width, filter_height):
        super(ScipyConv2d, self).__init__()
        self.filter = Parameter(torch.randn(filter_width, filter_height))
        self.bias = Parameter(torch.randn(1, 1))

    def forward(self, input):
        return ScipyConv2dFunction.apply(input, self.filter, self.bias)


###############################################################
# **用法示例:**

module = ScipyConv2d(3, 3)
print("Filter and bias: ", list(module.parameters()))
input = torch.randn(10, 10, requires_grad=True)
output = module(input)
print("Output from the convolution: ", output)
output.backward(torch.randn(8, 8))
print("Gradient for the input map: ", input.grad)

###############################################################
# **检查梯度:**

from torch.autograd.gradcheck import gradcheck

moduleConv = ScipyConv2d(3, 3)

input = [torch.randn(20, 20, dtype=torch.double, requires_grad=True)]
test = gradcheck(moduleConv, input, eps=1e-6, atol=1e-4)
print("Are the gradients correct: ", test)
