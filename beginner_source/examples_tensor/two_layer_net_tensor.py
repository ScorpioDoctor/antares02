# -*- coding: utf-8 -*-
"""
PyTorch: 张量
----------------

一个完全连接的ReLU网络，只有一个隐藏层，没有偏置，最小化欧氏误差训练从x预测y。

此实现使用PyTorch tensors 手动计算向前传递、损失和反向传递。

PyTorch Tensor 基本类似于 Numpy 数组: 它不知道任何关于深度学习、梯度或计算图的知识，
它只是执行通用数字计算的一种方法。

Numpy数组和PyTorch张量之间最大的区别是PyTorch张量可以在CPU或GPU上运行。
要在GPU上运行操作，只需将张量转换为cuda数据类型即可。
"""

import torch


dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # 去掉这行注释就可以在GPU上运行

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# 产生随机输入和输出数据
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# 随机初始化权重
w1 = torch.randn(D_in, H, device=device, dtype=dtype)
w2 = torch.randn(H, D_out, device=device, dtype=dtype)

learning_rate = 1e-6
for t in range(500):
    # 前向传递: 计算 predicted y
    h = x.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)

    # 计算并输出损失
    loss = (y_pred - y).pow(2).sum().item()
    print(t, loss)

    # 反向传播(Backprop) 去计算 w1 和 w2 相对于loss的梯度
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)

    # 使用梯度下降法更新权重
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
