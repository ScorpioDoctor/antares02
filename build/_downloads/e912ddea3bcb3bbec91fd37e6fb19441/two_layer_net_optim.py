# -*- coding: utf-8 -*-
"""
PyTorch: optim
--------------

一个完全连接的ReLU网络，只有一个隐藏层，没有偏置，最小化欧氏误差训练从x预测y。

这个实现利用了PyTorch中的 nn package 来构建网络。

我们没有像我们之前的例子中一直做的那样手动更新模型的权重，
而是使用optim package来定义一个将为我们更新权重的优化器。
optim包定义了许多在深度学习中常用的优化算法，包括SGD+动量、RMSProp、Adam等。
"""
import torch

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# 创建持有输入和输出的随机张量
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# 使用 nn package 来 定义模型和损失函数
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)
loss_fn = torch.nn.MSELoss(reduction='sum')

# 使用 optim package 来定义一个优化器(Optimizer),用于为我们更新模型的权重。
# 这里我们使用 Adam; optim package 包含很多其他的优化算法。
# Adam 构造函数的第一个参数告诉优化器哪些Tensors需要被更新。
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(500):
    # 前向传递: 通过把x传入模型来计算 预测值 y。
    y_pred = model(x)

    # 计算并输出 loss.
    loss = loss_fn(y_pred, y)
    print(t, loss.item())

    # 在向后传递之前，使用优化器对象把它将要更新的变量(模型的可学习参数)的所有梯度变为零。
    # 这是因为默认情况下，不管啥时候调用.backward()，梯度都会累积到缓存(i.e. 不是重新写入)。
    # 请查看 torch.autograd.backward 的文档获得更多信息。
    optimizer.zero_grad()

    # 向后传递: 计算损失相对于模型参数的梯度
    loss.backward()

    # 调用 Optimizer 的 step 函数对参数进行一步更新
    optimizer.step()
