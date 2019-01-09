# -*- coding: utf-8 -*-
"""
PyTorch: nn
-----------

一个完全连接的ReLU网络，只有一个隐藏层，没有偏置，最小化欧氏误差训练从x预测y。

这个实现使用PyTorch的nn包来构建网络。PyTorch Autograd使定义计算图和获取梯度变得很容易，
但是对于定义复杂的神经网络来说，原始的自动梯度可能太低级了；这就是nn包可以提供帮助的地方。
nn包定义了一组Modules，可以把它看作是一个神经网络层，它产生输入的输出，
并且可能具有一些可训练的权重。
"""
import torch

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# 使用 nn package 来把我们的模型定义为layers构成的序列。nn.Sequential
# 是一个包含了其他Modules的Module, 并把它们应用在序列中产生输出。
# 每个Linear Module使用线性函数从输入计算输出，并且持有内部张量用于存储它的权重和偏置。
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)

# nn package 也包含了各种广泛使用的损失函数;
# 在这里，我们使用 Mean Squared Error (MSE) 作为我们的损失函数。
loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-4
for t in range(500):
    # 前向传递: 把 x 传入 model 计算预测输出 y 。因为 Module objects 重载了 
    # __call__ 这个魔法函数，所以你可以像调用函数一样调用 model 。
    # 当你这么做的时候，你要把输入数据的Tensor传递到Module里面，并产生输出数据的Tensor.
    y_pred = model(x)

    # 计算并输出 loss. 我们把包含预测值的张量 y_pred 和真实值的张量 y 都传入损失函数，
    # 损失函数返回一个包含损失的张量。
    loss = loss_fn(y_pred, y)
    print(t, loss.item())

    # 在运行反向传播之前先将模型内部的梯度缓存都清零
    model.zero_grad()

    # 反向传递: 计算损失相对模型中所有可学习参数的梯度
    # 在内部, 每个 Module 的参数被存储在状态为
    # requires_grad=True 的 Tensors 中, 所以调用backward()后，
    # 将会计算模型中所有可学习参数的梯度。
    loss.backward()

    # 使用梯度下降算法更新权重. 每个参数是一个Tensor, 因此
    # 我们可以像之前一样通过 param.grad 来获取梯度
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad
