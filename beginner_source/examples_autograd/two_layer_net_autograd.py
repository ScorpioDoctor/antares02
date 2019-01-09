# -*- coding: utf-8 -*-
"""
PyTorch: 张量和自动梯度
-------------------------------

一个完全连接的ReLU网络，只有一个隐藏层，没有偏置，最小化欧氏误差训练从x预测y。

这个实现使用Pytorch的tensors上的运算操作计算前向传递，
并使用PyTorch的autograd计算梯度。

一个 PyTorch Tensor 代表了计算图上的一个节点。 如果 ``x`` 是一个状态为
``x.requires_grad=True`` 的张量，那么 ``x.grad`` 是另一个张量，它持有
``x`` 相对于某个标量的梯度。
"""
import torch

dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") #去掉这行注释就可以在GPU上运行

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# 创建随机张量以持有输入和输出.
# 设置 requires_grad=False 表明 我们在反向传递阶段
# 不需要计算相对于这些张量的梯度 
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# 创建随机张量用来存放模型的可学习参数: weights
# 设置 requires_grad=True 表明 我们在反向传递阶段
# 需要计算相对于这些张量的梯度 
w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    # 前向传递: 计算预测出的 y 使用Tensors相关的运算/操作; 
    # 这个地方与上一节中使用Tensor的同样的操作计算前向传递是一样的；
    # 但是我们不需要保留计算过程的中间值的引用，
    # 因为我们并没有去手动实现反向传递。
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    # 使用Tensors的操作 计算损失并输出
    # 现在损失是一个 shape 为 (1,) 的张量
    # loss.item() 可以获得张量loss中持有的数字
    loss = (y_pred - y).pow(2).sum()
    print(t, loss.item())

    # 使用 autograd 去计算反向传递。 这个调用将会计算
    # loss相对于所有状态为 requires_grad=True 的张量的梯度。
    # 调用完毕以后， w1.grad 和 w2.grad 将会是两个张量，分别持有
    # 损失相对于 w1 和 w2 的梯度。
    loss.backward()

    # 使用梯度下降法手动更新权重。并将代码分装在 torch.no_grad() 中。
    # 因为 权重张量的状态为 requires_grad=True, 但是我们不希望在
    # autograd 中去跟踪历史.
    # 另一种可选的方法是 直接操作 weight.data 和 weight.grad.data 。
    # 回想到 tensor.data 给出一个与其共享存储空间的张量，但是不会跟踪历史。
    # 你也可以使用 torch.optim.SGD 来达到此目的。
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # 更新完权重以后，手动将所有的梯度清零
        w1.grad.zero_()
        w2.grad.zero_()
