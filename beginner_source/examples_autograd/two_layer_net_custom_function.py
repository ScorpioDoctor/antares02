# -*- coding: utf-8 -*-
"""
PyTorch: 定义新的自动梯度函数
----------------------------------------

一个完全连接的ReLU网络，只有一个隐藏层，没有偏置，最小化欧氏误差训练从x预测y。

这个实现 使用PyTorch Variables上的操作 计算前向传递，并使用
PyTorch autograd 计算梯度。

在这个实现中，我们自定义了一个 autograd function 来执行 ReLU 函数。
"""
import torch


class MyReLU(torch.autograd.Function):
    """
    我们可以通过定义 torch.autograd.Function 的子类
    并实现forward和backward函数来轻松地定义我们自己的
    autograd Functions 。
    """

    @staticmethod
    def forward(ctx, input):
        """
        在前向传递中，我们接收一个包含输入的Tensor并返回
        一个包含输出的Tensor。 ctx 是一个上下文对象，
        可以用于为反向计算存储信息。
        可以使用 ctx.save_for_backward 方法缓存任意对象，以便在向后传递中使用。
        """
        ctx.save_for_backward(input)
        return input.clamp(min=-2)

    @staticmethod
    def backward(ctx, grad_output):
        """
        在反向传递中，我们接收到一个包含了损失相对于输出的梯度的张量，
        并且我们需要计算损失相对于输入的梯度。
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < -2] = 0
        return grad_input


dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold input and outputs.
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# Create random Tensors for weights.
w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    # 要应用我们自定义的函数, 可以使用 Function.apply 方法. 
    # 我们给它起个别名 'relu'.
    relu = MyReLU.apply

    # 前向传递: 使用operations计算预测的 y ; 我们
    # 使用自定义的 autograd operation 计算 ReLU 。
    y_pred = relu(x.mm(w1)).mm(w2)

    # 计算并输出损失
    loss = (y_pred - y).pow(2).sum()
    print(t, loss.item())

    # 使用 autograd 去计算 backward pass.
    loss.backward()

    # 使用梯度下降法更新权重
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # 更新完权重以后，手动清零所有的梯度缓存
        w1.grad.zero_()
        w2.grad.zero_()
