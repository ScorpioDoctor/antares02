# -*- coding: utf-8 -*-
"""
PyTorch: 控制流 + 权重共享
--------------------------------------

为了展示PyTorch动态图的威力，我们将实现一个非常奇怪的模型：
一个完全连接的relu网络，
它在每一次前向传递中随机选择一个1到4之间的数字，
并且在这次前向传递中就使用随机选择的这个数字这么多的隐藏层，
重复使用相同的权重多次计算最内部的隐藏层。
"""
import random
import torch


class DynamicNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        在构造函数中，我们创建3个 nn.Linear 的实例，它们将被用于前向传递中。
        """
        super(DynamicNet, self).__init__()
        self.input_linear = torch.nn.Linear(D_in, H)
        self.middle_linear = torch.nn.Linear(H, H)
        self.output_linear = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        在模型的前向传递中, 我们随机的选择 0, 1, 2, 或 3 中的一个数字，
        然后我们就重复使用middle_linear Module那么多次 作为计算隐藏层的表示。

        因为每一次前向传递都会构建一个动态的计算图，我们在定义模型的前向计算过程时
        可以使用普通的Python控制流操作比如 for-loops 或 条件表达式。

        在这里，我们还看到，在定义计算图时，多次重用同一个模块是完全安全的。
        这是对Lua Torch的一个很大的改进，在那里每个模块只能使用一次。
        """
        h_relu = self.input_linear(x).clamp(min=0)
        for _ in range(random.randint(0, 3)):
            h_relu = self.middle_linear(h_relu).clamp(min=0)
        y_pred = self.output_linear(h_relu)
        return y_pred


# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# 创建持有输入和输出的随机张量
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# 通过实例化上面定义的类，来创建模型
model = DynamicNet(D_in, H, D_out)

# 构建我们的损失函数和优化器。使用 普通的SGD 来训练这个奇怪的模型是很难的，
# 所以我们使用了带有动量项的SGD来优化模型。
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
for t in range(500):
    # 前向过程: 把 x 传递给model, 计算 predicted y 
    y_pred = model(x)

    # 计算并输出loss
    loss = criterion(y_pred, y)
    print(t, loss.item())

    # 把梯度置零， 执行后向传递, 以及 更新权重
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
