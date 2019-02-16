# -*- coding: utf-8 -*-
"""
PyTorch: 自定义 nn Modules
--------------------------

一个完全连接的ReLU网络，只有一个隐藏层，没有偏置，最小化欧氏误差训练从x预测y。

该实现将模型定义为自定义模块的子类。当你需要一个比已有的简单序列化模块更复杂的模型的时候，
你就需要用这种方式来定义你的模型。

请注意：这里有两个词，一个是 模型(model);另一个是 模块(module)。我们可以用模块的方式来定义模型
"""
import torch


class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        在构造函数中，我们实例化了两个nn.Linear模块，
        并将它们赋值为成员变量。
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        在前馈函数中，我们接受一个输入数据的 Tensor，
        并且我们必须返回输出数据的Tensor。在这里
        我们可以使用造函数中已经定义好的Modules和
        其他任意的Tensors上的算子来完成前馈函数的任务逻辑。
        """
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# 创建持有输入和输出的随机张量
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# 通过实例化上面定义的类，来创建模型
model = TwoLayerNet(D_in, H, D_out)

# 构建我们的损失函数和优化器。在SGD的构造器中调用 model.parameters()
# 将会包含来自两个nn.Linear modules的可学习参数；
# 这两个 nn.Linear modules 是我们自定义的模型的类成员。
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
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
