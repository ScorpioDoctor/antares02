# -*- coding: utf-8 -*-
"""
空间变换网络(STN)教程
=====================================
**翻译者**: `Antares博士 <http://www.studyai.com/antares>`__

.. figure:: /_static/img/stn/FSeq.png

在本教程中，您将学习如何使用一种称为空间变换器网络的视觉注意机制来增强您的网络。
您可以在 `DeepMind paper <https://arxiv.org/abs/1506.02025>`__ 中更多地阅读有关空间变换器网络的内容。

空间变换器网络(Spatial transformer networks, STN)是对任何空间变换的可微关注(differentiable attention)的推广。
STN允许神经网络学习如何对输入图像进行空间变换，以提高模型的几何不变性。
例如，它可以裁剪感兴趣的区域、缩放和纠正图像的方向。
这是一种有用的机制，因为CNN对图像旋转、尺度和更一般的仿射变换不具有不变性。

关于STN最好的事情之一是能够简单地将它插入到任何现有的CNN中，而很少做任何修改。
"""
# License: BSD
# Author: Ghassen Hamrouni

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
plt.ion()   # 交互式模式

######################################################################
# 加载数据
# ----------------
#
# 在这篇文章中，我们实验了经典的MNIST数据集。使用标准卷积网络和STN网络进行增广。
# 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 训练数据集
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='./data/mnist', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])), batch_size=64, shuffle=True, num_workers=4)
# 测试数据集
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='./data/mnist', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])), batch_size=64, shuffle=True, num_workers=4)

######################################################################
# 描述 STN 网络
# --------------------------------------
#
# 空间变换器网络可归结为三个主要组成部分:
#
# -  定位网络(localization network)是一个规则的CNN网络，它对变换参数进行回归。
#    该变换从不从此数据集中显式学习，而是由网络自动学习提高全局精度的空间转换。
# -  网格生成器从输出图像中生成与每个像素对应的输入图像中的坐标网格。
# -  采样器使用变换器的参数并将其应用于输入图像。
#
# .. figure:: /_static/img/stn/stn-arch.png
#
# .. Note::
#    我们需要包含 affine_grid 和 grid_sample modules 的 PyTorch.
#    (PyToch 1.0 及以上)。
#


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # 空间变换定位网络(localization-network)
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # 用于估计 3 * 2 仿射矩阵的回归网络
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # 使用恒等变换(identity transformation)初始化 weights/bias
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # STN网络的前向函数
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # 对输入进行变换
        x = self.stn(x)

        # 执行通常的前向传递
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


model = Net().to(device)

######################################################################
# 训练模型
# ------------------
#
# 现在，让我们使用SGD算法来训练模型。网络以监督的方式学习分类任务。同时，
# 模型以端到端的方式自动学习STN。
# 

optimizer = optim.SGD(model.parameters(), lr=0.01)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 500 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
#
# 在MNIST上测试STN性能的一种简单的测试方法。 
#


def test():
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # sum up batch loss
            test_loss += F.nll_loss(output, target, size_average=False).item()
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
              .format(test_loss, correct, len(test_loader.dataset),
                      100. * correct / len(test_loader.dataset)))

######################################################################
# 可视化 STN 的结果
# ---------------------------
#
# 现在，我们将检查我们学习到的视觉注意机制的结果。
#
# 我们定义了一个小辅助函数，以便在训练时可视化变换。
# 

def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp

# 在训练结束后，我们要可视化STN层的输出。
# 可视化一批输入图像和对应的使用STN变换产生的批次。


def visualize_stn():
    with torch.no_grad():
        # Get a batch of training data
        data = next(iter(test_loader))[0].to(device)

        input_tensor = data.cpu()
        transformed_input_tensor = model.stn(data).cpu()

        in_grid = convert_image_np(
            torchvision.utils.make_grid(input_tensor))

        out_grid = convert_image_np(
            torchvision.utils.make_grid(transformed_input_tensor))

        # Plot the results side-by-side
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(in_grid)
        axarr[0].set_title('Dataset Images')

        axarr[1].imshow(out_grid)
        axarr[1].set_title('Transformed Images')

for epoch in range(1, 20 + 1):
    train(epoch)
    test()

# Visualize the STN transformation on some input batch
visualize_stn()

plt.ioff()
plt.show()
