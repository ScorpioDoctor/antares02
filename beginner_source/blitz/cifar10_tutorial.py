# -*- coding: utf-8 -*-
"""
训练一个分类器
=====================
**翻译者**: `Antares博士 <http://www.studyai.com/antares>`_

就是这个!你已经了解了如何定义神经网络、计算损失和更新网络的权重。

现在你可能在想，.......

数据如何?
----------------

通常，当你必须处理图像、文本、音频或视频数据时，可以使用标准python包将数据加载到numpy数组中。
然后你可以把这个array转换成一个 ``torch.*Tensor`` 。

-  对于图像, packages 比如 Pillow, OpenCV 很有用
-  对于音频, packages 比如 scipy 和 librosa
-  对于文本, 或者 raw Python 或者 Cython based 加载, 或 NLTK 和SpaCy 很有用

我们专门为vision创建了一个名为 ``torchvision`` 的包，它为常见数据集(如Imagenet、CIFAR 10、MNIST等)提供了数据加载器。
以及用于图像的数据变换器(data transformers)，即 ``torchvision.datasets`` 和 ``torch.utils.data.DataLoader`` 。

这提供了巨大的方便，避免了编写样板代码。

对于本教程，我们将使用CIFAR 10数据集。它有“飞机”、“汽车”、“鸟”、“猫”、“鹿”、“狗”、“青蛙”、“马”、“船”、“卡车”。
CIFAR-10中的图像大小为3x32x32，即尺寸为32x32像素的3通道彩色图像。

.. figure:: /_static/img/cifar10.png
   :alt: cifar10

   cifar10


训练一个图像分类器
----------------------------

我们将按顺序做一下几步事情:

1. 使用 ``torchvision`` 加载和归一化 CIFAR10 训练集 和 测试集
2. 定义一个 卷积神经网络(Convolutional Neural Network)
3. 定义一个损失函数
4. 在训练集上训练网络
5. 在测试集上测试网络

1. 加载和归一化 CIFAR10
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

使用 ``torchvision``, 加载 CIFAR10 极其简单。
"""
import torch
import torchvision
import torchvision.transforms as transforms

########################################################################
# torchvision datasets 的输出是 PILImage 类型的 images, 像素取值范围在 [0, 1]。
# 我们将其变换为归一化范围在[-1 , 1] 的 Tensors 类型

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

########################################################################
# 让我们显示一些训练集中的图像, 玩玩.

import matplotlib.pyplot as plt
import numpy as np
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


# 用来显示一张图像的函数

def imshow(img):
    img = img / 2 + 0.5     # 去除归一化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# 随机获取一些训练集图片
dataiter = iter(trainloader)
images, labels = dataiter.next()

# 显示图像
imshow(torchvision.utils.make_grid(images))
# 输出对应的标签
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


########################################################################
# 2. 定义一个 卷积神经网络
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 把我们上一节定义的神经网络拷贝过来然后修改，
# 让它接受 3-通道 图像 (我们上节定义的是单通道的图像输入)。

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

########################################################################
# 3. 定义损失函数和优化器
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 让我们使用 分类交叉熵损失(Classification Cross-Entropy loss)和带有动量项的SGD优化器。

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

########################################################################
# 4. 训练网络
# ^^^^^^^^^^^^^^^^^^^^
#
# 这是事情开始变得有趣的时候。我们只需在数据迭代器(data iterator)上循环，并将输入提供给网络并进行优化。

for epoch in range(2):  # 在整个数据集上轮番训练多次，轮训一次叫一个回合(epoch)

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # 获得输入
        inputs, labels = data

        # 将可训练参数的梯度全部置零
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 输出一些关于训练的统计信息
        running_loss += loss.item()
        if i % 2000 == 1999:    # 每 2000 个 mini-batches 输出一次
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

########################################################################
# 5. 在测试数据上测试网络
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# 我们已经在整个训练集上轮番训练了2次网络。但是我们需要检查一下网络有没有学到任何东西。
#
# 我们将通过预测神经网络输出的类标签来验证这一点，并根据实际情况对其进行检查。
# 如果预测是正确的，我们将示例添加到正确的预测列表中。
#
# 好的，第一步，让我们显示来自测试集的图像来熟悉一下

dataiter = iter(testloader)
images, labels = dataiter.next()

# 输出图像和正确的类标签
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

########################################################################
# 好了，现在让我们看看神经网络是怎么想的，上面的例子是:

outputs = net(images)

########################################################################
# 输出是10类的能量。一个类的能量越高，网络就越认为该图像是特定类的。
# 那么，让我们得到最高能量的索引：
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

########################################################################
# 结果看起来相当不错.
#
# 让我们看看网络在整个测试集上的表现吧。

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

########################################################################
# 结果看起来比随机猜测好多啦, 随机猜测只有 10% 的准确率(accuracy) 
# (随机猜测： randomly picking a class out of 10 classes).
# 这说明网络似乎学习到了一些东西。
#
# 那么, 网络在哪些类上表现的较好，而哪些类上表现的较差呢:

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

########################################################################
# 好啦, 那我们接下来干嘛呢？
#
# 我们如何在GPU上训练我们的网络呢？
#
# 在GPU上训练
# ----------------
# 就像我们在前面把一个 Tensor 迁移到 GPU 上去时所作的那样, 
# 你可以用同样的方式把你的神经网络也迁移到 GPU 上。
#
# 如果手头有可用的CUDA,那么我们首先把我们的device定义为第一个可用的cuda device:

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 假定我们在一个 CUDA machine 上跑这个代码, 那么这将输出 CUDA device:

print(device)

########################################################################
# 本节剩余的内容都假定 `device` 是一个 CUDA device.
#
# 然后，这些方法将递归遍历所有模块并将其参数和缓冲区转换为CUDA tensors:
#
# .. code:: python
#
#     net.to(device)
#
#
# 请记住，还要必须在每一步将输入和对应的目标值发送到GPU:
#
# .. code:: python
#
#         inputs, labels = inputs.to(device), labels.to(device)
#
# 与CPU相比，我为什么没有看到大量的加速呢？因为你的网络真的很小。
#
# **练习:** 尝试增加网络的宽度(要注意 第一个 ``nn.Conv2d`` 的参数2和第二个 ``nn.Conv2d`` 的参数1-它们需要相同的数目)，
# 看看你得到了什么样的加速。
#
# **目标顺利达成**:
#
# - 从高层次上理解PyTorch张量库和神经网络。
# - 训练一个小神经网络对图像进行分类
#
# 在多个GPUs上训练模型
# -------------------------
# 如果你想看到更多的大规模加速使用你的所有GPU，请查看  :doc:`data_parallel_tutorial` 。
#
# 下一步向何方去?
# -------------------
#
# -  :doc:`Train neural nets to play video games </intermediate/reinforcement_q_learning>`
# -  `Train a state-of-the-art ResNet network on imagenet`_
# -  `Train a face generator using Generative Adversarial Networks`_
# -  `Train a word-level language model using Recurrent LSTM networks`_
# -  `More examples`_
# -  `More tutorials`_
# -  `Discuss PyTorch on the Forums`_
# -  `Chat with other users on Slack`_
#
# .. _Train a state-of-the-art ResNet network on imagenet: https://github.com/pytorch/examples/tree/master/imagenet
# .. _Train a face generator using Generative Adversarial Networks: https://github.com/pytorch/examples/tree/master/dcgan
# .. _Train a word-level language model using Recurrent LSTM networks: https://github.com/pytorch/examples/tree/master/word_language_model
# .. _More examples: https://github.com/pytorch/examples
# .. _More tutorials: https://github.com/pytorch/tutorials
# .. _Discuss PyTorch on the Forums: https://discuss.pytorch.org/
# .. _Chat with other users on Slack: https://pytorch.slack.com/messages/beginner/

# %%%%%%INVISIBLE_CODE_BLOCK%%%%%%
del dataiter
# %%%%%%INVISIBLE_CODE_BLOCK%%%%%%
