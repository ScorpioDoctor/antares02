# -*- coding: utf-8 -*-
"""
在STL10数据集上训练CNN
==============================

**作者**: `Antares博士 <http://www.studyai.com/antares>`__

我们将采取以下步骤:

1. 使用 ``torchvision`` 加载和规范训练和测试数据集
2. 定义卷积神经网络模型
3. 将模型写入文件并用TensorBoardX查看
4. 定义损失函数和优化器
5. 在训练数据上训练网络
6. 在测试数据上测试网络

"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

# 忽略 warnings
import warnings
warnings.filterwarnings("ignore")

################################################################################
# STL10 数据集的加载与预处理
# -------------------------------------------
# 
# STL-10数据集是一种用于开发无监督特征学习、深度学习、
# 自学学习算法的图像识别数据集.它受CIFAR-10数据集的启发，
# 但作了一些修改。特别是，每一类有标记的训练样例比CIFAR-10少，
# 但是在监督训练之前提供了一组非常大的未标记样例来学习图像模型。
# 主要的挑战是利用未标记的数据(来自与标记数据相似但不同的分布)来构建有用的先验数据。
# 我们还期望这个数据集(96x96)的更高分辨率将使它成为开发更
# 可扩展的无监督学习方法的具有挑战性的基准。
# 详细信息看官网：https://cs.stanford.edu/~acoates/stl10/
# 
# 数据集总体情况：
# 
# - 1、10 个类: airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck.
# - 2、图像是彩色的96x96像素的
# - 3、训练图像500张(10张预定义)，每个类800张测试图像
# - 4、100000张无标记图像用于无监督学习。这些例子是从相似但更广泛的图像分布中提取出来的。
#   例如，除了标记集中的那些动物之外，它包含其他类型的动物(熊、兔子等)，及车辆(火车、巴士等)。
# - 5、图像是从ImageNet上标记的示例中获取的。
# 

############################################################################
# 定义变换，加载训练集和测试集
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

batch_size = 32


transform_train = transforms.Compose([transforms.Pad(4), transforms.RandomCrop(96), 
                                      transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])

transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.STL10(root='./data/stl10', split='train', download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.STL10(root='./data/stl10', split='test', download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

classes = ('airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck')

# 这样算出来的样本数量不是很严格
print("训练集大小：",len(trainloader)*batch_size)
print("测试集大小：",len(testloader)*batch_size)

############################################################################
# 展示一个批次的图片
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# 

# 用于显示一张图像的函数
def imshow(img):
    img = img / 2 + 0.5     # 去归一化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# 获取一个批次的图像，一次迭代取出batch_size张图片
dataiter = iter(trainloader)
images, labels = dataiter.next()

# 显示一个批次的图像
imshow(torchvision.utils.make_grid(images))
# 输出 对应批次图像的标签
print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

##################################################################################
# 定义CNN网络模型
# ---------------------
# 

import os
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F

model_urls = {
    'svhn': 'http://ml.cs.tsinghua.edu.cn/~chenxi/pytorch-models/svhn-f564f3d8.pth',
}

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), # 96
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 48

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 48
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 24
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1), # 24
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 12
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1), # 12
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 6
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64 * 6 * 6, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    
model_urls = {
    'stl10': 'http://ml.cs.tsinghua.edu.cn/~chenxi/pytorch-models/stl10-866321e9.pth',
}

class STL10(nn.Module):
    def __init__(self, features, n_channel, num_classes):
        super(STL10, self).__init__()
        assert isinstance(features, nn.Sequential), type(features)
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(n_channel, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for i, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            padding = v[1] if isinstance(v, tuple) else 1
            out_channels = v[0] if isinstance(v, tuple) else v
            conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(out_channels, affine=False), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = out_channels
    return nn.Sequential(*layers)

def stl10(n_channel, pretrained=None):
    cfg = [
        n_channel, 'M',
        2*n_channel, 'M',
        4*n_channel, 'M',
        4*n_channel, 'M',
        (8*n_channel, 0), (8*n_channel, 0), 'M'
    ]
    layers = make_layers(cfg, batch_norm=True)
    model = STL10(layers, n_channel=8*n_channel, num_classes=10)
    if pretrained is not None:
        m = model_zoo.load_url(model_urls['stl10'])
        state_dict = m.state_dict() if isinstance(m, nn.Module) else m
        assert isinstance(state_dict, (dict, OrderedDict)), type(state_dict)
        model.load_state_dict(state_dict)
    return model


#####################################################
# 将模型写入文件并用TensorBoardX查看
# -----------------------------------
# 

from tensorboardX import SummaryWriter

# 虚拟输入，与 STL10 的一个batch数据的shape相同
dummy_input = torch.autograd.Variable(torch.rand(batch_size, 3, 96, 96))

model = stl10(n_channel=32, pretrained=None)
# model = SimpleNet()
print(model)
with SummaryWriter(comment='_stl10_net1') as w:
    w.add_graph(model, (dummy_input, ))


############################################################################
# 把模型迁移到GPU上
#----------------------------------------
# 
seed = 117
cuda = torch.cuda.is_available()
torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

net = model.to(device) 

############################################################################
# 定义损失函数和优化器
#----------------------------------------
# 

import torch.optim as optim

loss = nn.CrossEntropyLoss()
# loss = F.cross_entropy

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.001)

#########################################################################
# 计算初始网络的准确率
# --------------------------
# 

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

############################################################################
# 开始训练
# ----------------
#
 
import time
num_epochs = 30
num_batches = len(trainloader)
log_interval = 20

best_acc, old_file = 0, None
t_begin = time.time()

decreasing_lr=[10, 20]
writer = SummaryWriter(comment='_stl10_logs')

net.train()

for epoch in range(num_epochs):
    
    if epoch in decreasing_lr:
            optimizer.param_groups[0]['lr'] *= 0.5
            
    running_loss = 0.0        
    for batch_idx, batch_data in enumerate(trainloader):
        n_iter = epoch * num_batches + batch_idx
        images, labels = batch_data[0].to(device), batch_data[1].to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        out = net(images)
        # 计算损失
        loss_value = loss(out, labels)
        # backward
        loss_value.backward()
        # optimise
        optimizer.step()
        # LOGGING
        writer.add_scalar('loss', loss_value.item(), n_iter)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], n_iter)

        running_loss += loss_value.item()
        if batch_idx % log_interval == 0 and batch_idx > 0:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / log_interval))
            running_loss = 0.0

    elapse_time = time.time() - t_begin
    speed_epoch = elapse_time / (epoch + 1)
    speed_batch = speed_epoch / num_batches
    eta = speed_epoch * num_epochs - elapse_time
    print("Elapsed {:.2f}s, {:.2f} s/epoch, {:.2f} s/batch, ets {:.2f}s".format(elapse_time, speed_epoch, speed_batch, eta))
        
print("Total Elapse: {:.2f}, Best Result: {:.3f}%".format(time.time()-t_begin, best_acc))
writer.close()
print('Finished Training')

#############################################################################
# 开始测试
# ----------------
# 
# 在整个测试集上的准确率
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

net.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

#################################################################################
# 统计每个类上的准确率
# ~~~~~~~~~~~~~~~~~~~~~~~
# 

net.eval()
num_classes = len(classes)
class_correct = list(0. for i in range(num_classes))
class_total = list(0. for i in range(num_classes))
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(labels.size()[0]):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(num_classes):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

################################################################################
# 展示测试样本，真实标签和预测标签
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

outputs = net(images.to(device))
_, predicted = torch.max(outputs, 1)

print('Predicted:   ', ' '.join('%5s' % classes[predicted[j]] for j in range(batch_size)))

