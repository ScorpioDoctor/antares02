# -*- coding: utf-8 -*-
"""
在CIFAR100上训练CNN
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
# Cifar100 数据集的加载与预处理
# -------------------------------------------
# 
# CIFAR-100 dataset: 数据集包含100小类，每小类包含600个图像，其中有500个训练图像和100个测试图像。
# 100类被分组为20个大类。每个图像带有1个小类的“fine”标签和1个大类“coarse”标签。
# 数据集中样本的顺序是随机的，某些训练批次内各个类的样本数量不一定严格相同
# 详细信息看官网：http://www.cs.toronto.edu/~kriz/cifar.html
# 

############################################################################
# 定义变换，加载训练集和测试集
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

batch_size = 32

transform_train = transforms.Compose([transforms.Pad(4), transforms.RandomCrop(32), 
                                      transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])

transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR100(root='./data/cifar100', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(root='./data/cifar100', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

# 这样算出来的样本数量不是很严格
print("训练集大小：",len(trainloader)*batch_size)
print("测试集大小：",len(testloader)*batch_size)

############################################################################
# 加载标签名称文件
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

labels_filepath = "./data/cifar100/cifar-100-python/meta"
labels_dict = unpickle(labels_filepath)

print(labels_dict.keys(),'\n')

classes = labels_dict[b'fine_label_names']
superclasses = labels_dict[b'coarse_label_names']

print("100个精细分类的名称：", classes,'\n')
print("20个粗略分类的名称：", superclasses)

############################################################################
# 展示一个批次的图片
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# 

# 用于显示一张图像的函数
def imshow(img, title=None):
    img = img / 2 + 0.5     # 去归一化
    npimg = img.numpy()
    plt.figure(figsize=[6.5, 2.5])
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    if title is not None:
        plt.title(title)
    plt.show()
    plt.pause(0.1)

# 获取一个批次的图像，一次迭代取出batch_size张图片
dataiter = iter(trainloader)
images, labels = dataiter.next()

# 显示一个批次的图像
imshow(torchvision.utils.make_grid(images),"One Batch Train Images")
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


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), # 32
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 16

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 16
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 8
                      
            nn.Conv2d(64, 64, kernel_size=3, padding=1), # 8
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 4
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 100)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
model_urls = {
    'cifar100': 'http://ml.cs.tsinghua.edu.cn/~chenxi/pytorch-models/cifar100-3a55a987.pth',
}

class CIFAR(nn.Module):
    def __init__(self, features, n_channel, num_classes):
        super(CIFAR, self).__init__()
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

def cifar100(n_channel, pretrained=None):
    cfg = [n_channel, n_channel, 'M', 2*n_channel, 2*n_channel, 'M', 4*n_channel, 4*n_channel, 'M', (8*n_channel, 0), 'M']
    layers = make_layers(cfg, batch_norm=True)
    model = CIFAR(layers, n_channel=8*n_channel, num_classes=100)
    if pretrained is not None:
        m = model_zoo.load_url(model_urls['cifar100'])
        state_dict = m.state_dict() if isinstance(m, nn.Module) else m
        assert isinstance(state_dict, (dict, OrderedDict)), type(state_dict)
        model.load_state_dict(state_dict)
    return model

#####################################################
# 将模型写入文件并用TensorBoardX查看
# -----------------------------------
# 

from tensorboardX import SummaryWriter

# 虚拟输入，与 CIFAR10 的一个batch数据的shape相同
dummy_input = torch.autograd.Variable(torch.rand(batch_size, 3, 32, 32))

model = cifar100(n_channel=32, pretrained=None)
#或 model = SimpleNet()

print(model)

with SummaryWriter(comment='_cifar100_net') as w:
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
num_epochs = 50
num_batches = len(trainloader)
log_interval = 200

best_acc, old_file = 0, None
t_begin = time.time()

decreasing_lr=[20, 30]
writer = SummaryWriter(comment='_cifar100_logs')

net.train()

for epoch in range(num_epochs):
    
    if epoch in decreasing_lr:
            optimizer.param_groups[0]['lr'] *= 0.1
            
    running_loss = 0.0        
    for batch_idx, batch_data in enumerate(trainloader):
        n_iter = epoch * num_batches + batch_idx
        images, labels = batch_data[0].to(device), batch_data[1].to(device)
        # 将梯度清零
        optimizer.zero_grad()
        # forward
        out = net(images)
        # 计算损失
        loss_value = loss(out, labels)
        # backward
        loss_value.backward()
        # optimise
        optimizer.step()
        # 将损失和学习率写入日志文件
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
imshow(torchvision.utils.make_grid(images), "One Test Batch Images")
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

outputs = net(images.to(device))
_, predicted = torch.max(outputs, 1)

print('Predicted:   ', ' '.join('%5s' % classes[predicted[j]] for j in range(batch_size)))

