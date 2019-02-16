# -*- coding: utf-8 -*-
"""
迁移学习教程
==========================
**翻译者:** `Antares <http://wwww.studyai.com/antares>`_

在本教程中，您将学习如何使用迁移学习(transfer learning)来训练您的网络。
你可以在 `cs231n 笔记 <http://cs231n.github.io/transfer-learning/>`__ 
上读到更多关于转移学习的内容。

引用一段来自笔记的话,

    在实践中，很少有人从零开始训练整个卷积网络(随机初始化)，因为拥有足够大小的数据集相对较少。
    相反，通常在非常大的数据集上对ConvNet进行预训练(例如ImageNet，其中包含120万幅图像，包含1000个类别)，
    然后使用ConvNet作为初始化或固定特征提取器来执行感兴趣的任务。

两种主要的迁移学习场景如下所示:

-  **微调卷积网络**: 我们用预先训练过的网络(比如在ImageNet 1000数据集上训练的网络)
   来初始化网络，而不是随机初始化。 初始化以后其余的训练看起来像往常一样。
-  **把ConvNet作为固定特征提取器**: 在这里，我们将冻结所有网络的权重，但最终的完全连接层除外。
   最后一个完全连接的层被一个具有随机权重的新层所取代，并且只有这个层被训练。

"""
# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
plt.ion()   # interactive mode

######################################################################
# 加载数据
# ---------
#
# 我们将使用 torchvision 和 torch.utils.data 包来加载数据。
#
# 我们今天要解决的问题是训练一个模型来分类蚂蚁和蜜蜂。
# 我们有大约120张针对蚂蚁和蜜蜂的训练图像。每个类有75个验证图像。
# 通常，如果从零开始训练，这是一个非常小的数据集。
# 由于我们使用迁移学习，我们应该能够合理地泛化。
#
# 该数据集是 imagenet 的一个非常小的子集.
#
# .. Note ::
#    从 `这里 <https://download.pytorch.org/tutorial/hymenoptera_data.zip>`_ 
#    下载数据并把它们解压到当前工作目录
# 
# 数据增广和归一化用于训练；
# 只归一化用于验证
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = './data/hymenoptera_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

######################################################################
# 可视化一些图像
# ^^^^^^^^^^^^^^^^^^^^^^
# 让我们可视化一些训练图像，以了解数据增强(data augmentations)。

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# 获取训练数据的一个批次
inputs, classes = next(iter(dataloaders['train']))

# 把一个批次的图像制作成图像网格
out = torchvision.utils.make_grid(inputs)

plt.figure(figsize=[6.5,2.5])
imshow(out, title=[class_names[x] for x in classes])


######################################################################
# 训练模型
# ------------------
#
# 现在，让我们编写一个通用函数来训练一个模型。在此，我们将说明:
#
# -  调度学习率
# -  保存最优的模型
#
# 在下面的示例中，参数 ``scheduler`` 是 ``torch.optim.lr_scheduler`` 中的LR调度器对象。


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


######################################################################
# 可视化模型的预测
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# 显示若干图像的预测结果的通用函数
#

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

######################################################################
# 微调卷积网络
# ----------------------
#
# 加载一个预训练的模型 并且 重置最终的全连接层。
#

model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# 可以看到 网络的所有可训练参数都被优化啦
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# 每7个回合(epochs)衰减 LR, 衰减因子是 0.1
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

######################################################################
# 训练 和 评估
# ^^^^^^^^^^^^^^^^^^
#
# 在CPU上它应该需要15-25分钟。不过，在GPU上，它所用的时间还不到一分钟。
#

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25)

######################################################################
#

visualize_model(model_ft)


######################################################################
# 把ConvNet作为固定特征提取器
# ----------------------------------
#
# 在这里，我们需要冻结所有的网络，除了最后一层。我们需要设置 ``requires_grad == False``  来冻结参数，
# 这样梯度就不会在 ``backward()`` 中计算。
#
# 你可以在文档中查看更多 `点这里 <https://pytorch.org/docs/notes/autograd.html#excluding-subgraphs-from-backward>`__。
#

model_conv = torchvision.models.resnet18(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False

# 最新添加的模块的参数默认情况下 ``requires_grad=True``` 
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 2)

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# 我们看到 只有最后一层的参数被优化了，这与前面的微调网络是不一样的。
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# 每7个回合(epoch)衰减 LR, 衰减因子是 0.1
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)


######################################################################
# 训练和评估
# ^^^^^^^^^^^^^^^^^^
#
# 在 CPU 上与上面的微调网络相比这将花费大约一半的时间。
# 这是预料之中的，因为对大部分网络其梯度不需要计算。然而，前向传播确实需要计算。
#

model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=25)

######################################################################
#

visualize_model(model_conv)

plt.ioff()
plt.show()
