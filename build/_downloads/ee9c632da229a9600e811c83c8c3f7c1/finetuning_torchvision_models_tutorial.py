"""
微调 Torchvision 模型
=============================

**翻译者**: `Antares博士 <http://www.studyai.com/antares>`__

"""


######################################################################
# 在本教程中，我们将更深入地了解如何 微调预训练的模型 和 把预训练的模型作为特征提取器。
# 我们要使用的预训练模型是 `torchvision models <https://pytorch.org/docs/stable/torchvision/models.html>`__,
# 所有这些models都已在1000类Imagenet数据集上进行了预训练。
# 本教程将给出一个深入的研究如何使用几种现代CNN架构，并帮你建立一个微调任何PyTorch模型的直觉。
# 因为每个模型架构是不同的，所以没有在所有场景中都能工作的样板化代码。
# 相反，研究人员必须查看现有的体系结构，并对每个模型进行自定义调整。
# 
# 在本文档中，我们将执行两种类型的迁移学习：微调(finetuning)和特征提取(feature extraction)。
# 在 微调 过程中，我们从一个预先训练的模型开始，并为我们的新任务 **更新模型的所有参数** ，
# 本质上是对整个模型进行再训练。在 特征提取 中，我们从预先训练的模型开始， **只更新最后一层权值** ，从而得到预测。
# 它被称为特征提取，因为我们使用预先训练的CNN作为固定的特征提取器，并且只改变输出层。
# 有关迁移学习的更多技术信息，请参见 `这里 <http://cs231n.github.io/transfer-learning/>`__ 和 
# `这里 <http://ruder.io/transfer-learning/>`__ 。
# 
# 一般来说，这两种迁移学习方法遵循相同的几个步骤:
# 
# -  初始化预训练模型。
# -  重新塑造(Reshape)最终层，使其具有与新数据集中的类数相同的输出数。
# -  为优化算法指定在训练期间要更新哪些参数。
# -  运行训练步骤
# 

from __future__ import print_function 
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)


######################################################################
# 输入
# ------
# 
# 以下是运行时要更改的所有参数。我们将使用 *hymenoptera_data* ，可以在 
# `这里 <https://download.pytorch.org/tutorial/hymenoptera_data.zip>`__ 下载。
# 此数据集包含两个类(蜜蜂和蚂蚁)，其结构使我们可以使用 
# `ImageFolder <https://pytorch.org/docs/stable/torchvision/datasets.html#torchvision.datasets.ImageFolder>`__ 数据集，
# 而不是编写我们自己的自定义数据集。下载数据并将 ``data_dir`` 输入设置为dataset的根目录。 
# ``model_name`` 输入是您希望使用的模型的名称，必须从此列表中选择:
# 
# ::
# 
#    [resnet, alexnet, vgg, squeezenet, densenet, inception]
# 
# 其他输入如下：``num_classes`` 是数据集中的类数， ``batch_size`` 是用于训练的批大小，
# 可以根据机器的能力进行调整， ``num_epochs`` 是我们想要运行的训练期的数目，
# 而 ``feature_extract`` 是一个布尔值，它定义了我们是在 微调 还是 提取特征。
# 如果 ``feature_extract = False`` ，则对模型进行 微调，并更新所有模型参数。
# 如果 ``feature_extract = True`` ，则只更新最后一个层参数，其他参数则保持不变。
# 

# 顶层数据目录。在这里，我们假设目录的格式符合 ImageFolder 结构 
data_dir = "./data/hymenoptera_data"

# 可供选择的模型 [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "resnet"

# 数据集中的类数
num_classes = 2

# 用于训练的批处理大小(根据您的内存大小更改) 
batch_size = 8

# 要训练的回合(epochs)数
num_epochs = 15

# 用于特征提取的标志。当其取值为False时我们微调整个模型，当其取值为True时只更新被重塑的层参数
feature_extract = True


######################################################################
# 辅助函数
# ----------------
# 
# 在编写调整模型的代码之前，让我们定义一些辅助函数。
# 
# 模型训练和验证的代码
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# ``train_model`` 函数处理给定模型的训练和验证。作为输入，它需要一个PyTorch模型、
# 一个dataloader的字典、一个损失函数、一个优化器、需要训练和验证的指定数量的回合(epoches)，
# 以及一个布尔标志指示是否将 Inception model 作为初始模型。 *is_inception* 
# 标志用于容纳 *Inception v3* 模型，因为该体系结构使用辅助输出，而总体模型损失既包括辅助输出的损失，
# 也包括最终输出的损失，如下所述。该函数对指定数量的回合数(epoches)进行训练，并在每个epoch之后
# 运行一个完整的验证步骤。它还跟踪性能最好的模型(在验证精度方面)，并在训练结束时返回性能最好的模型。
# 每个epoch之后，训练和验证的准确率被打印出来。
# 

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 每个回合(epoch)都有一个训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # 在数据上迭代.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 把 参数梯度 置零
                optimizer.zero_grad()

                # 前向过程
                # 只有在 train 模式下才会跟踪历史
                with torch.set_grad_enabled(phase == 'train'):
                    # 获得模型输出并计算损失
                    # inception 模型比较特别，因为它在训练阶段它还有一个辅助输出。
                    #   在训练模式下，我们通过对最终输出和辅助输出求和计算损失
                    #   但是在最终模式下，我们只考虑最终输出 
                    if is_inception and phase == 'train':
                        # 来自于 https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # 只有在训练阶段才有 backward + optimize
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计准确率信息
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # 深度拷贝模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


######################################################################
# 设置模型参数的 .requires_grad 属性
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# 当我们提取特征时，这个辅助函数将模型中参数的 ``.requires_grad`` 属性设置为false。
# 默认情况下，当我们加载一个预先训练过的模型时，所有参数都有 ``.requires_grad=True`` ，
# 如果我们从零开始训练或微调模型的话，这是很好的。但是，如果我们要进行特征提取，
# 并且只想为新初始化的层计算梯度，那么我们希望所有其他参数都不需要梯度。
# 稍后这就更有意义了。
# 

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


######################################################################
# 初始化和重塑网络
# -----------------------------------
# 
# 现在是最有趣的部分。这里是我们把每个网络进行重塑的地方。请注意，这不是一个自动的过程，
# 对每个模型来说其重塑(reshaping)是独特的，模型相关的。
# 回想一下，CNN模型的最后一层(通常是FC层)的节点数与数据集中的输出类数相同。由于所有模型都在Imagenet上进行了预训练，
# 它们都有大小为1000的输出层，每个类都有一个节点。这里的目标是重塑(reshape)最后一个层，使其具有与以前相同的输入数，
# 并具有与dataset中的类数相同的输出数。在下面的部分中，我们将讨论如何单独更改每个模型的体系结构。
# 但是首先，对于微调(finetuning)和特征提取(feature-extraction)之间的区别有一个重要的细节.
# 
# 在特征提取时，我们只想更新最后一层的参数，换句话说，我们只想更新我们正在重塑的层的参数。
# 因此，我们不需要计算我们不改变的参数的梯度，因此为了提高效率，我们将 ``.requires_grad``` 属性设置为false。
# 这很重要，因为默认情况下，此属性设置为True。然后，当我们初始化新层时，默认情况下，新参数有 ``.requires_grad=True`` ，
# 因此只有新层的参数才会被更新。当我们微调模型的时候，我们可以将所有 ``.required_grad``` 的设置保留为True的默认值。
# 
# 最后, 请注意 inception_v3 需要的输入尺寸为 (299,299), 而其他模型的期望输入尺寸为 (224,224)。
# 
# Resnet
# ~~~~~~
# 
# Resnet 出自这篇文章 `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__. 
# 有各种不同大小的变体版本, 包括 Resnet18, Resnet34, Resnet50, Resnet101, 和 Resnet152,
# 所有这些都可以从 torchvision models 中获得。 这里我们使用 Resnet18, 因为我们的数据集是只有两个类的小数据集。
# 当我们输出模型的时候，我们可以看到最后一层是一个全连接层，如下所示:
# 
# ::
# 
#    (fc): Linear(in_features=512, out_features=1000, bias=True) 
# 
# 因此，我们必须将 ``model.fc`` 重新初始化为一个具有512个输入特征和2个输出特征的线性层，如下所示:
# 
# ::
# 
#    model.fc = nn.Linear(512, num_classes)
# 
# Alexnet
# ~~~~~~~
# 
# Alexnet 出自这篇文章 `ImageNet Classification with Deep Convolutional Neural Networks 
# <https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf>`__
# 而且是第一个在ImageNet数据集上非常成功的CNN。 当我们输出模型架构的时候，我们看到来自分类器第6层的模型输出，如下：
# 
# ::
# 
#    (classifier): Sequential(
#        ...
#        (6): Linear(in_features=4096, out_features=1000, bias=True)
#     ) 
# 
# 为了在我们的数据集上使用这个模型，我们把模型输出层重新初始化为:
# 
# ::
# 
#    model.classifier[6] = nn.Linear(4096,num_classes)
# 
# VGG
# ~~~
# 
# VGG 出自文章 `Very Deep Convolutional Networks for Large-Scale Image Recognition <https://arxiv.org/pdf/1409.1556.pdf>`__.
# Torchvision 提供了 VGG 的8个不同版本，有些具有不同的长度，有些具有 batch normalizations layers。
# 这里我们使用带有batch normalization 的 VGG-11 。 VGG的输出层类似于 Alexnet, i.e.
# 
# ::
# 
#    (classifier): Sequential(
#        ...
#        (6): Linear(in_features=4096, out_features=1000, bias=True)
#     )
# 
# 因此, 我们使用相同的办法修改最后一层：
# 
# ::
# 
#    model.classifier[6] = nn.Linear(4096,num_classes)
# 
# Squeezenet
# ~~~~~~~~~~
# 
# Squeeznet 出自这篇文章 `SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model
# size <https://arxiv.org/abs/1602.07360>`__  。  它使用的输出结构与我们上面介绍的那些模型都不一样。
# Torchvision 里面有两个版本的 Squeezenet，我们使用 1.0 版。 它的输出来自一个 1x1 卷积层(分类器的第一个层):
# 
# ::
# 
#    (classifier): Sequential(
#        (0): Dropout(p=0.5)
#        (1): Conv2d(512, 1000, kernel_size=(1, 1), stride=(1, 1))
#        (2): ReLU(inplace)
#        (3): AvgPool2d(kernel_size=13, stride=1, padding=0)
#     ) 
# 
# 为了修改这个网络, 我们重新初始化 Conv2d layer 来获得深度为2的输出特征图：
# 
# ::
# 
#    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
# 
# Densenet
# ~~~~~~~~
# 
# Densenet 出自这篇文章 `Densely Connected Convolutional Networks <https://arxiv.org/abs/1608.06993>`__ 。 
# Torchvision 有四种 Densenet 的版本，但我们只使用 Densenet-121 。 输出层是一个具有1024个输入特征的线性层:
# 
# ::
# 
#    (classifier): Linear(in_features=1024, out_features=1000, bias=True) 
# 
# 为了重塑这个网络, 我们重新初始化分类器的线性层:
# 
# ::
# 
#    model.classifier = nn.Linear(1024, num_classes)
# 
# Inception v3
# ~~~~~~~~~~~~
# 
# Inception v3 出自这篇文章 `Rethinking the Inception Architecture for Computer
# Vision <https://arxiv.org/pdf/1512.00567v1.pdf>`__ 。 
# 这个网络是独特的，因为它在训练时有两个输出层。第二个输出称为辅助输出，包含在网络的 AuxLogits 部分。
# 主要输出是网络末端的线性层。注意，在测试时，我们只考虑主要输出。该模型的辅助输出和主输出打印如下：
# 
# ::
# 
#    (AuxLogits): InceptionAux(
#        ...
#        (fc): Linear(in_features=768, out_features=1000, bias=True)
#     )
#     ...
#    (fc): Linear(in_features=2048, out_features=1000, bias=True)
# 
# 为了微调这个模型，我们必须重新塑造两个layers。这是通过以下方法完成的:
# 
# ::
# 
#    model.AuxLogits.fc = nn.Linear(768, num_classes)
#    model.fc = nn.Linear(2048, num_classes)
# 
# 请注意，许多模型具有相似的输出结构，但每个模型的处理方式必须略有不同。
# 此外，请检查已重塑的网络的打印模型体系结构，并确保输出特征的数量与数据集中的类数相同。
# 

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes) 
        input_size = 224

    elif model_name == "inception":
        """ Inception v3 
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()
    
    return model_ft, input_size

# 运行下面的代码初始化模型
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

# 打印出我们刚刚初始化的模型
print(model_ft) 


######################################################################
# 加载数据
# ---------
# 
# 现在我们知道了输入大小必须是什么，我们可以初始化数据转换器、图像数据集类和数据加载器。
# 注意，模型是用硬编码的归一化值预先训练的，请看 `这里 <https://pytorch.org/docs/master/torchvision/models.html>`__ 。
# 

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

print("Initializing Datasets and Dataloaders...")

# 创建训练和验证数据集
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
# 创建训练和验证加载器
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

# 检测是否有 GPU 可以用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


######################################################################
# 创建优化器(Optimizer) 
# -----------------------
# 
# 既然模型结构是正确的，微调和特征提取的最后一步就是创建一个 **只更新所需参数** 的优化器。
# 回想一下，在加载经过预先训练的模型之后，但是在重新构建之前，如果 ``feature_extract=True`` ，
# 我们会手动将所有参数的 ``.requires_grad`` 属性设置为false。然后，
# 重新初始化层的参数在默认情况下有 ``.requires_grad=True`` 。
# 因此，现在我们知道， *所有具有 ``.requires_grad=True`` 的参数都应该进行优化* 。
# 接下来，我们列出了这些参数， 并将这个列表输入到SGD算法构造函数中。
# 
# 要验证这一点，请查看打印出的参数来学习。当finetuning时，这个列表应该很长，包括所有的模型参数。
# 然而，当特征提取时，这个列表应该是短的，并且只包含被重塑层的权重和偏置。
# 

# 把模型发送到 GPU
model_ft = model_ft.to(device)

# 在此运行中收集要优化/更新的参数。如果我们正在进行 finetuning，我们将更新所有参数。
# 但是，如果我们是在做特征提取方法，我们将只更新我们刚刚初始化的参数 ，
# 也就是 requires_grad == True 的那些参数。
params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

# 观察到 all parameters are being optimized
optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)


######################################################################
# 运行训练和验证步骤
# --------------------------------
# 
# 最后，最后一步是建立模型的损失，然后运行训练和验证函数为设定的回合(epochs)数。
# 注意，这一步在CPU上可能需要一段时间，取决于时间的长短。
# 另外，对于所有的模型，默认的学习率并不是最优的，
# 因此为了达到最大的精度，需要对每个模型分别进行调整。
# 

# 设置损失函数
criterion = nn.CrossEntropyLoss()

# 训练 和 评估
model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))


######################################################################
# 与从零开始训练的模型进行比较
# ------------------------------------------
# 
# 只是为了好玩，让我们看看模型如何学习，如果我们不使用迁移学习。finetuning vs. feature extracting 
# 的性能在很大程度上取决于数据集，但与从头开始训练的模型相比，这两种迁移学习方法在训练时间和总体精度方面
# 都取得了良好的效果。
# 

# 初始化没有预训练的模型，玩玩
scratch_model,_ = initialize_model(model_name, num_classes, feature_extract=False, use_pretrained=False)
scratch_model = scratch_model.to(device)
scratch_optimizer = optim.SGD(scratch_model.parameters(), lr=0.001, momentum=0.9)
scratch_criterion = nn.CrossEntropyLoss()
_,scratch_hist = train_model(scratch_model, dataloaders_dict, scratch_criterion, scratch_optimizer, num_epochs=num_epochs, is_inception=(model_name=="inception"))

# 为迁移学习方法绘制验证准确率与训练时间(epoch)的训练曲线，并与从零开始训练模型得到的曲线比较。
ohist = []
shist = []

ohist = [h.cpu().numpy() for h in hist]
shist = [h.cpu().numpy() for h in scratch_hist]

plt.title("Validation Accuracy vs. Number of Training Epochs")
plt.xlabel("Training Epochs")
plt.ylabel("Validation Accuracy")
plt.plot(range(1,num_epochs+1),ohist,label="Pretrained")
plt.plot(range(1,num_epochs+1),shist,label="Scratch")
plt.ylim((0,1.))
plt.xticks(np.arange(1, num_epochs+1, 1.0))
plt.legend()
plt.show()


######################################################################
# 最后的想法和下一步该去哪里
# -----------------------------------
# 
# 试着运行一些其他的模型，看看有多好的准确性。另外，请注意，特征提取所花费的时间较少，
# 因为在向后传递中，我们不必计算大部分梯度。从这里有很多地方可去。你可以：
# 
# -  使用更难的数据集运行此代码，并看到迁移学习的更多好处。
# -  使用这里描述的方法，使用 transfer learning  来更新不同的模型，可能是在一个新的领域(例如NLP、音频等)。
# -  一旦您对模型感到满意，就可以将其导出为ONNX模型，或者使用混合前端跟踪它，以获得更多的速度和优化机会。
# 

