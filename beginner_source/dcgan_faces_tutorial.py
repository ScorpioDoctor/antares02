# -*- coding: utf-8 -*-
"""
DCGAN 教程
==============

**翻译者**: `Antares博士 <http://www.studyai.com/antares>`_

"""


######################################################################
# 介绍
# ------------
# 
# 本教程将通过一个示例介绍DCGANs。我们将训练一个生成对抗网络(generative adversarial network, GAN)，
# 在给它展示许多名流的照片之后，产生新的名人。这里的大部分代码都来自 `pytorch/examples <https://github.com/pytorch/examples>`__ 的实现，
# 本文档将详细解释实现，并阐明该模型是如何工作的和为什么工作的。但别担心，不需要事先知道GANs，
# 但它可能需要第一次花一些时间来推理在表象的下面真正发生了什么。此外，为了时间，有一个或两个GPU可能是个好事儿。
# 让我们从头开始。
# 
# 生成对抗网络
# -------------------------------
# 
# 什么是 GAN?
# ~~~~~~~~~~~~~~
# 
# GANS是一个框架，它教授DL模型以捕获训练数据的分布，这样我们就可以从相同的分布生成新的数据。
# GANs 是由伊恩·古德费罗于2014年发明的，并首次在论文 
# `Generative Adversarial Nets <https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf>`__ 
# 中进行了描述。它们由两种不同的模型组成，一种是生成器(*generator*)，另一种是判别器(*discriminator*)。
# 生成器的工作是生成看起来像训练图像的“假”图像。判别器的工作是查看图像并输出它是真实的训练图像还是来自生成器的假图像。
# 在训练过程中，生成器不断地试图通过生成越来越好的伪图像来胜过判别器，而判别器正在努力成为一名更好的侦探，
# 并正确地对真假图像进行分类。这个游戏的均衡是当生成器生成看起来像是直接来自训练数据的完美假象时，
# 判别器总是以50%的信心猜测生成器输出是真是假的。
# 
# 现在，让我们从判别器开始，在整个教程中定义一些要使用的符号。假设 :math:`x` 是表示图像的数据。 
# :math:`D(x)` 是判别器网络，它输出 :math:`x` 来自训练数据而不是生成器的(标量)概率。这里，
# 由于我们处理的是图像，:math:`D(x)` 的输入是HWC大小为3x64x64的图像。
# 直觉上，当 :math:`x` 来自训练数据时， :math:`D(x)` 应该是高的，
# 当 :math:`x` 来自生成器时，:math:`D(x)`  应该是低的。
# :math:`D(x)` 也可以看作是一种传统的二元分类器。
# 
# 对于生成器的表示法，设 :math:`z` 是从标准正态分布中采样的潜在空间向量(latent space vector)。
# :math:`G(z)` 表示生成函数，它将潜在向量 :math:`z` 映射到数据空间。 
# :math:`G` 的目标是估计训练数据的分布 (:math:`p_{data}`) ，从而从估计出的分布(:math:`p_g`)中生成假样本。
# 
# 因此, :math:`D(G(z))` 是生成器 :math:`G` 输出的图像为真实图像的概率(标量)。
# 正如 `古德费罗的论文 <https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf>`__, 所描述的那样，
# :math:`D` 和 :math:`G` 玩了一个极小极大的博弈(minimax game)，其中 :math:`D` 
# 试图最大化它正确地分类真图像和假图像的概率(:math:`logD(x)`)，:math:`G` 试图最小化 :math:`D` 
# 预测其输出是假的的概率 (:math:`log(1-D(G(x)))`) 。文中给出了GAN损失函数:
# 
# .. math:: \underset{G}{\text{min}} \underset{D}{\text{max}}V(D,G) = \mathbb{E}_{x\sim p_{data}(x)}\big[logD(x)\big] + \mathbb{E}_{z\sim p_{z}(z)}\big[log(1-D(G(x)))\big]
# 
# 理论上，这个极小极大博弈的解是 在 :math:`p_g = p_{data}` 时，判别器只能随机猜测输入是真还是假。
# 然而，GANS的收敛理论仍在积极研究之中，而在现实中，模型并不总是训练到这一点。
# 
# 什么又是 DCGAN?
# ~~~~~~~~~~~~~~~~
# 
# DCGAN是上述GANs的直接扩展，只是它在鉴别器和生成器中分别显式地使用卷积和卷积转置层。
# 它首先由Radford在文章 `Unsupervised Representation Learning With
# Deep Convolutional Generative Adversarial Networks <https://arxiv.org/pdf/1511.06434.pdf>`__ 
# 提出了一种基于深层卷积生成对抗网络的无监督表示学习方法。
# 判别器由跨步卷积层(`strided convolution layers <https://pytorch.org/docs/stable/nn.html#torch.nn.Conv2d>`__ )、
# 批归一化层(`batch norm layers <https://pytorch.org/docs/stable/nn.html#torch.nn.BatchNorm2d>`__)
# 和 `LeakyReLU <https://pytorch.org/docs/stable/nn.html#torch.nn.LeakyReLU>`__ 激活函数构成。
# 输入是3x64x64图像，输出是 输入来自真实数据分布的 标量概率。
# 生成器由卷积转置层(`convolutional-transpose <https://pytorch.org/docs/stable/nn.html#torch.nn.ConvTranspose2d>`__)、
# 批归一化层和 `ReLU <https://pytorch.org/docs/stable/nn.html#relu>`__ 激活层组成。
# 输入是从标准正态分布中提取的潜在矢量(latent vector) :math:`z` ，输出是 3x64x64 的RGB图像。
# 跨步卷积转置层(strided conv-transpose layers)允许将潜在矢量(latent vector)变换为具有与图像相同的shape。
# 作者还就如何设置优化器、如何计算损失函数以及如何初始化模型的权重等方面给出了一些提示，这些都将在后面的章节中加以说明。
# 

from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# Set random seem for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


######################################################################
# 输入
# ------
# 
# 我们先来定义一些输入:
# 
# -  **dataroot** - dataset 文件夹根目录的路径。我们将在下一节中更多地讨论数据集。
# -  **workers** - 用于用 DataLoader 加载数据的工作线程数。
# -  **batch_size** - 训练中使用的批次大小。DCGAN 使用的批次大小为128。
# -  **image_size** - 用于训练的图像的空间大小。此实现默认为64x64。
#    如果需要另一个尺寸，则必须改变D和G的结构。有关更多细节，请参阅 `这里 <https://github.com/pytorch/examples/issues/70>`__ 。
# -  **nc** - 输入图像的颜色通道数. 彩色图像是3通道的。
# -  **nz** - 潜在向量(latent vector)的长度
# -  **ngf** - 与通过生成器进行的特征映射的深度有关。
# -  **ndf** - 设置通过鉴别器传播的特征映射的深度。
# -  **num_epochs** - 要运行的训练回合(epoch)数。长期的训练可能会带来更好的效果，但也需要更长的时间。
# -  **lr** - 用于训练的学习率. 就像在 DCGAN 论文中建议的, 这个参数设为 0.0002 。
# -  **beta1** - Adam 优化器的beta1超参数。 就像在 DCGAN 论文中建议的, 这个参数设为 0.5 。
# -  **ngpu** - 可用的 GPUs 数量。 如果没有GPU, 代码将会在 CPU 模式下运行。 如果有多个GPU,那就可以加速计算了。
# 

# Root directory for dataset
dataroot = "data/celeba"

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 128

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 5

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1


######################################################################
# 数据
# ----
# 
# 在本教程中，我们将使用 `Celeb-A Faces  <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>`__  数据集，
# 该数据集可以在链接的站点上下载，也可以在GoogleDrive中下载。dataset将作为一个名为 *img_align_celeba.zip* 的文件下载。
# 下载完后，创建一个名为 *celeba* 的目录，并将zip文件解压缩到该目录中。
# 然后，将此笔记本的 *dataroot* 输入设置为您刚刚创建的renarba目录。由此产生的目录结构应该是：
# 
# ::
# 
#    /path/to/celeba
#        -> img_align_celeba  
#            -> 188242.jpg
#            -> 173822.jpg
#            -> 284702.jpg
#            -> 537394.jpg
#               ...
# 
# 这是一个重要的步骤，因为我们将使用 ImageFolder 类，它需要在dataset的根文件夹中有子目录。
# 现在，我们可以创建 dataset ，dataloader ，设置设备运行，并最终可视化一些训练数据。
# 

# We can use an image folder dataset the way we have it setup.
# Create the dataset
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Plot some training images
real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))



######################################################################
# 实现
# --------------
# 
# 在设置了输入参数并准备好数据集之后，我们现在可以进入实现了。我们将从wigthts初始化策略开始，
# 然后详细讨论生成器、判别器、损失函数和训练循环。
# 
# 权重初始化
# ~~~~~~~~~~~~~~~~~~~~~
# 
# 从DCGAN的文献中，作者指出所有模型的权重都应从均值=0，stdev=0.2的正态分布中随机初始化。
# 权值函数以初始化模型作为输入，并重新初始化所有卷积、卷积-转置和批处理归一化层，以满足这一标准。
# 该函数在初始化后立即应用于模型。
# 

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


######################################################################
# 生成器(Generator)
# ~~~~~~~~~~~~~~~~~~~~~~
# 
# 生成器 :math:`G` 被设计用于将潜在空间矢量(:math:`z`)映射到数据空间。由于我们的数据是图像，
# 将 :math:`z` 转换为数据空间意味着最终创建一个与训练图像(即3x64x64)相同大小的RGB图像。
# 在实践中，这是通过一系列strided 2d convolutional transpose layers 来实现的，
# 每个层与一个2d batch norm layer和一个relu activation层配对。
# 生成器的输出送入到一个tanh函数，将其输出值压缩在 :math:`[-1,1]` 的范围。
# 值得注意的是batch norm functions是在conv-transpose layers之后的，
# 因为这是DCGAN论文的一个关键贡献。这些层有助于训练期间的梯度流。
# DCGAN文章中给出的生成器的结构如下所示。
#
# .. figure:: /_static/img/dcgan_generator.png
#    :alt: dcgan_generator
#
# 注意，我们在输入部分(*nz*, *ngf*, 和 *nc*) 中设置的输入如何影响代码中的生成器体系结构。
# *nz* 是 z 输入向量的长度， *ngf*  与通过生成器传播的特征图的大小有关，
# *nc* 是输出图像中的通道数(对于RGB图像设置为3)。下面是生成器的代码。
# 

# Generator Code

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


######################################################################
# 现在，我们可以实例化生成器并应用 ``weights_init`` 函数。
# 查看打印的模型，看看生成器对象是如何构造的。
# 

# 创建生成器对象
netG = Generator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# 应用 weights_init 函数 来随机初始化 所有权重到 mean=0, stdev=0.2.
netG.apply(weights_init)

# 打印输出模型
print(netG)


######################################################################
# 判别器(Discriminator)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# 如上所述，判别器 :math:`D` 是一种两类分类网络，它以图像为输入，输出 输入图像为真(而不是假)的标量概率。
# 这里，:math:`D` 接受一个 3x64x64 输入图像，通过一系列Conv2d、BatchNorm2d和LeakyReLU层处理它，
# 并通过 sigmoid 激活函数输出最终的概率。如果有必要的话，可以用更多的层来扩展这个体系结构，
# 但是使用strided convolution、BatchNorm和LeakyReLU是很有意义的。DCGAN的论文提到，
# 使用strided convolution而不是pooling来降采样是一种很好的做法，
# 因为它让网络学习自己的池化函数。此外，batch norm 和leaky relu函数促进了健康的梯度流，
# 这对于 :math:`G` 和 :math:`D` 的学习过程都是至关重要的。
# 

#########################################################################
# Discriminator Code

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


######################################################################
# 现在，和生成器一样，我们可以创建判别器，应用 ``weights_init`` 函数，并打印模型的结构。
# 

# 创建 Discriminator
netD = Discriminator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))
    
# 应用 weights_init 函数，随机初始化所有权重到 mean=0, stdev=0.2.
netD.apply(weights_init)

# 打印输出模型
print(netD)


######################################################################
# 损失函数和优化器
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# 当 :math:`D` 和 :math:`G` 设置好以后, 我们可以指定它们如何通过损失函数和优化器学习。
# 我们将使用二值交叉熵损失(Binary Cross Entropy loss (`BCELoss <https://pytorch.org/docs/stable/nn.html#torch.nn.BCELoss>`__))
# 函数，在 PyTorch 中是如下定义的:
# 
# .. math:: \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad l_n = - \left[ y_n \cdot \log x_n + (1 - y_n) \cdot \log (1 - x_n) \right]
# 
# 注意这个函数提供目标函数中的两个对数组件的计算 (i.e. :math:`log(D(x))` 和 :math:`log(1-D(G(z)))`) 。
# 我们可以使用 :math:`y` 指定 BCE 等式的哪一部分将被计算。 这将在训练过程中完成，稍后会讲到。但是理解我们如何通过
# 改变 :math:`y` 的值(i.e. GT labels) 去选择我们想要计算的损失函数的一部分是非常重要的。
# 
# 接下来，我们将真标签定义为1，假标签定义为0。这些标签将用于计算 :math:`D` 和 :math:`G` 的损失，
# 这也是在原始GAN文章中使用的约定。最后，我们建立了两个分开的优化器，一个用于 :math:`D` ，
# 一个用于 :math:`G` 。正如DCGAN论文所指出的，两者都是Adam优化器，其学习速率为0.0002，Beta1=0.5。
# 为了跟踪生成器的学习过程，我们将从高斯分布(即固定噪声)中生成固定批次的潜在向量(latent vectors)。
# 在训练循环中，我们将周期性地将这个固定的噪声输入到  :math:`G`  中。在迭代过程中，我们将看到图像从噪声中形成。
# 

# 初始化 BCELoss 函数
criterion = nn.BCELoss()

# 创建一批 latent vectors 用于可视化生成器的进度过程
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# 为在训练过程中的真假标签建立约定
real_label = 1
fake_label = 0

# 为 G 和 D 设置 Adam optimizers 
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))


######################################################################
# 训练
# ~~~~~~~~
# 
# 最后，现在我们已经定义了GAN框架的所有部分，我们可以对其进行训练。请注意，
# 训练GANs是一种艺术，因为不正确的超参数设置会导致模式崩溃，
# 而对错误的原因几乎没有解释。在这里，我们将密切遵循古德费罗论文中的算法1，
# 同时遵循在 `ganhacks <https://github.com/soumith/ganhacks>`__ 中显示的一些最佳实践。
# 也就是说，我们将“为真假图像构造不同的小批量”图像，
# 并调整G的目标函数，使 :math:`logD(G(z))` 最大化。训练分为两个主要部分。
# 第1部分更新判别器，第2部分更新生成器。
# 
# **Part 1 - 训练判别器(Discriminator) **
# 
# 回想一下，训练判别器的目标是最大化将给定的输入正确分类为真或假的概率。
# 我们希望“通过提升判别器的随机梯度来更新判别器”。
# 实际上，我们希望最大化 :math:`log(D(x)) + log(1-D(G(z)))` 。
# 由于来自于ganhacks 的separate mini-batch的建议，
# 我们将用两个步骤来实现上述最大化的计算过程。首先从训练集构造一批真实样本，前向通过 :math:`D` ，
# 计算损失(:math:`log(D(x))`) ，然后计算后传梯度。 
# 其次，用当前生成器构造一批假样本，通过 :math:`D` 向前传递该批样本，
# 计算损失 (:math:`log(1-D(G(z)))`) ，并用反向传递累积梯度。
# 现在，有了全真和全假批次样本中积累的梯度，我们再调用判别器的优化器进行一步优化。
# 
# **Part 2 - 训练生成器(Generator) **
# 
# 正如在最初的论文中所述，我们希望通过最小化 :math:`log(1-D(G(z)))` 来训练生成器，以产生更好的假样本。
# 正如前面提到的，Goodfellow没有提供足够的梯度，特别是在学习过程的早期。作为修正，
# 我们希望最大化 :math:`log(D(G(z)))` 。在代码中，我们通过以下方法实现了这一点：
# 用第1部分的判别器对生成器的输出进行分类，使用真标签作为GroundTruth计算G的损失, 
# ，随后在向后传递中计算G的梯度，最后用优化器的 ``step`` 方法更新G的参数。
# 使用真标签作为GT标签用于损失函数的计算似乎有违直觉，但这允许我们使用BCELoss的 :math:`log(x)` 部分
# (而不是 :math:`log(1-x)` 部分)，这正是我们想要的。
# 
# 最后，我们将做一些统计报告，并在每个epoch结束时，我们将把固定批次噪声推到生成器中
# 以可视化地跟踪G的训练进度。所报告的训练统计数字如下：
# 
# -  **Loss_D** - 判别器损失，是所有真批次和所有假批次样本上的损失之和 (:math:`log(D(x)) + log(D(G(z)))`)。
# -  **Loss_G** - 生成器损失，用 :math:`log(D(G(z)))` 计算。
# -  **D(x)** - 所有批次的真样本上判别器的平均输出(跨batch)。这个值应该开始接近1，然后当G变得更好时，理论上收敛到0.5。想想这是为什么。
# -  **D(G(z))** - 所有批次的假样本上判别器的平均输出。这个值应该开始接近0，后面随着生成器越来越好就收敛到0.5。想想这是为什么。
# 
# **Note:** 这一步可能会花点时间, 这取决于你要运行多少个epoch以及如果你从数据集移除一些数据。
# 

# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):
        
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, device=device)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()
        
        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        
        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())
        
        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            
        iters += 1


######################################################################
# 结果
# -------
# 
# 最后，让我们来看看我们是如何做到的。在这里，我们将看到三个不同的结果。
# 首先，我们将看到D和G在训练中的损失是如何变化的。第二，我们将在每个epoch的固定噪声批次上可视化G的输出。
# 第三，我们将看到一批真数据，旁边是一批来自G的假数据。
# 
# **Loss versus training iteration**
# 
# 下面是迭代过程中 D 与 G 的损失对比图。 
# 

plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()


######################################################################
# **G的进度的可视化**
# 
# 记住，在每个训练回合(epoch)之后，我们是如何将generator的输出保存在固定噪声批次上的。
# 现在，我们可以用动画来可视化G的训练进度。按“播放”按钮启动动画。
# 

#%%capture
fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

HTML(ani.to_jshtml())


######################################################################
# **真图像(Real Images) vs. 假图像(Fake Images)**
# 
# 最后, 让我们看看真图像和假图像吧！
# 

# 从 dataloader 中抓取一个批次的真图像
real_batch = next(iter(dataloader))

# Plot the real images
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

# 绘制最后一个epoch的假图像
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()


######################################################################
# 下一步去哪里
# ----------------
# 
# 我们的旅程已经到了尽头，但是有几个地方你可以从这里去。你可以：
# 
# -  训练更长的时间看看得到的结果有多好
# -  修改此模型让其接收不同的数据集 和 可能改变的图像大小与模型架构
# -  检查其他一些很酷的 GAN 项目 `这里 <https://github.com/nashory/gans-awesome-applications>`__ 。
# -  创建一个 GANs 让它产生 `音乐 <https://deepmind.com/blog/wavenet-generative-model-raw-audio/>`__
# 

