"""
使用 PyTorch 进行神经风格迁移
=============================

**翻译者**: `Antares博士 <http://www.studyai.com/antares>`_


介绍
------------

本教程介绍了如何实现由Leon A.Gatys开发的 `Neural-Style algorithm <https://arxiv.org/abs/1508.06576>`__ 。
Neural-Style, 或 Neural-Transfer, 允许你对一幅图像采取一种新的艺术风格的形象和再现。
该算法接受输入图像(input image)、 内容图像(content-image)和风格图像(style-image)三种图像，并对输入进行修改，
使之与内容图像的内容和风格图像的艺术风格相似。
 
.. figure:: /_static/img/neural-style/neuralstyle.png
   :alt: content1
"""

######################################################################
# 底层原理
# --------------------
# 
# 原理很简单：我们定义了两个距离，一个用于内容(:math:`D_C`) ，一个用于样式(:math:`D_S`)。
# :math:`D_C` 测量两个图像之间的内容有多不同，而 :math:`D_S` 测量两个图像之间的风格有多不同。
# 然后，我们接受第三个图像作为输入，并转换它，以最小化它与内容图像的内容距离和
# 与样式图像的风格距离。现在我们可以导入必要的包并开始 neural transfer。
# 
# 导入包和选择设备
# -----------------------------------------
# 下面所列出的包都是实现 neural transfer 时所用到的包。
#
# -  ``torch``, ``torch.nn``, ``numpy`` (用PyTorch神经网络不可缺少的软件包)
# -  ``torch.optim`` (高效的梯度下降算法优化包)
# -  ``PIL``, ``PIL.Image``, ``matplotlib.pyplot`` (加载和展示图像的包)
# -  ``torchvision.transforms`` (把 PIL 图像转换为tensors)
# -  ``torchvision.models`` (训练 和 加载 预训练的模型)
# -  ``copy`` (深度拷贝模型; system package)

from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy


######################################################################
# 接下来，我们需要选择在哪个设备上运行网络，并导入内容和样式图像。
# 在大型图像上运行neural transfer算法需要花费更长的时间，并且在GPU上运行的速度要快得多。
# 我们可以使用 ``torch.cuda.is_available()`` 来检测是否有可用的GPU。
# 接下来，我们将 ``torch.device`` 设置为在整个教程中使用。
# 此外，``.to(device)`` 方法用于将张量或模块移动到所需的设备。

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

######################################################################
# 加载图像
# ------------------
#
# 现在我们将导入样式图像和内容图像。原始PIL图像的值介于0到255之间，但是当转换为torch tensors时，
# 它们的值被转换为0和1之间。图像也需要调整大小以具有相同的尺寸。需要注意的一个重要细节是，
# torch library中的神经网络的张量值从0到1变化。如果您试图向网络提供取值为0到255的张量图像，
# 那么激活的特征映射将无法感觉到预期的内容和样式。 然而，来自Caffe库的预训练网络被训练成0到255的张量图像.
#
# .. Note::
#     这里是本教程用到的两张图片的下载地址:
#     `picasso.jpg <https://pytorch.org/tutorials/_static/img/neural-style/picasso.jpg>`__ 和
#     `dancing.jpg <https://pytorch.org/tutorials/_static/img/neural-style/dancing.jpg>`__.
#     下载这两张图片然后将它们放到你当前工作目录中名称为 ``images`` 的文件夹中。

# 输出图像的期望尺寸
imsize = 512 if torch.cuda.is_available() else 128  # 如果没有GPU的话，就把尺寸搞小点儿

loader = transforms.Compose([
    transforms.Resize(imsize),  # 缩放导入的图像
    transforms.ToTensor()])  # 把它转换成 torch tensor


def image_loader(image_name):
    image = Image.open(image_name)
    # 虚拟的 batch 维 ，为了满足网络输入对纬度的要求
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


style_img = image_loader("./data/images/neural-style/picasso.jpg")
content_img = image_loader("./data/images/neural-style/dancing.jpg")

assert style_img.size() == content_img.size(), \
    "we need to import style and content images of the same size"


######################################################################
# 现在，让我们通过将图像的副本转换为PIL格式并使用 ``plt.imshow`` 显示副本
# 来创建一个显示图像的函数。我们将尝试显示内容图像和样式图像，
# 以确保它们被正确导入。

unloader = transforms.ToPILImage()  # 再次转换为 PIL image

plt.ion()

def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # pause a bit so that plots are updated


plt.figure()
imshow(style_img, title='Style Image')

plt.figure()
imshow(content_img, title='Content Image')

######################################################################
# 损失函数
# --------------
# 内容损失(Content Loss)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# 内容损失是一个函数，它表示了一个单独层的加权内容距离。该函数接收处理输入 :math:`X` 的网络的
# 层 :math:`L` 的特征图 :math:`F_{XL}` ，返回输入图像 :math:`X` 和 内容图像 :math:`C`之间的
# 加权内容距离 :math:`w_{CL}.D_C^L(X,C)` 。 内容图像的特征图(:math:`F_{CL}`)必须已知以便能够计算
# 内容距离。我们将这个函数实现为一个 torch module ，它有一个构造器 接受 :math:`F_{CL}` 作为输入。
# 该距离 :math:`\|F_{XL} - F_{CL}\|^2` 是两个特征图集合之间的平均平方误差，可以使用 ``nn.MSELoss`` 
# 来计算。
# 
# 我们将把这个内容损失module直接加到计算内容距离的卷积层后面。在这种方式下，每次网络接到一张输入图像，
# 内容损失将会在需要的层被计算出来，并且因为 auto grad, 所有的梯度将会被计算。
# 现在, 为了使得内容损失层变得透明，我们需要定义一个 ``forward`` 方法 来计算内容损失然后返回该层的输入。
# 计算出的损失被保存为module的参数。
# 

class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

######################################################################
# .. Note::
#    **重要细节**: 虽然这个模块名为 ``ContentLoss``，但它不是一个真正的PyTorch损失函数。
#    如果要将内容损失定义为PyTorch损失函数，则必须创建PyTorch自动梯度函数，以便在 
#    ``backward`` 方法中手动计算/实现梯度。

######################################################################
# 风格损失(Style Loss)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# The style loss module 的实现与 content loss module 的实现类似。
# 它在网络中作为一个透明的层去计算该层的风格损失，我们需要计算 gram 矩阵 :math:`G_{XL}` 。
# gram 矩阵是给定矩阵和该矩阵的转置相乘的结果。在这个应用中，给定的矩阵是
# 层 :math:`L` 的特征图 :math:`F_{XL}` 的 reshaped 版本。
# :math:`F_{XL}` 被 reshape 来形成 :math:`\hat{F}_{XL}`, 一个 :math:`K`\ x\ :math:`N`
# 矩阵, 其中 :math:`K` 是层 :math:`L` 的特征图的数量，而 :math:`N` 是任意向量化的特征图 
# :math:`F_{XL}^k` 的长度。比如说，:math:`\hat{F}_{XL}` 的第一行对应于第一个
# 向量化的特征图 :math:`F_{XL}^1` 。
# 
# 最终, gram 矩阵必须通过将每个元素除以矩阵中的元素总数来标准化。这种归一化是为了抵消
# 具有大 :math:`N` 维数的 :math:`\hat{F}_{XL}` 矩阵在gram矩阵中产生较大值的事实。
# 这种特别大的值将会引起前面的层(在池化层之前的层)在梯度下降过程中施加重要的影响。
# Style features 倾向于在网络中更深的层，所以这个归一化步骤极其重要。
# 

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


######################################################################
# 现在，style loss module 看起来与content loss module 几乎完全一样。
# 使用 :math:`G_{XL}` 和 :math:`G_{SL}` 之间的均方误差计算style distance。
# 

class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


######################################################################
# 导入模型
# -------------------
# 
# 现在我们需要引进一个预先训练过的神经网络。我们将使用19层VGG网络，就像论文中使用的那样。
# 
# PyTorch实现的VGG是一个模块(module)，分为两个子 ``Sequential`` 模块：
# ``features`` (包含卷积层和池化层)和 ``classifier`` (包含完全连接的层)。
# 我们将使用``features`` module，因为我们需要 个别卷积层的输出 以测量内容损失和风格损失。
# 有些层在训练过程中的行为与评估不同，因此我们必须使用 ``.eval()`` 将网络设置为评估模式。
# 


cnn = models.vgg19(pretrained=True).features.to(device).eval()



######################################################################
# 另外, VGG 网络 是在每个通道被均值为 mean=[0.485, 0.456, 0.406] 和 
# std=[0.229, 0.224, 0.225] 所规范化的图像上训练的。
# 我们将使用它们归一化图像，然后把归一化图像送给网络处理。
# 

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# 创建一个module去归一化输入图像，以便我们可以简单滴将它们送给 nn.Sequential 。
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


######################################################################
# ``Sequential`` module 包含一个由child modules构成的有序的list。
# 比如, ``vgg19.features`` 包含一个序列 (Conv2d, ReLU, MaxPool2d,
# Conv2d, ReLU…) aligned in the right order of depth. 
# 我们需要在他们检测到的卷积层之后立即添加内容损失层和风格损失层。
# 为此，我们必须创建一个内容损失模块和风格损失模块被正确插入的
# 新 ``Sequential`` 模块。
# 

# 计算 style/content losses 所需要的深度的层:
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses


######################################################################
# 接下来我们选择输入图像。 你可以使用内容图像的副本或者一副白噪声图像 作为输入图像。
# 

input_img = content_img.clone()
# 如果你想使用白噪声，就去掉下面这行代码的注释:
# input_img = torch.randn(content_img.data.size(), device=device)

# 把原始输入图像加入到 figure 中:
plt.figure()
imshow(input_img, title='Input Image')


######################################################################
# 梯度下降
# ----------------
# 
# 该算法的作者 `建议 <https://discuss.pytorch.org/t/pytorch-tutorial-for-neural-transfert-of-artistic-style/336/20?u=alexis-jacq>`__, 
# 我们使用 L-BFGS 算法来运行梯度下降。不像训练一个网络，我们想要训练的是输入图像以便最小化 content/style losses。
# 我们将创建一个 PyTorch L-BFGS 优化器 ``optim.LBFGS`` 并把我们的图像传递给它作为要被优化的张量。
# 

def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer


######################################################################
# 最后，我们必须定义一个执行neural transfer的函数。对于网络的每一次迭代，
# 它都得到一个更新的输入，并计算新的损失。
# 我们将运行每个loss module的 ``backward`` 方法来动态地计算它们的梯度。
# 优化器需要一个 “closure” 函数，它重新评估模块并返回损失。
# 
# 我们还有最后一个制约因素要解决。网络可以尝试优化输入值，其值超过图像的0到1张量范围。
# 我们可以通过每次网络运行时将输入值修正为0到1来解决这个问题。
# 

def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1000000, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)
    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    input_img.data.clamp_(0, 1)

    return input_img


######################################################################
# 最后, 我们可以运行算法
# 

output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                            content_img, style_img, input_img)

plt.figure()
imshow(output, title='Output Image')

# sphinx_gallery_thumbnail_number = 4
plt.ioff()
plt.show()

