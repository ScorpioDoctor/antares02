# -*- coding: utf-8 -*-
"""
对抗样本生成
==============================

**Author:** `Nathan Inkawhich <https://github.com/inkawhich>`__

**翻译者**: `Antares博士 <http://www.studyai.com/antares>`__

如果你正在阅读这篇文章，希望你能体会到一些机器学习模型是多么的有效。研究不断推动ML模型变得更快、更准确和更高效。
然而，设计和训练模型的一个经常被忽视的方面是安全性和健壮性，特别是在面对希望欺骗模型的对手时。

本教程将提高您对ML模型的安全漏洞的认识，并将深入了解对抗性机器学习的热门话题。
您可能会惊讶地发现，在图像中添加不可察觉的扰动会导致截然不同的模型性能。
鉴于这是一个教程，我们将通过一个图像分类器的例子来探讨这个主题。
具体来说，我们将使用第一种也是最流行的攻击方法-快速梯度符号攻击(Fast Gradient Sign Attack ,FGSM)来欺骗MNIST分类器。

"""


######################################################################
# 威胁模型(Threat Model)
# -------------------------
# 
# 有很多种类的对抗性攻击，每种攻击都有不同的目标和攻击者的知识假设。但是，总体目标
# 是在输入数据中增加最少的扰动量，以导致期望的错误分类。攻击者的知识有几种假设，其中两种假设是：
# **白盒子(white-box)** 和 **黑盒子(black-box)**。
# *白盒子* 攻击假定攻击者拥有对模型的全部知识和访问权限，包括体系结构、输入、输出和权重。
# *黑盒子* 攻击假设攻击者只能访问模型的输入和输出，而对底层架构或权重一无所知。
# 还有几种目标类型，包括 **错误分类(misclassification)** 和 **源/目标错误分类(source/target misclassification)** 。
# *错误分类* 的目标意味着对手只希望输出分类是错误的，而不关心新的分类是什么。
# *源/目标错误分类* 意味着对手希望更改最初属于特定源类的图像，从而将其归类为特定的目标类。
# 
# 在这种情况下，FGSM攻击是以 *错误分类* 为目标的 *白盒攻击* 。 有了这些背景信息，我们现在可以详细讨论攻击(attack)了。
# 
# 快速梯度符号攻击(Fast Gradient Sign Attack)
# --------------------------------------------
# 
# 迄今为止，第一次也是最流行的对抗性攻击(adversarial attacks)之一被称为 *快速梯度符号攻击(FGSM)* ，
# 古德费尔特对此进行了描述:  `Explaining and Harnessing Adversarial Examples <https://arxiv.org/abs/1412.6572>`__。
# 攻击是非常强大的，但却是直观的。它是设计用来攻击神经网络，利用他们的学习方式，*梯度* 。其思想很简单，
# 不是通过调整基于反向传播梯度的权重来最小化损失，而是 *基于相同的反向传播梯度调整输入数据，
# 使损失最大化* 。换句话说，攻击使用损失W.r.t输入数据的梯度，然后调整输入数据以最大化损失。
# 
# 在我们进入代码之前，让我们看一下著名的 `FGSM <https://arxiv.org/abs/1412.6572>`__  熊猫示例，并提取一些记号(notation)。
#
# .. figure:: /_static/img/fgsm_panda_image.png
#    :alt: fgsm_panda_image
#
# 从图片中, :math:`\mathbf{x}` 是被正确分类为“panda”的原始图像， :math:`y` 
# 是 :math:`\mathbf{x}` 的真正的类标签。
# :math:`\mathbf{\theta}` 表示模型参数，并且 :math:`J(\mathbf{\theta}, \mathbf{x}, y)` 用来
# 训练网络的损失。 攻击将梯度反向传播回输入数据以进行计算 :math:`\nabla_{x} J(\mathbf{\theta}, \mathbf{x}, y)` 。
# 然后，它沿着使损失最大化的方向(i.e. :math:`sign(\nabla_{x} J(\mathbf{\theta}, \mathbf{x}, y))`) 上
# 调整输入数据一小步(:math:`\epsilon` 或 :math:`0.007` 在图片中)。
# 由此产生的扰动图像(perturbed image), :math:`x'`, 就会被目标网络 *误分类(misclassified)* 为 “gibbon”，
# 但事实上 被扰动的图像依然是个 “panda” 。
# 
# 希望现在你已明了本教程的动机了，所以让我们跳到它的具体实现吧。
# 

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt


######################################################################
# 实现
# --------------
# 
# 在这一小节中, 我们将讨论输入参数，定义在攻击之下的模型，然后编写攻击代码然后将一些测试跑起来。
# 
# 输入
# ~~~~~~
# 
# 本教程只有三个输入，定义如下:
# 
# -  **epsilons** - 要用于运行的epsilon值列表。在列表中保持0很重要，因为它代表了原始测试集上的模型性能。而且，从直觉上说，
#    我们认为epsilon越大，扰动越明显，但攻击越有效，降低了模型的准确性。由于 数据的范围是 :math:`[0,1]` ，任何epsilon值都不应超过1。
# 
# -  **pretrained_model** - 通向预先训练过的MNIST模型的路径，该模型是用 
#    `pytorch/examples/mnist <https://github.com/pytorch/examples/tree/master/mnist>`__ 。
#    为了简单起见，请在 `这里 <https://drive.google.com/drive/folders/1fn83DF14tWmit0RTKWRhPq5uVXt73e0h?usp=sharing>`__ 
#    下载经过预先训练的模型。
# 
# -  **use_cuda** - 布尔标志使用CUDA(如果需要和可用的话)。注意，带有CUDA的GPU对于本教程来说并不重要，因为CPU不会花费太多时间。
# 

epsilons = [0, .05, .1, .15, .2, .25, .3]
pretrained_model = "./data/lenet_mnist_model.pth"
use_cuda=True


######################################################################
# 受攻击模型(Model Under Attack)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# 如前所述，受攻击的模型是与 `pytorch/examples/mnist <https://github.com/pytorch/examples/tree/master/mnist>`__ 
# 相同的MNIST模型。您可以训练和保存自己的MNIST模型，也可以下载和使用所提供的模型。
# 这里的网络定义和测试dataloader是从MNIST示例中复制的。本节的目的是定义model和dataloader，
# 然后初始化模型并加载预先训练的权重。
# 

# LeNet Model definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# MNIST Test dataset 和 dataloader 声明
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data/mnist', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            ])), 
        batch_size=1, shuffle=True)

# 定义我们要使用的设备
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# 初始化网络
model = Net().to(device)

# 加载预训练模型
model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))

# 将模型设置为评估模式. 这是为了 Dropout layers。
model.eval()


######################################################################
# FGSM Attack
# ~~~~~~~~~~~~~~~~~~~
# 
# 现在，我们可以通过扰动原始输入来定义创建对抗性样例(adversarial examples)的函数。
# ``fgsm_attack`` 函数接收三个输入： *image* 是原始的干净图像 (:math:`x`), *epsilon* 是
# 逐像素扰动量 (:math:`\epsilon`), 而 *data_grad* 是损失相对于(w.r.t)输入图像的梯度：
# (:math:`\nabla_{x} J(\mathbf{\theta}, \mathbf{x}, y)`) 。 有了这三个输入，该函数就会按下述方法
# 创建扰动图像(perturbed image):
# 
# .. math:: perturbed\_image = image + epsilon*sign(data\_grad) = x + \epsilon * sign(\nabla_{x} J(\mathbf{\theta}, \mathbf{x}, y))
# 
# 最后, 为了保持数据的原始范围，将扰动图像裁剪到 :math:`[0,1]` 范围内。
# 

# FGSM 攻击代码
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


######################################################################
# 测试函数
# ~~~~~~~~~~~~~~~~
# 
# 最后，本教程的中心结果来自于 ``test`` 函数。每次调用该测试函数都会在MNIST测试集上执行完整的测试步骤，
# 并报告最终的准确性。但是，请注意，此函数也接受 *epsilon* 输入。这是因为 ``test`` 函数报告了一个模型的准确性，
# 该模型正受到来自实力 :math:`\epsilon` 的对手的攻击。更具体地说，对于测试集中的每个样本，
# 该函数计算loss w.r.t the input data (:math:`data\_grad`)，用 ``fgsm_attack`` (:math:`perturbed\_data`) 
# 创建一个受扰动的图像，然后检查被扰动的样例是否是对抗性的。除了测试模型的准确性外，
# 该函数还保存并返回了一些成功的对抗性样例，以供以后可视化。
# 

def test( model, device, test_loader, epsilon ):

    # Accuracy counter
    correct = 0
    adv_examples = []

    # Loop over all examples in test set
    for data, target in test_loader:

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            continue

        # Calculate the loss
        loss = F.nll_loss(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # Re-classify the perturbed image
        output = model(perturbed_data)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples


######################################################################
# 运行 Attack
# ~~~~~~~~~~~~~~~~~~
# 
# 实现的最后一部分是实际运行攻击。在这里，我们对 *epsilons* 输入中的每个epsilon值运行一个完整的测试步骤。
# 对于每个epsilon，我们还保存了最终的准确性和一些成功的对抗性样例，将在接下来绘制出来。
# 注意打印精度是如何随着epsilon值的增加而降低的。另外，请注意 :math:`\epsilon=0` 
# 表示原始测试的准确性，没有任何攻击。
# 

accuracies = []
examples = []

# Run test for each epsilon
for eps in epsilons:
    acc, ex = test(model, device, test_loader, eps)
    accuracies.append(acc)
    examples.append(ex)


######################################################################
# 结果
# -------
# 
# Accuracy vs Epsilon
# ~~~~~~~~~~~~~~~~~~~~~
# 
# 第一个结果是accuracy vs epsilon的图。正如前面提到的，随着epsilon的增加，我们预计测试的准确性会下降。
# 这是因为更大的epsilon意味着我们朝着最大化损失的方向迈出了更大的一步。注意，即使epsilon值是线性的，
# 曲线中的趋势也不是线性的。例如，在 :math:`\epsilon=0.05` 处的准确度仅比 :math:`\epsilon=0.15` 低4%，
# 而 :math:`\epsilon=0.2` 的准确度比 :math:`\epsilon=0.15` 低25%。
# 另外，注意模型的精度对10类分类器的随机精度影响在 :math:`\epsilon=0.25` 和 :math:`\epsilon=0.3` 之间。
# 

plt.figure(figsize=(5,5))
plt.plot(epsilons, accuracies, "*-")
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.arange(0, .35, step=0.05))
plt.title("Accuracy vs Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.show()


######################################################################
# 一些对抗性样本
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# 还记得没有免费午餐的思想吗？在这种情况下，随着epsilon的增加，测试精度降低，但扰动变得更容易察觉。
# 实际上，攻击者必须考虑的是准确性、程度和可感知性之间的权衡。在这里，我们展示了在每个epsilon值下
# 一些成功的对抗性样例。图中的每一行都显示不同的epsilon值。第一行是 :math:`\epsilon=0` 示例，
# 它表示原始的“干净”图像，没有任何扰动。每幅图像的标题显示“原始分类->对抗性分类”。
# 注意，当 :math:`\epsilon=0.15` 时，扰动开始变得明显，在 :math:`\epsilon=0.3` 时非常明显。
# 然而，在所有情况下，人类仍然能够识别正确的类别，尽管增加了噪音。
# 

# Plot several examples of adversarial samples at each epsilon
cnt = 0
plt.figure(figsize=(8,10))
for i in range(len(epsilons)):
    for j in range(len(examples[i])):
        cnt += 1
        plt.subplot(len(epsilons),len(examples[0]),cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        if j == 0:
            plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
        orig,adv,ex = examples[i][j]
        plt.title("{} -> {}".format(orig, adv))
        plt.imshow(ex, cmap="gray")
plt.tight_layout()
plt.show()


######################################################################
# 下一步去哪里?
# -----------------
# 
# 希望本教程能提供一些关于对抗性机器学习主题的见解。这里有许多潜在的方向可走。
# 这种攻击代表了对抗性攻击研究的开始，并且由于有许多关于如何攻击和保护ML模型不受对手攻击的想法。
# 实际上，在NIPS 2017的比赛中，存在着一种对抗性的攻防竞争，
# `本文 <https://arxiv.org/pdf/1804.00097.pdf>`__ 介绍了在这场比赛中所采用的许多方法：对抗攻击和防御竞争。
# 防御方面的工作也带来了使机器学习模型在一般情况下更加健壮的想法，
# 使机器学习模型既具有自然的扰动性，又具有对抗性的输入。
# 
# 另一个方向是不同领域的对抗攻击和防御。对抗性研究并不局限于图像领域，请看 `这个 <https://arxiv.org/pdf/1801.01944.pdf>`__  
# 对语音到文本模型的攻击。
# 但是也许了解更多对抗性机器学习的最好方法是弄脏你的手(意思是让你动手尝试)。
# 尝试实现来自NIPS 2017 竞赛的不同的攻击策略，看看它与FGSM有何不同。然后，试着保护模型不受你自己的攻击。
# 

