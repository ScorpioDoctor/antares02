# -*- coding: utf-8 -*-
"""
数据加载和处理教程
====================================
**翻译者**: `Antares <http://www.studyai.com/antares>`_

在解决任何机器学习问题时我们通常付出了很大的努力来准备数据。PyTorch提供了许多工具来简化数据加载，
并希望能够使您的代码更具可读性。在本教程中，我们将了解如何从非平凡的数据集中加载和预处理/增强数据。

要运行本教程, 请确保安装了这些 packages :

-  ``scikit-image``: 用于图像的输入输出(IO)和变换(transforms)
-  ``pandas``: 用于更加简单的解析 csv 文件

"""

from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # 交互式模式

######################################################################
# 我们要处理的数据集是面部姿态(facial pose).
# 这意味着一张人脸将被如下标注:
#
# .. figure:: /_static/img/landmarked_face2.png
#    :width: 400
#
# 总体上, 每张脸上标注了 68 个不同的landmark points。
#
# .. note::
#     从 `这里 <https://download.pytorch.org/tutorial/faces.zip>`_ 下载数据集， 
#     以便 图像数据的存放目录结构是这样的：'data/faces/' 。
#     这个数据集事实上是使用 `dlib 的姿态估计 <http://blog.dlib.net/2014/08/real-time-face-pose-estimation.html>`__
#     来产生的，所用的图像来自于 imagenet 中标记为 'face' 的若干张图像。
#
# 数据集自带一个 csv 文件，里面是存放着 标注(annotations)，就像这样哒:
#
# ::
#
#     image_name,part_0_x,part_0_y,part_1_x,part_1_y,part_2_x, ... ,part_67_x,part_67_y
#     0805personali01.jpg,27,83,27,98, ... 84,134
#     1084239450_e76e00b7e7.jpg,70,236,71,257, ... ,128,312
#
# 让我们快速读取 CSV 文件 然后获得标注信息，并保存到一个 (N, 2) 的数组中去吧，其中 N 是
# landmarks 的数量。
#

landmarks_frame = pd.read_csv('data/faces/face_landmarks.csv')

n = 65
img_name = landmarks_frame.iloc[n, 0]
landmarks = landmarks_frame.iloc[n, 1:].as_matrix()
landmarks = landmarks.astype('float').reshape(-1, 2)

print('Image name: {}'.format(img_name))
print('Landmarks shape: {}'.format(landmarks.shape))
print('First 4 Landmarks: {}'.format(landmarks[:4]))


######################################################################
# 让我们编写一个简单的辅助函数来显示一个图像及其标注，并使用它来显示一个示例。
#

def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated

plt.figure()
show_landmarks(io.imread(os.path.join('data/faces/', img_name)),
               landmarks)
plt.show()


######################################################################
# Dataset 类
# -------------
#
# ``torch.utils.data.Dataset`` 是表示数据集的抽象类。
# 您的自定义数据集类应该继承 ``Dataset`` 并覆盖以下方法：
#
# -  ``__len__`` 以便 ``len(dataset)`` 可以返回数据集的size。
# -  ``__getitem__`` 用于支持类似 ``dataset[i]`` 这样的索引，用来取得第 :math:`i` 个样本。
#
# 让我们为我们的脸地标数据集创建一个DataSet类。我们将在 ``__init__`` 中读取CSV，
# 但将图像的读取留给 ``__getitem__`` 。
# 这是内存有效的，因为所有的图像不是一次存储在内存中，而是根据需要读取。
#
# 我们数据集的样本将会是一个字典，像这样 
# ``{'image': image, 'landmarks': landmarks}``。 我们的数据集将会接受一个可选参数 
# ``transform`` 以便任何需要的数据预处理可以施加到样本上。 
# 我们将会在下一个小节看到 ``transform`` 的用处。
#

class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample


######################################################################
# 让我们实例化这个类并迭代数据样本。我们将打印前4个样本的大小并显示它们的landmarks。
#

face_dataset = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv',
                                    root_dir='data/faces/')

fig = plt.figure()

for i in range(len(face_dataset)):
    sample = face_dataset[i]

    print(i, sample['image'].shape, sample['landmarks'].shape)

    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    show_landmarks(**sample)

    if i == 3:
        plt.show()
        break


######################################################################
# 变换(Transforms)
# ---------------------
#
# 从上面我们可以看到一个问题，就是样本的尺寸不一样。大多数神经网络都期望得到固定
# 大小的图像。因此，我们需要编写一些预处理代码。让我们创建三个转换：
#
# -  ``Rescale``: 缩放图像
# -  ``RandomCrop``: 随机裁剪图像. 用于数据增广(data augmentation).
# -  ``ToTensor``: 把 numpy images 转换为 torch images (我们需要交换坐标轴).
#
# 我们将把它们写成可调用的类(callable classes)，而不是简单的函数，
# 这样每次调用transform时都不需要传递转换的参数。
# 为此，我们只需实现 ``__call__`` 方法，如果需要，则实现 ``__init__`` 方法。
# 然后我们可以使用这样的转换：
#
# ::
#
#     tsfm = Transform(params)
#     transformed_sample = tsfm(sample)
#
# 下面观察如何将这些转换应用于图像和landmarks。 
#

class Rescale(object):
    """把图像缩放到一个给定的尺寸。

    Args:
        output_size (tuple or int): 想要的输出尺寸. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}


class RandomCrop(object):
    """在一个图像样本上随机裁切图像.

    Args:
        output_size (tuple or int): 想要的输出尺寸. If int, square crop is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}


class ToTensor(object):
    """把样本的 ndarrays 转换为 Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}


######################################################################
# 复合式变换器
# ~~~~~~~~~~~~~~~~~~
#
# 现在，我们将转换应用于一个示例。
#
# 比方说，我们想要将图像的较短部分恢复到256，
# 然后从图像中随机裁剪出大小为224的方形图像。
# 也就是说，我们想要合成 ``Rescale`` 和 ``RandomCrop`` 变换。
# ``torchvision.transforms.Compose`` 是一个简单的可调用类，它允许我们这样做。
#

scale = Rescale(256)
crop = RandomCrop(128)
composed = transforms.Compose([Rescale(256),
                               RandomCrop(224)])

# 将上述每个转换应用于示例
fig = plt.figure()
sample = face_dataset[65]
for i, tsfrm in enumerate([scale, crop, composed]):
    transformed_sample = tsfrm(sample)

    ax = plt.subplot(1, 3, i + 1)
    plt.tight_layout()
    ax.set_title(type(tsfrm).__name__)
    show_landmarks(**transformed_sample)

plt.show()


######################################################################
# 在数据集上迭代遍历
# -----------------------------
#
# 让我们将所有这些放在一起创建一个具有组合转换的数据集。总之，每次采样该数据集时
#
# -  从文件中动态读取一张图像。
# -  Transforms 被应用于读取出的图像。
# -  由于其中一个变换是随机的，所以在采样时会增加数据。 
#
# 我们可以像以前一样使用 ``for i in range`` 循环迭代创建的数据集。
#

transformed_dataset = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv',
                                           root_dir='data/faces/',
                                           transform=transforms.Compose([
                                               Rescale(256),
                                               RandomCrop(224),
                                               ToTensor()
                                           ]))

for i in range(len(transformed_dataset)):
    sample = transformed_dataset[i]

    print(i, sample['image'].size(), sample['landmarks'].size())

    if i == 3:
        break



######################################################################
# 但是，通过使用一个简单的for循环来迭代数据，我们失去了很多特性。特别是，我们错过了:
#
# -  批量化数据
# -  随机打乱数据
# -  使用 ``multiprocessing`` workers 并行加载数据.
#
# ``torch.utils.data.DataLoader`` 是一个提供了上述特性的迭代器。
# 我们应该对下面所用的参数有所明了。 其中一个有意思的参数是 ``collate_fn`` 。
# 你可以通过 ``collate_fn`` 来明确的指定样本如何被批量化。
# 然而, 默认的设置已经足以应付大多数情况啦。
#

dataloader = DataLoader(transformed_dataset, batch_size=4,
                        shuffle=True, num_workers=4)


# 显示一个批次的辅助函数
def show_landmarks_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, landmarks_batch = \
            sample_batched['image'], sample_batched['landmarks']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    for i in range(batch_size):
        plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size,
                    landmarks_batch[i, :, 1].numpy(),
                    s=10, marker='.', c='r')

        plt.title('Batch from dataloader')

for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['image'].size(),
          sample_batched['landmarks'].size())

    # observe 4th batch and stop.
    if i_batch == 3:
        plt.figure()
        show_landmarks_batch(sample_batched)
        plt.axis('off')
        plt.ioff()
        plt.show()
        break


######################################################################
# 后记: torchvision
# ----------------------
#
# 在本教程中，我们已经了解了如何编写和使用数据集类、变换器类和数据加载器类。
# ``torchvision`` 包提供了一些常见的数据集类和变换器类。 您甚至可能不必编写自定义类。
# ``torchvision`` 中可用的更通用的数据集之一是 ``ImageFolder`` 。
# 它假定图像的组织方式如下::
#
#     root/ants/xxx.png
#     root/ants/xxy.jpeg
#     root/ants/xxz.png
#     .
#     .
#     .
#     root/bees/123.jpg
#     root/bees/nsdf3.png
#     root/bees/asd932_.png
#
# 其中 'ants', 'bees' etc. 是类标签。 类似的，操作 ``PIL.Image`` 类型的图像的通用变换，比如
# ``RandomHorizontalFlip``, ``Scale`` 等也是可用的。
# 你可以使用这些来写一个 dataloader，就像这样::
#
#   import torch
#   from torchvision import transforms, datasets
#
#   data_transform = transforms.Compose([
#           transforms.RandomSizedCrop(224),
#           transforms.RandomHorizontalFlip(),
#           transforms.ToTensor(),
#           transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                std=[0.229, 0.224, 0.225])
#       ])
#   hymenoptera_dataset = datasets.ImageFolder(root='hymenoptera_data/train',
#                                              transform=data_transform)
#   dataset_loader = torch.utils.data.DataLoader(hymenoptera_dataset,
#                                                batch_size=4, shuffle=True,
#                                                num_workers=4)
#
# 要找数据加载器，变换器和训练过程结合起来的例子, 请看
# :doc:`transfer_learning_tutorial`.
