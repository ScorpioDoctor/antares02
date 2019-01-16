# -*- coding: utf-8 -*-
"""
MSCOCO数据集的加载读取
==============================

**作者**: `Antares博士 <http://www.studyai.com/antares>`__

我们将介绍MSCOCO数据集的pycocotools的Python API的使用方法，以及 ``torchvison.datasets.CocoCaptions`` 、
``torchvison.datasets.CocoDetection`` 的用法

"""

##############################################################################################
# 安装 pycocotools
# -------------------------
# 
# 请按照这个文档的说明下载安装：https://blog.csdn.net/daniaokuye/article/details/78699138
# 


##############################################################################################
# 数据文件目录结构
# -------------------------
# 
# **下载数据**: 
# 
# - annotations_trainval2017.zip  (`[下载连接] <http://images.cocodataset.org/annotations/annotations_trainval2017.zip>`__)
# - image_info_test2017.zip     (`[下载连接] <http://images.cocodataset.org/annotations/image_info_test2017.zip>`__)
# - stuff_annotations_trainval2017.zip (`[连接] <http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip>`__)
# - test2017.zip    (`[下载连接] <http://images.cocodataset.org/zips/test2017.zip>`__)
# - train2017.zip   (`[下载连接] <http://images.cocodataset.org/zips/train2017.zip>`__)
# - val2017.zip    (`[下载连接] <http://images.cocodataset.org/zips/val2017.zip>`__)
# 
# 将zip文件解压到coco这个目录下，其中，图像放在 ```coco/images/``` 下面，标注放在 ```coco/annotations``` 下面
# 
# 下面是我的目录树，安装上Ubuntu的tree命令就可以查看

# ```
# (pytorchenv1) zhjm@tower:~/Downloads/MSCOCO2017$ tree -L 3
# .
# ├── coco
# │   ├── annotations
# │   │   ├── annotations_trainval2017
# │   │   ├── image_info_test2017
# │   │   └── stuff_annotations_trainval2017
# │   └── images
# │       ├── test2017
# │       ├── train2017
# │       └── val2017
# └── zip files
#     ├── annotations_trainval2017.zip 
#     ├── image_info_test2017.zip     
#     ├── stuff_annotations_trainval2017.zip 
#     ├── test2017.zip  
#     ├── train2017.zip  
#     └── val2017.zip  

# ```



##############################################################################################
# 导入依赖包
# -------------------------
# 

from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

##############################################################################################
# 指定数据存放路径
# -------------------------
# 
dataDir='/home/zhjm/Downloads/MSCOCO2017/coco'
dataType='val2017'
annFile='{}/annotations/annotations_trainval2017/instances_{}.json'.format(dataDir,dataType)
print(annFile)

##############################################################################################
# Pycocotools的API的用法
# ----------------------------
# 总共有三种任务：
# 
# * 实例标注(Instances Annotations);
# * 人体关键点标注(Human Keypoints Annotations);
# * 看图说话标注(Caption Annotations)
# 
# 

##############################################################################################
# 初始化实例标注的COCO API
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
coco=COCO(annFile)

##############################################################################################
# 展示COCO类别和超类别
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))

##############################################################################################
# 获取包含某些类别的所有图像
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# 意思是获取到的每个图像都必须包含catNms=['person','dog','skateboard']中指定的物体实例。
# 
# 满足检索条件的只有三张图像：同时包含了 人， 狗 ， 滑板(skateboard)

catIds = coco.getCatIds(catNms=['person','dog','skateboard']);
print("类别编号：", catIds)
imgIds = coco.getImgIds(catIds=catIds);
print(type(imgIds),len(imgIds))
print("图像编号：", imgIds)

##############################################################################################
# 在符合条件的多张图片中随机选择一张
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
imgIds = coco.getImgIds(imgIds = [549220, 324158, 279278])
print(imgIds)
img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
print(type(img))
print(list(img.keys()))
print(img)

##############################################################################################
# 加载与显示图像
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

# I = io.imread('%s/images/%s/%s'%(dataDir,dataType,img['file_name']))
# use url to load image
I = io.imread(img['coco_url'])  # img 是个字典，保存着该图像的anns信息
plt.axis('off')
plt.imshow(I)
plt.show()

##############################################################################################
# 加载与显示实例标注信息
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
plt.imshow(I); plt.axis('off')
annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco.loadAnns(annIds)
coco.showAnns(anns)

##############################################################################################
# 初始化人体关键点标注信息的COCO API
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
annFile = '{}/annotations/annotations_trainval2017/person_keypoints_{}.json'.format(dataDir,dataType)
coco_kps=COCO(annFile)

##############################################################################################
# 加载与显示关键点标注信息
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
plt.imshow(I); plt.axis('off')
ax = plt.gca()
annIds = coco_kps.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco_kps.loadAnns(annIds)
coco_kps.showAnns(anns)

##############################################################################################
# 初始化Caption标注信息的COCO API
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
annFile = '{}/annotations/annotations_trainval2017/captions_{}.json'.format(dataDir,dataType)
coco_caps=COCO(annFile)

##############################################################################################
# 加载和显示Caption标注信息
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
annIds = coco_caps.getAnnIds(imgIds=img['id']);
anns = coco_caps.loadAnns(annIds)
coco_caps.showAnns(anns)
plt.imshow(I); plt.axis('off'); plt.show()

##############################################################################################
# PyTorch中的MSCOCO读取方法
# -----------------------------------
# 主要介绍 ``torchvison.datasets.CocoCaptions`` 、
# ``torchvison.datasets.CocoDetection`` 的用法

##############################################################################################
# CocoCaptions(···)的使用方法
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
import torchvision.datasets as dset
import torchvision.transforms as transforms

dataDir='/home/zhjm/Downloads/MSCOCO2017/coco'
dataType='train2017'  #如果要获取测试集，就 'test2017'
annFile = '{}/annotations/annotations_trainval2017/captions_{}.json'.format(dataDir,dataType)
print(annFile)

root = dataDir + '/images/' + dataType

# 对数据做变换的变换类
transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = dset.CocoCaptions(root = root, annFile = annFile, transform=transform_train)

print('样本数量: ', len(trainset))
img, target = trainset[3] # 加载第四个样本

print("数据类型：", type(img))
print("图像尺寸: ", img.size())
print("标注信息：",target)


#############################################################################################
# 显示图像
import matplotlib.pyplot as plt
import numpy as np

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    # 把 Tensor 转换为 Numpy类型
    npimg = img.numpy()
    # 必须要调整通道
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off'); plt.show()

# get a random training image
image, target = trainset[np.random.randint(0,len(trainset))]

# show images
imshow(image)
# print labels
print(target)

##############################################################################################
# CocoDetection(···)的使用方法
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
import torchvision.datasets as dset
import torchvision.transforms as transforms

dataDir='/home/zhjm/Downloads/MSCOCO2017/coco'
dataType='val2017'  #如果要获取测试集，就 'val2017'
annFile = '{}/annotations/annotations_trainval2017/instances_{}.json'.format(dataDir,dataType)
print(annFile)

root = dataDir + '/images/' + dataType

# 对数据做变换的变换类
transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = dset.CocoDetection(root = root, annFile = annFile, transform=transform_train)

print('样本数量: ', len(trainset))
img, target = trainset[3] # 加载第四个样本,Tuple (image, target). target is the object returned by `coco.loadAnns`

print("图像数据类型：", type(img))
print("图像尺寸: ", img.size())
print("数据类型：",type(target))  # 是一个字典列表，每个物体对应一个字典
print("物体数量：",len(target))  # 
print("字典的键：",list(target[0].keys()))  # 
print("第一个物体的标注信息：",target[0])

#############################################################################################
# 显示图像

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image and anns
def imshow(img, anns=None):
    img = img / 2 + 0.5     # unnormalize
    # 把 Tensor 转换为 Numpy类型
    npimg = img.numpy()
    # 必须要调整通道
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if anns is not None:
        coco.showAnns(anns=anns)
    plt.axis('off'); plt.show()

# get a random training image
image, target = trainset[np.random.randint(0,len(trainset))]

# show images and target(anns)
imshow(image, target)
