# -*- coding: utf-8 -*-
"""
保存和加载模型
=========================
**翻译者:** `Antares <http://wwww.studyai.com/antares>`_

本文档提供了关于PyTorch模型的保存和加载的各种用例的解决方案。
可以随意阅读整个文档，或者跳过所需的代码以获得所需的用例。

当涉及到保存和加载模型时，需要熟悉三个核心函数:

1) `torch.save <https://pytorch.org/docs/stable/torch.html?highlight=save#torch.save>`__:
   将序列化对象保存到磁盘。此函数使用Python的 `pickle <https://docs.python.org/3/library/pickle.html>`__ 
   实用程序进行序列化。使用此函数可以保存各种对象的模型、张量和字典。

2) `torch.load <https://pytorch.org/docs/stable/torch.html?highlight=torch%20load#torch.load>`__:
   使用 `pickle <https://docs.python.org/3/library/pickle.html>`__ 的unpickling facilities
   将被pickled的对象文件反序列化到内存。此函数还可方便设备将数据加载进来(请看 
   `Saving & Loading Model Across Devices <#saving-loading-model-across-devices>`__).

3) `torch.nn.Module.load_state_dict <https://pytorch.org/docs/stable/nn.html?highlight=load_state_dict#torch.nn.Module.load_state_dict>`__:
   使用反序列化的 *state_dict* 加载模型的参数字典。 关于 *state_dict* 的更多信息, 请看 `什么是 state_dict? <#what-is-a-state-dict>`__.



**Contents:**

-  `什么是 state_dict? <#what-is-a-state-dict>`__
-  `保存 & 加载 Model 用于推断 <#saving-loading-model-for-inference>`__
-  `保存 & 加载一个CheckPointCheckPoint <#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training>`__
-  `保存多个 Models 到一个文件 <#saving-multiple-models-in-one-file>`__
-  `使用来自不同Model的参数热启动另一个Model <#warmstarting-model-using-parameters-from-a-different-model>`__
-  `跨设备 保存 & 加载 Model <#saving-loading-model-across-devices>`__

"""


######################################################################
# 什么是 ``state_dict``?
# -------------------------
#
# 在PyTorch中，``torch.nn.Module`` 模型的可学习参数(即权重和偏置)包含在模型的 
# *parameters* 中(使用 ``model.parameters()`` 访问)。
# *state_dict* 只是一个Python字典对象，它将每个层映射到其参数张量。
# 请注意，只有具有可学习参数的层(卷积层、线性层等)在模型的 *state_dict* 中有条目(entries)。
# Optimizer对象(``torch.optim``)还有一个 *state_dict* ，它包含关于优化器状态的信息以及使用的超参数。
# 
# 因为 *state_dict* 对象是Python字典，所以可以轻松地保存、更新、修改和恢复它们，
# 从而为PyTorch模型和优化器添加了大量的模块化。
#
# 例子:
# ^^^^^^^^
#
# 让我们从 `Training a classifier <https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py>`__  
# 中使用的简单模型中查看 *state_dict* 。
#
# .. code:: python
#
#    # 定义模型
#    class TheModelClass(nn.Module):
#        def __init__(self):
#            super(TheModelClass, self).__init__()
#            self.conv1 = nn.Conv2d(3, 6, 5)
#            self.pool = nn.MaxPool2d(2, 2)
#            self.conv2 = nn.Conv2d(6, 16, 5)
#            self.fc1 = nn.Linear(16 * 5 * 5, 120)
#            self.fc2 = nn.Linear(120, 84)
#            self.fc3 = nn.Linear(84, 10)
#
#        def forward(self, x):
#            x = self.pool(F.relu(self.conv1(x)))
#            x = self.pool(F.relu(self.conv2(x)))
#            x = x.view(-1, 16 * 5 * 5)
#            x = F.relu(self.fc1(x))
#            x = F.relu(self.fc2(x))
#            x = self.fc3(x)
#            return x
#
#    # 初始化 model
#    model = TheModelClass()
#
#    # 初始化 optimizer
#    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
#
#    # 输出 model 的 state_dict
#    print("Model's state_dict:")
#    for param_tensor in model.state_dict():
#        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
#
#    # 输出 optimizer 的 state_dict
#    print("Optimizer's state_dict:")
#    for var_name in optimizer.state_dict():
#        print(var_name, "\t", optimizer.state_dict()[var_name])
#
# **Output:**
#
# ::
#
#    Model's state_dict:
#    conv1.weight     torch.Size([6, 3, 5, 5])
#    conv1.bias   torch.Size([6])
#    conv2.weight     torch.Size([16, 6, 5, 5])
#    conv2.bias   torch.Size([16])
#    fc1.weight   torch.Size([120, 400])
#    fc1.bias     torch.Size([120])
#    fc2.weight   torch.Size([84, 120])
#    fc2.bias     torch.Size([84])
#    fc3.weight   torch.Size([10, 84])
#    fc3.bias     torch.Size([10])
#
#    Optimizer's state_dict:
#    state    {}
#    param_groups     [{'lr': 0.001, 'momentum': 0.9, 'dampening': 0, 'weight_decay': 0, 'nesterov': False, 'params': [4675713712, 4675713784, 4675714000, 4675714072, 4675714216, 4675714288, 4675714432, 4675714504, 4675714648, 4675714720]}]
#


######################################################################
# 保存 & 加载 Model 用于推断
# ------------------------------------
#
# 保存/加载 ``state_dict`` (推荐方式)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# **保存:**
#
# .. code:: python
#
#    torch.save(model.state_dict(), PATH)
#
# **加载:**
#
# .. code:: python
#
#    model = TheModelClass(*args, **kwargs)
#    model.load_state_dict(torch.load(PATH))
#    model.eval()
#
# 在保存模型进行推理时，只需保存经过训练的模型的学习参数即可。使用 ``torch.save()`` 函数
# 保存模型的 *state_dict* 将为以后恢复模型提供最大的灵活性，这就是为什么推荐使用它来保存模型。
#
# 一个常见的PyTorch约定是使用  ``.pt`` 或 ``.pth`` 文件扩展名保存模型。
#
# 请记住，在运行推理之前，您必须调用 ``model.eval()``  来将 
# dropout 和 batch normalization layers 
# 设置为评估模式。如果不这样做，就会产生不一致的推理结果。
#
# .. Note ::
#
#    注意，``load_state_dict()`` 函数接受字典对象，而 **不是** 保存对象的路径。
#    这意味着在将保存的 *state_dict* 传递给 ``load_state_dict()`` 函数之前，必须对其进行反序列化。
#    例如，不能使用 ``model.load_state_dict(Path)`` 加载 。
#
#
# 保存/加载 整个 Model
# ^^^^^^^^^^^^^^^^^^^^^^
#
# **保存:**
#
# .. code:: python
#
#    torch.save(model, PATH)
#
# **加载:**
#
# .. code:: python
#
#    # Model class must be defined somewhere
#    model = torch.load(PATH)
#    model.eval()
#
# 这个保存/加载过程使用最直观的语法，涉及的代码最少。以这种方式保存模型将使用Python的 
# `pickle <https://docs.python.org/3/library/pickle.html>`__ 模块保存整个model。
# 这种方法的缺点是序列化数据被绑定到保存模型时使用的特定类和精确的目录结构。
# 原因是pickle没有保存模型类本身。相反，它保存到包含类的文件的路径，该类在加载时使用。
# 正因为如此，当您在其他项目中使用时或在重构之后，您的代码可能以各种方式中断。
#
# 一个常见的PyTorch约定是使用  ``.pt`` 或 ``.pth`` 文件扩展名保存模型。
#
# 请记住，在运行推理之前，您必须调用 ``model.eval()``  来将 
# dropout 和 batch normalization layers 
# 设置为评估模式。如果不这样做，就会产生不一致的推理结果。
#


######################################################################
# 保存 & 加载 Checkpoint 用于 推断 and/or 恢复训练
# ----------------------------------------------------------------------------
#
# 保存:
# ^^^^^
#
# .. code:: python
#
#    torch.save({
#                'epoch': epoch,
#                'model_state_dict': model.state_dict(),
#                'optimizer_state_dict': optimizer.state_dict(),
#                'loss': loss,
#                ...
#                }, PATH)
#
# 加载:
# ^^^^^
#
# .. code:: python
#
#    model = TheModelClass(*args, **kwargs)
#    optimizer = TheOptimizerClass(*args, **kwargs)
#
#    checkpoint = torch.load(PATH)
#    model.load_state_dict(checkpoint['model_state_dict'])
#    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#    epoch = checkpoint['epoch']
#    loss = checkpoint['loss']
#
#    model.eval()
#    # - or -
#    model.train()
#
# 当保存一般检查点(用于推理或恢复训练)时，您必须保存的不仅仅是模型的 *state_dict* 。
# 保存优化器的 *state_dict* 也很重要，因为它包含缓冲区和参数，这些缓冲区和参数是随着模型的训练而更新的。
# 其他你可能想要保存的项是 你离开训练过程时的epoch，最新记录的训练损失，外部的 ``torch.nn.Embedding`` 层，等等。
#
# 若要保存多个组件，请将它们组织在字典中，并使用 ``torch.save()`` 对字典进行序列化。
# 一个常见的PyTorch约定是使用  ``.tar``  文件扩展名保存这些检查点。
#
# 要加载这些项，首先初始化模型和优化器，然后使用 ``torch.load()`` 在本地加载字典。从这里，您可以很容易地访问保存的项目，
# 只需查询字典，正如您所期望的。
# 
# 请记住，在运行推理之前，您必须调用 ``model.eval()``  来将 dropout 和 batch normalization layers 
# 设置为评估模式。如果不这样做，就会产生不一致的推理结果。
# 如果希望恢复training，请调用 ``model.train()`` 以确保这些层处于训练模式。
#


######################################################################
# 在一个文件中保存多个 Models 
# ----------------------------------
#
# 保存:
# ^^^^^
#
# .. code:: python
#
#    torch.save({
#                'modelA_state_dict': modelA.state_dict(),
#                'modelB_state_dict': modelB.state_dict(),
#                'optimizerA_state_dict': optimizerA.state_dict(),
#                'optimizerB_state_dict': optimizerB.state_dict(),
#                ...
#                }, PATH)
#
# 加载:
# ^^^^^
#
# .. code:: python
#
#    modelA = TheModelAClass(*args, **kwargs)
#    modelB = TheModelBClass(*args, **kwargs)
#    optimizerA = TheOptimizerAClass(*args, **kwargs)
#    optimizerB = TheOptimizerBClass(*args, **kwargs)
#
#    checkpoint = torch.load(PATH)
#    modelA.load_state_dict(checkpoint['modelA_state_dict'])
#    modelB.load_state_dict(checkpoint['modelB_state_dict'])
#    optimizerA.load_state_dict(checkpoint['optimizerA_state_dict'])
#    optimizerB.load_state_dict(checkpoint['optimizerB_state_dict'])
#
#    modelA.eval()
#    modelB.eval()
#    # - or -
#    modelA.train()
#    modelB.train()
#
# 当保存由多个 ``torch.nn.Modules`` 组成的模型时，例如GAN、sequence-to-sequence model 或 ensemble of models，
# 您将遵循与保存一般检查点相同的方法。换句话说，保存每个模型的 *state_dict* 的字典和相应的优化器。
# 如前所述，只需将任何其他项目追加到字典中，就可以保存可能帮助您恢复训练的任何其他项目。
#
# 一个常见的PyTorch约定是使用  ``.tar``  文件扩展名保存这些检查点。
#
# 要加载模型，首先初始化模型和优化器，然后使用 ``torch.load()`` 本地加载字典。
# 从这里，您可以很容易地访问保存的项目，只需查询字典，正如您所期望的。
#
# 请记住，在运行推理之前，您必须调用 ``model.eval()``  来将 dropout 和 batch normalization layers 
# 设置为评估模式。如果不这样做，就会产生不一致的推理结果。
# 如果希望恢复training，请调用 ``model.train()`` 以确保这些层处于训练模式。
#


######################################################################
# 使用来自不同Model的参数热启动另一个Model 
# ----------------------------------------------------------
#
# 保存:
# ^^^^^
#
# .. code:: python
#
#    torch.save(modelA.state_dict(), PATH)
#
# 加载:
# ^^^^^
#
# .. code:: python
#
#    modelB = TheModelBClass(*args, **kwargs)
#    modelB.load_state_dict(torch.load(PATH), strict=False)
#
# 当迁移学习或训练一个新的复杂模型时，部分加载模型或加载部分模型是常见的场景。
# 利用经过训练的参数，即使只有少数参数是可用的，也将有助于启动训练过程，
# 并有望帮助您的模型比从头开始的训练更快地收敛。
#
# 无论您是从缺少一些键的部分 *state_dict* 加载，还是加载一个 *state_dict* 中的键比您要加载的模型多，
# 您都可以在 ``load_state_dict()`` 函数中将 ``strict``  参数设置为 ``false`` ，以忽略不匹配的键。
#
# 如果希望将参数从一个层加载到另一个层，但有些键不匹配，则只需更改要加载的 *state_dict*  中参数键的名称，
# 以与加载到的模型中的键匹配。 
#


######################################################################
# 跨设备 保存 & 加载 Model 
# -------------------------------------
#
# 保存在GPU, 加载到CPU
# ^^^^^^^^^^^^^^^^^^^^^^^^
#
# **保存:**
#
# .. code:: python
#
#    torch.save(model.state_dict(), PATH)
#
# **加载:**
#
# .. code:: python
#
#    device = torch.device('cpu')
#    model = TheModelClass(*args, **kwargs)
#    model.load_state_dict(torch.load(PATH, map_location=device))
#
# 在CPU上加载用GPU训练的模型时，将 ``torch.device('cpu')`` 传递到 ``torch.load()`` 函数中的 ``map_location`` 参数。
# 在这种情况下，在张量基础上的存储将使用 ``map_location`` 参数动态地重映射到CPU设备。
#
# 保存在 GPU, 加载到 GPU
# ^^^^^^^^^^^^^^^^^^^^^^^^
#
# **保存:**
#
# .. code:: python
#
#    torch.save(model.state_dict(), PATH)
#
# **加载:**
#
# .. code:: python
#
#    device = torch.device("cuda")
#    model = TheModelClass(*args, **kwargs)
#    model.load_state_dict(torch.load(PATH))
#    model.to(device)
#    # Make sure to call input = input.to(device) on any input tensors that you feed to the model
#
# 当把在GPU上训练和保存的模型加载到GPU上时，只需使用 ``model.to(torch.device('cuda'))`` 将初始化的 ``model`` 转换为CUDA优化模型。
# 此外，确保在所有模型输入上使用 ``.to(torch.device('cuda'))`` 函数来为模型准备数据。
# 注意，调用 ``my_tensor.to(device)`` 将返回GPU上 ``my_tensor`` 的新副本。它 **不会** 覆盖  ``my_tensor`` 。
# 因此，请记住手动覆盖张量：``my_tensor = my_tensor.to(torch.device('cuda'))`` 。
# 
#
# 保存在CPU, 加载到GPU
# ^^^^^^^^^^^^^^^^^^^^^^^^
#
# **保存:**
#
# .. code:: python
#
#    torch.save(model.state_dict(), PATH)
#
# **加载:**
#
# .. code:: python
#
#    device = torch.device("cuda")
#    model = TheModelClass(*args, **kwargs)
#    model.load_state_dict(torch.load(PATH, map_location="cuda:0"))  # Choose whatever GPU device number you want
#    model.to(device)
#    # 确保调用 input = input.to(device) 在任意的输入tensors上 
#
# 当把在CPU上训练和保存的模型 加载到GPU上时，请将 ``torch.load()`` 函数中的 ``map_location`` 参数设置为
# *cuda:device_id* 。这会将模型加载到给定的GPU设备上。接下来，请确保调用
# ``model.to(torch.device('cuda'))`` 将模型的参数张量转换为CUDA张量。
# 最后，确保在所有模型输入上使用 ``.to(torch.device('cuda'))`` 函数来为CUDA优化模型准备数据。
# 注意，调用 ``my_tensor.to(device)`` 将返回GPU上 ``my_tensor`` 的新副本。它不会覆盖 ``my_tensor`` 。
# 因此，请记住手动覆盖张量：``my_tensor = my_tensor.to(torch.device('cuda'))`` 。
#

# 保存 ``torch.nn.DataParallel`` Models
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# **保存:**
#
# .. code:: python
#
#    torch.save(model.module.state_dict(), PATH)
#
# **加载:**
#
# .. code:: python
#
#    # 加载到任何你想加载的设备上
#
# ``torch.nn.DataParallel`` 是一个支持并行GPU利用率的模型包装器。
# 要一般性地保存 ``DataParallel`` 模型，请保存 ``model.module.state_dict()``。
# 这样，您就可以灵活地将模型加载到任何您想要的设备上。
#
