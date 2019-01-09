学习 PyTorch 的 Examples
******************************
**翻译者**: `Antares博士 <http://www.studyai.com/antares>`_

本教程通过自包含的例子介绍 `PyTorch <https://github.com/pytorch/pytorch>`__  的基本概念。

PyTorch的核心是提供了两个主要特性:

- n维Tensor，类似于numpy，但可以在GPU上运行。
- 建立和训练神经网络的自动微分

我们将使用一个完全连接的relu网络作为我们的运行示例。
该网络将有一个单一的隐藏层，并将用梯度下降训练，
为了适应随机数据，通过最小化网络输出和真正的输出的欧氏距离
来更新网络模型参数。

.. Note::
	你可以单独浏览和下载这个示例，在 :ref:`这个页面的最后 <examples-download>` 。
	

.. contents:: Table of Contents
	:local:

张量
=======

热身: numpy
--------------

在介绍PyTorch之前，我们首先使用numpy实现网络。

Numpy提供了一个n维数组对象，以及许多用于操作这些数组的函数.
Numpy是一个用于科学计算的通用框架；它对计算图、深度学习或梯度一无所知。
但是，我们可以很容易地使用numpy来拟合两层网络中的随机数据，
方法是使用numpy操作手动实现前后向通过网络：

.. includenodoc:: /beginner/examples_tensor/two_layer_net_numpy.py


PyTorch: Tensors
----------------

Numpy是一个很好的框架，但它不能使用GPU加速其数值计算。对于现代的深层神经网络，
GPU通常提供50倍或更高的速度，因此不幸的是，numpy不足以满足现代深度学习的需要。

这里我们介绍最基础的PyTorch概念：**张量(Tensor)** 。PyTorch张量在概念上与numpy数组相同：
一个Tensor是一个n维数组，PyTorch提供了许多在这些张量上操作的函数。
在幕后，张量可以跟踪计算图和梯度，但它们也是科学计算的通用工具。

与Numpy不同的是，PyTorch张量可以利用GPU加速它们的数值计算。
要在GPU上运行PyTorch张量，只需将其转换为新的数据类型即可。

在这里，我们使用PyTorch张量对随机数据进行两层网络拟合。
与上面的numpy示例一样，我们需要手动实现通过网络向前和向后传递的内容:

.. includenodoc:: /beginner/examples_tensor/two_layer_net_tensor.py


自动梯度
========

PyTorch: Tensors 和 autograd
-------------------------------

在上面的例子中，我们必须手动实现我们的神经网络的前向、后向传播过程。
对于一个小的两层网络来说，手动实现反向传递并不是什么大问题，
但是对于大型复杂网络来说，它很快就会变得令人害怕的(hairy)。

值得庆幸的是，我们可以使用自动微分
(`automatic differentiation <https://en.wikipedia.org/wiki/Automatic_differentiation>`__)
来自动计算神经网络中的反向传播。
PyTorch中的 **autograd** 包提供了这个功能。使用autograd时，
网络的前向传播过将定义一个计算图(**computational graph**)；图中的节点将是张量，
边将是从输入张量产生输出张量的函数。
然后，通过这个图进行反向传播，您可以轻松地计算梯度。

这听起来很复杂，在实践中很容易使用。每个张量表示计算图中的一个节点。
如果 ``x`` 是具有 ``x.requires_grad=True`` 状态的张量，
则 ``x.grad`` 是另一个张量，它持有 ``x`` 相对于某个标量值的梯度。

在这里，我们使用PyTorch张量和自动梯度来实现我们的两层网络；
现在我们不再需要手动实现通过网络的反向传递:

.. includenodoc:: /beginner/examples_autograd/two_layer_net_autograd.py

PyTorch: 定义一个新的 autograd 函数
----------------------------------------

在这种情况下，每个原始的 自动梯度算子(autograd operator) 实际上是两个作用于张量的函数。
**forward** 函数从输入张量计算输出张量。**backward** 函数接收输出张量相对于某个标量值的梯度，
并计算输入张量相对于该标量值的梯度。

在PyTorch中，我们可以通过定义 ``torch.autograd.Function`` 的子类来
轻松地定义我们自己的自动梯度算子 并 实现 ``forward`` 和 ``backward`` 函数。
然后，我们可以使用新的自动梯度算子，方法是构造一个类实例并像函数一样调用它，
传递包含输入数据的张量。

在这个例子中，我们定义了自定义的自动梯度函数来执行relu非线性，并使用它来实现我们的两层网络:

.. includenodoc:: /beginner/examples_autograd/two_layer_net_custom_function.py

TensorFlow: 静态计算图
-------------------------

PyTorch Autograd看起来很像TensorFlow：在这两个框架中，我们定义了一个计算图，
并使用自动微分来计算梯度。两者最大的区别是TensorFlow的计算图是 **静态的(static)** ，
PyTorch使用 **动态(dynamic)** 计算图。

在TensorFlow中，我们定义一次计算图，然后一次又一次地执行相同的图，
可能会将不同的输入数据输入到图中。
在PyTorch中，每一次前向传播过程都定义一个新的计算图。

静态图很好，因为您可以预先优化它；例如，一个框架可能决定融合一些图节点操作以提高效率，
或者想出一种在多个GPU或多台机器上分配图上计算节点的策略。如果您一次又一次地重用相同的图，
那么这个潜在的代价高昂的预先优化可以被摊还，因为相同的图会一次又一次地重复运行。

静态图和动态图不同的一个方面是控制流(control flow)。对于某些模型，我们可能希望对每个数据点执行不同的计算；
例如，对于每个数据点，可能会对不同的时间步骤展开递归网络；这种展开可以作为一个循环来实现。
对于静态图，循环构造需要是图的一部分；因此，TensorFlow提供了诸如 ``tf.scan`` 之类的操作符，
用于将循环嵌入到图中。对于动态图，情况更简单：因为我们为每个示例动态构建图，
所以我们可以使用正常的命令式流控制来执行对每个输入不同的计算。

为了与上面的PyTorch Autograd的示例作对比，这里我们使用TensorFlow来拟合一个简单的两层网络:

.. includenodoc:: /beginner/examples_autograd/tf_two_layer_net.py

`nn` 模块
===========

PyTorch: nn
-----------

计算图和自动梯度是定义复杂算子和自动获取导数的一个非常强大的paradigm；
然而，对于大型神经网络来说，raw autograd 可能有点过于低级。

在建立神经网络时，我们经常会考虑将计算组织成 **层(layers)** ，其中有些具有可学习的参数(**learnable parameters**)，
在学习过程中会进行优化。

在TensorFlow中，
`Keras <https://github.com/fchollet/keras>`__,
`TensorFlow-Slim <https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim>`__,
和 `TFLearn <http://tflearn.org/>`__ 等包提供了比原始计算图更高层次的抽象，这对于构建神经网络非常有用。

在PyTorch中，``nn`` 包也有同样的用途。``nn`` 包定义了一组 **模块(Modules)** ，它们大致相当于神经网络的层。
模块接收输入张量并计算输出张量，同时也可以保持内部状态，例如包含可学习参数的张量。
``nn`` 包还定义了一组有用的损失函数，这些函数是训练神经网络时常用的。

在本例中，我们使用nn包来实现我们的两层网络：

.. includenodoc:: /beginner/examples_nn/two_layer_net_nn.py

PyTorch: optim
--------------

到目前为止，我们已经通过手动修改包含可学习参数的张量来更新模型的权重(使用 ``torch.no_grad()`` 
或 ``.data`` ，以避免在autograd中跟踪历史记录)。对于像随机梯度下降这样的简单优化算法来说，
这并不是一个巨大的负担，但在实践中，我们经常使用更复杂的优化器，
如AdaGrad、RMSProp、Adam 等来训练神经网络。

PyTorch中的 ``optim`` package 抽象了优化算法的思想，并提供了常用优化算法的实现。

在这个例子中，我们将像前面一样使用 ``nn`` 包来定义我们的模型，但是我们将
使用 ``optim`` package提供的 Adam 算法来优化模型:

.. includenodoc:: /beginner/examples_nn/two_layer_net_optim.py

PyTorch: 自定义 nn 模块
--------------------------

有时，您可能希望指定比现有模块序列更复杂的模型；在这些情况下，
您可以通过定义 ``nn.Module`` 的子类和定义一个 ``forward`` 来定义您自己的模块，
它接收输入张量并使用其他模块或对张量的其他自动梯度算子生成输出张量。

在本例中，我们将我们的两层网络实现为自定义模块子类:

.. includenodoc:: /beginner/examples_nn/two_layer_net_module.py

PyTorch: 控制流 + 权重共享
--------------------------------------

作为动态图和权重共享的一个例子，我们实现了一个非常奇怪的模型：一个完全连接的ReLU网络，
它在每个前向通路上选择一个介于1到4之间的随机数，并使用那许多隐藏层，
重复使用相同的权重多次计算最内部的隐藏层。

对于这个模型，我们可以使用普通的Python流控制来实现循环，我们可以通过在定义前向传递时
多次重用相同的模块来实现最内部层之间的权重共享。

我们可以很容易地将这个模型实现为一个模块子类:

.. includenodoc:: /beginner/examples_nn/dynamic_net.py


.. _examples-download:

示例
========

您可以在这里单独浏览和下载上面的每个示例代码。

Tensors
-------

.. toctree::
   :maxdepth: 2

   /beginner/examples_tensor/two_layer_net_numpy
   /beginner/examples_tensor/two_layer_net_tensor

.. galleryitem:: /beginner/examples_tensor/two_layer_net_numpy.py

.. galleryitem:: /beginner/examples_tensor/two_layer_net_tensor.py

.. raw:: html

    <div style='clear:both'></div>

Autograd
--------

.. toctree::
   :maxdepth: 2

   /beginner/examples_autograd/two_layer_net_autograd
   /beginner/examples_autograd/two_layer_net_custom_function
   /beginner/examples_autograd/tf_two_layer_net


.. galleryitem:: /beginner/examples_autograd/two_layer_net_autograd.py

.. galleryitem:: /beginner/examples_autograd/two_layer_net_custom_function.py

.. galleryitem:: /beginner/examples_autograd/tf_two_layer_net.py

.. raw:: html

    <div style='clear:both'></div>

`nn` module
-----------

.. toctree::
   :maxdepth: 2

   /beginner/examples_nn/two_layer_net_nn
   /beginner/examples_nn/two_layer_net_optim
   /beginner/examples_nn/two_layer_net_module
   /beginner/examples_nn/dynamic_net


.. galleryitem:: /beginner/examples_nn/two_layer_net_nn.py

.. galleryitem:: /beginner/examples_nn/two_layer_net_optim.py

.. galleryitem:: /beginner/examples_nn/two_layer_net_module.py

.. galleryitem:: /beginner/examples_nn/dynamic_net.py

.. raw:: html

    <div style='clear:both'></div>
