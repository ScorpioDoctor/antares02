
ONNX 教程
==================

本教程将介绍如何使用ONNX将从PyTorch导出的神经风格迁移模型(neural style transfer model)转换为Apple CoreML格式。
这将使您可以轻松地在Apple设备上运行深度学习模型，在这种情况下，您可以从摄像机中实时运行流。

什么是 ONNX?
-------------

开放式神经网络交换(ONNX，Open Neural Network Exchange)是一种表示深度学习模型的开放格式。
使用ONNX，人工智能开发人员可以更容易地在最先进的工具之间移动模型，并选择最适合他们的组合。
ONNX是由一个合作伙伴社区开发和支持的。您可以通过 `onnx.ai <http://onnx.ai/>`_ 了解更多关于ONNX和支持哪些工具的信息。

教程预览
-----------------

这个教程将带你完成以下四步:


#. `下载 (或 训练) PyTorch 神经风格迁移模型`_
#. `把 PyTorch 模型转换为 ONNX 模型`_
#. `把 ONNX 模型 转换为 CoreML 模型`_
#. `在一个风格迁移 IOS app中运行 CoreML 模型`_

准备环境
-------------------------

为了避免与本地包的冲突，我们将在虚拟环境中工作。在本教程中，我们也使用Python3.6，但其他版本也可以工作。

.. code-block:: python

   python3.6 -m venv venv
   source ./venv/bin/activate


你需要安装 pytorch 和 onnx->coreml 的转换器:

.. code-block:: bash

   pip install torchvision onnx-coreml


如果你想在IPhone上运行iOS风格迁移APP，你还需要安装XCode。您也可以在Linux中转换模型，但是要运行IOS APP本身，您需要一个Mac。

下载 (或 训练) PyTorch 神经风格迁移模型
-------------------------------------------------

在本教程中，我们将使用在 https://github.com/pytorch/examples/tree/master/fast_neural_style 中
的风格迁移模型。如果你想用别的模型，那就跳过这个小节吧！

这些模型是为了在静止图像上应用风格迁移，确实没有优化到足够快进行视频处理。
然而，如果我们把分辨率降到足够低，也可以很好地处理视频。

让我们下载这个模型:

.. code-block:: bash

   git clone https://github.com/pytorch/examples
   cd examples/fast_neural_style


如果您想亲自训练这些模型，那么您刚才克隆的 pytorch/examples 库就有更多关于如何做到这一点的信息。
现在，我们将使用存储库提供的脚本下载经过预先训练的模型:

.. code-block:: bash

   ./download_saved_models.sh


此脚本下载经过预先训练的PyTorch模型，并将它们放入 ``saved_models`` 文件夹中。现在您的目录中应该有4个文件：
``candy.pth``\ , ``mosaic.pth``\ , ``rain_princess.pth`` 和 ``udnie.pth`` 。

把 PyTorch 模型转换为 ONNX 模型
-----------------------------------------

现在，我们已经将预先训练好的PyTorch模型作为 ``.pth`` 文件放在 ``saved_models`` 文件夹中，
我们需要将它们转换为ONNX格式。模型定义在我们之前克隆的 pytorch/examples 库中，
通过几行python，我们可以将其导出到ONNX。在本例中，我们将调用 ``torch.onnx._export``\ ，
而不是实际运行神经网络，它随PyTorch作为API提供，用于从PyTorch直接导出ONNX格式的模型。
但是，在这种情况下，我们甚至不需要这样做，因为已经存在 ``neural_style/neural_style.py`` 脚本，
它将为我们做到这一点。如果您想将该脚本应用于其他模型，也可以查看它。

从PyTorch导出ONNX格式本质上是跟踪您的神经网络，因此这个API调用将在内部运行网络上的“虚拟数据”，以便生成Graph。为此，
它需要一个输入图像来应用风格迁移，而这种迁移可以只是一个空白图像。
但是，该图像的像素大小非常重要，因为这将是导出风格迁移模型的大小。
为了获得良好的性能，我们将使用250x540的分辨率。如果你不太关心FPS，而更关心风格转换的质量，那么可以自由地采取更大的解决方案。

让我们使用 `ImageMagick <http://www.imagemagick.org/>`_ 来创建一个指定分辨率的空图像吧:

.. code-block:: bash

   convert -size 250x540 xc:white png24:dummy.jpg


然后使用它导出 PyTorch 模型:

.. code-block:: bash

   python ./neural_style/neural_style.py eval --content-image dummy.jpg --output-image dummy-out.jpg --model ./saved_models/candy.pth --cuda 0 --export_onnx ./saved_models/candy.onnx
   python ./neural_style/neural_style.py eval --content-image dummy.jpg --output-image dummy-out.jpg --model ./saved_models/udnie.pth --cuda 0 --export_onnx ./saved_models/udnie.onnx
   python ./neural_style/neural_style.py eval --content-image dummy.jpg --output-image dummy-out.jpg --model ./saved_models/rain_princess.pth --cuda 0 --export_onnx ./saved_models/rain_princess.onnx
   python ./neural_style/neural_style.py eval --content-image dummy.jpg --output-image dummy-out.jpg --model ./saved_models/mosaic.pth --cuda 0 --export_onnx ./saved_models/mosaic.onnx


这一步完成之后, 你应该有4个文件： ``candy.onnx``\ , ``mosaic.onnx``\ , ``rain_princess.onnx`` 和 ``udnie.onnx``\ , 它们是从对应的 ``.pth`` 文件创造出来的。

把 ONNX 模型 转换为 CoreML 模型
----------------------------------------

现在我们有了ONNX模型，我们可以将它们转换为CoreML模型，以便在Apple设备上运行它们。
为此，我们使用我们之前安装的onnx-coreml转换器。
转换器自带了一个 ``convert-onnx-to-coreml`` 脚本，上面的安装步骤添加到我们的路径中。不幸的是，这将不适用于我们，
因为我们需要将网络的输入和输出标记为图像，虽然转换器支持这一点，但只有在从python调用转换器时才支持它。

查看风格迁移模型(例如，在像  `Netron <https://github.com/lutzroeder/Netron>`_\ 这样的应用程序中打开.onnx文件)，
我们看到输入被命名为 '0' ，输出命名为 '186' 。这些只是PyTorch分配的数字ID。我们需要将这些标记为图像。

因此我们创建一个小的Python文件，命名为 ``onnx_to_coreml.py`` 。 这可以通过使用 touch 命令创建，并使用您最喜欢的编辑器进行编辑，以添加以下代码行。

.. code-block:: python

   import sys
   from onnx import onnx_pb
   from onnx_coreml import convert

   model_in = sys.argv[1]
   model_out = sys.argv[2]

   model_file = open(model_in, 'rb')
   model_proto = onnx_pb.ModelProto()
   model_proto.ParseFromString(model_file.read())
   coreml_model = convert(model_proto, image_input_names=['0'], image_output_names=['186'])
   coreml_model.save(model_out)


现在我们运行这个文件:

.. code-block:: bash

   python onnx_to_coreml.py ./saved_models/candy.onnx ./saved_models/candy.mlmodel
   python onnx_to_coreml.py ./saved_models/udnie.onnx ./saved_models/udnie.mlmodel
   python onnx_to_coreml.py ./saved_models/rain_princess.onnx ./saved_models/rain_princess.mlmodel
   python onnx_to_coreml.py ./saved_models/mosaic.onnx ./saved_models/mosaic.mlmodel


现在应该有4个 CoreML 模型在你的 ``saved_models`` 目录: ``candy.mlmodel``\ , ``mosaic.mlmodel``\ , ``rain_princess.mlmodel`` 和 ``udnie.mlmodel`` 。

在一个风格迁移 IOS app中运行 CoreML 模型
-------------------------------------------------

这个repository(也就是您目前正在阅读的README.md)包含一个IOS应用程序，可以在手机摄像头上的实时相机流上运行CoreML风格迁移模型。让我们克隆repository::

.. code-block:: bash

   git clone https://github.com/onnx/tutorials


然后在 XCode 中打开 ``tutorials/examples/CoreML/ONNXLive/ONNXLive.xcodeproj``  工程.
我们建议使用XCode 9.3和iPhoneX。在旧设备或XCode版本上可能会出现问题。

在 ``Models/`` 文件夹中, 该项目包含一些 .mlmodel 文件。我们将用我们刚刚创建的模型来取代它们。

然后在你的iPhone上运行这个应用程序app，你就都准备好了。点击屏幕切换模型。

总结
----------

我们希望本教程向您概述ONNX是关于什么的，以及如何使用它在框架之间转换神经网络，在这个案例中，从PyTorch迁移到CoreML的是神经风格迁移模型。

可以自由地尝试这些步骤，并在您自己的模型上测试它们。如果您遇到任何问题或想提供反馈，请通知我们。我们想听听你的想法。
