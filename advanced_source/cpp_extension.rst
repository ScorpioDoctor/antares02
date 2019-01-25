自定义 C++ 和 CUDA 扩展
==============================
**翻译者**: `Antares博士 <http://www.studyai.com/antares>`_


PyTorch提供了大量与神经网络、任意张量代数、数据管理等相关的操作。
但是，您可能仍然需要一个更定制化的操作。例如，您可能希望使用您在论文中发现的新的激活函数，
或者实现您在研究中开发的操作。

在PyTorch中集成这种自定义操作的最简单方法是用Python编写它, 通过扩展 :class:`Function` 和 :class:`Module` 
来实现，`如本文所述 <https://pytorch.org/docs/master/notes/extending.html>`_ 。
这为您提供了自动微分的全部功能(不需要编写派生函数)以及常用的Python代码。
但是，有时您的操作在C++中实现会更好。例如，您的代码可能需要非常快，因为它在您的模型中被频繁调用，
或者即使很少调用也非常昂贵。另一个合理的原因是它依赖于或与其他C或C++库交互。
为了解决这种情况，PyTorch提供了一种编写自定义C++扩展的非常简单的方法。

C+扩展是我们开发的一种机制，它允许用户(您)创建PyTorch操作符，在源代码之外定义，即与PyTorch后端是分开的。
这种方法不同于原生(native) PyTorch操作的实现方式。C++扩展旨在为您节省与PyTorch后端集成操作相关的大量样板代码，
同时为您的基于PyTorch的项目提供高度的灵活性。然而，一旦将操作定义为C++扩展，
将其转换为native PyTorch函数在很大程度上是一个代码组织问题，如果您决定向上游贡献您的操作算法，
您可以在此之后处理这个问题。


动机与实例
----------------------

本说明的其余部分将介绍编写和使用C++(和CUDA)扩展的实用示例。如果您正在被追赶，或者有人会解雇您，
如果您无法在一天结束之前完成操作，您可以跳过这一节，直接进入下一节的实现细节。

假设您已经提出了一种新的递归单元，您发现它具有比最先进的性能优越的特性。
这个递归单元类似于LSTM，但其不同之处在于它没有忘记门(*forget gate*)，
并且使用指数线性单元(ELU)作为内部激活函数。
因为这个单元永远不会忘记，我们称之为 *LLTM* ，或长期记忆单元(*Long-Long-Term-Memory* unit)。

LLTMs与普通LSTMs的两种不同之处足以使我们无法为我们的目的配置PyTorch的 ``LSTMCell`` ，
因此我们必须创建一个自定义cell。这方面的第一个也是最简单的方法-在所有情况下都可能是一个很好的第一步-
是用Python在普通PyTorch中实现我们想要的功能。为此，我们需要继承 :class:`torch.nn.Module` 类
并实现 LLTM 的向前传递。这个看起来是这样的::

  class LLTM(torch.nn.Module):
      def __init__(self, input_features, state_size):
          super(LLTM, self).__init__()
          self.input_features = input_features
          self.state_size = state_size
          # 3 * state_size for input gate, output gate and candidate cell gate.
          # input_features + state_size because we will multiply with [input, h].
          self.weights = torch.nn.Parameter(
              torch.empty(3 * state_size, input_features + state_size))
          self.bias = torch.nn.Parameter(torch.empty(3 * state_size))
          self.reset_parameters()

      def reset_parameters(self):
          stdv = 1.0 / math.sqrt(self.state_size)
          for weight in self.parameters():
              weight.data.uniform_(-stdv, +stdv)

      def forward(self, input, state):
          old_h, old_cell = state
          X = torch.cat([old_h, input], dim=1)

          # Compute the input, output and candidate cell gates with one MM.
          gate_weights = F.linear(X, self.weights, self.bias)
          # Split the combined gate weight matrix into its components.
          gates = gate_weights.chunk(3, dim=1)

          input_gate = F.sigmoid(gates[0])
          output_gate = F.sigmoid(gates[1])
          # Here we use an ELU instead of the usual tanh.
          candidate_cell = F.elu(gates[2])

          # Compute the new cell state.
          new_cell = old_cell + candidate_cell * input_gate
          # Compute the new hidden state and output.
          new_h = F.tanh(new_cell) * output_gate

          return new_h, new_cell

定义好这个子类，我们就能以如下方法使用它啦::

  import torch

  X = torch.randn(batch_size, input_features)
  h = torch.randn(batch_size, state_size)
  C = torch.randn(batch_size, state_size)

  rnn = LLTM(input_features, state_size)

  new_h, new_C = rnn(X, (h, C))

当然，如果有可能和可信的话，您应该使用这种方法来扩展PyTorch。由于PyTorch为CPU和GPU的操作提供了高度优化的实现，
由 `NVIDIA cuDNN <https://developer.nvidia.com/cudnn>`_ 、
`Intel MKL <https://software.intel.com/en-us/mkl>`_  或 
`NNPACK <https://github.com/Maratyszcza/NNPACK>`_ 
等库提供动力，因此像上面这样的PyTorch代码通常足够快。
不过，我们也可以看到，为何在某些情况下，性能仍有进一步改善的空间。最明显的原因是PyTorch不知道您正在实现的算法。
它只知道用于组成算法的单个操作。因此，PyTorch必须一个接一个地执行您的操作。由于对操作的实现(或内核)的每个单独调用
(可能涉及启动CUDA内核)都有一定的开销，因此在许多函数调用中，这种开销可能会变得很大。
此外，运行我们代码的Python解释器本身可以减慢我们的程序。

因此，加快速度的一个明确方法是用C++(或CUDA)重写某些部分，并融合特定的操作组。
融合意味着将许多函数的实现合并到一个函数中，
这些函数可以从更少的内核启动和其他优化中获益，我们可以通过提高全局数据流的可见性来执行这些优化。

让我们看看如何使用C++扩展来实现LLTM的融合版本。我们将从使用普通C++编写它开始，使用支持PyTorch后端的 
`Aten <https://github.com/zdevito/ATen>`_  库，看看它如何容易地让我们翻译Python代码。
然后，我们将通过把模型的部分转移到CUDA内核，从而更快地加快速度，从而从GPU提供的大规模并行性中获益。

写一个C++扩展
-----------------------

C++ 扩展有两种形式：可以使用 :mod:`setuptools` “提前(ahead of time)” 构建它们，
也可以通过 :func:`torch.utils.cpp_extension.load` “即时(just in time)”构建它们。
我们将从第一种方法开始，稍后讨论后者。

使用 :mod:`setuptools` 构建
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

对于“提前”构建这种方法，我们通过编写 ``setup.py`` 脚本来构建我们的C++扩展，
它使用setuptools编译我们的C++代码。对于LLTM，这看起来很简单::

  from setuptools import setup
  from torch.utils.cpp_extension import CppExtension, BuildExtension

  setup(name='lltm',
        ext_modules=[CppExtension('lltm', ['lltm.cpp'])],
        cmdclass={'build_ext': BuildExtension})


在上面这段代码中, :class:`CppExtension` 类是对 :class:`setuptools.Extension` 类的一个方便的封装，
只需传入正确的头文件包含路经并把扩展语言名称指定为 C++ 。与上面代码等价的，使用 :mod:`setuptools` 的代码也很简单，
如下所示::

  setuptools.Extension(
     name='lltm',
     sources=['lltm.cpp'],
     include_dirs=torch.utils.cpp_extension.include_paths(),
     language='c++')

:class:`BuildExtension` 类执行一些所需的配置步骤和检查，并在混合C++/CUDA扩展的情况下管理混合编译。
这就是我们现在真正需要了解的关于构建C++扩展的所有知识！现在让我们来看看我们的C++扩展的实现，
这将进入 ``lltm.cpp`` 文件的编写。

写一个C++操作(Op)
^^^^^^^^^^^^^^^^^^

让我们开始在C++中实现LLTM！向后传递需要的一个函数是sigmoid的导数。
这是一段足够小的代码，可以讨论编写C++扩展时可用的总体环境：


.. code-block:: cpp

  #include <torch/torch.h>

  #include <iostream>

  at::Tensor d_sigmoid(at::Tensor z) {
    auto s = at::sigmoid(z);
    return (1 - s) * s;
  }

``<torch/torch.h>`` 是包含编写C++扩展所需的所有PyTorch bits的一站式头文件。它包括:

- ATen 库, 是用于张量计算的主要API,
- `pybind11 <https://github.com/pybind/pybind11>`_, 是我们曾样为我们的C++代码创建Python绑定(bindings),
- 若干头文件用于管理ATen和pybindll之间的交互的细节。

:func:`d_sigmoid` 函数的实现演示如何使用ATen API。PyTorch的张量和变量接口是从ATen库自动生成的，
因此我们可以或多或少地将Python实现1：1转换为C++。我们所有计算的主要数据类型都是 :class:`at::Tensor` 。
`这里 <https://pytorch.org/cppdocs/api/classat_1_1_tensor.html>`_ 可以检查其完整的API。
还请注意，我们可以包括 ``<iostream>`` 或 *任何其他C或C++头文件* --
我们可以使用C++11的全部能力。

前向传递过程
******************

接下来我们可以把我们整个的前向传递弄到 C++ 中:

.. code-block:: cpp

  #include <vector>

  std::vector<at::Tensor> lltm_forward(
      at::Tensor input,
      at::Tensor weights,
      at::Tensor bias,
      at::Tensor old_h,
      at::Tensor old_cell) {
    auto X = at::cat({old_h, input}, /*dim=*/1);

    auto gate_weights = at::addmm(bias, X, weights.transpose(0, 1));
    auto gates = gate_weights.chunk(3, /*dim=*/1);

    auto input_gate = at::sigmoid(gates[0]);
    auto output_gate = at::sigmoid(gates[1]);
    auto candidate_cell = at::elu(gates[2], /*alpha=*/1.0);

    auto new_cell = old_cell + candidate_cell * input_gate;
    auto new_h = at::tanh(new_cell) * output_gate;

    return {new_h,
            new_cell,
            input_gate,
            output_gate,
            candidate_cell,
            X,
            gate_weights};
  }


反向传递过程
*************

C++扩展API目前没有为我们提供自动生成后向传递(backwards)函数的方法。因此，我们还必须实现LLTM的后向传递，
它计算损失相对于前向传递的每个输入的导数。最后，我们将前向函数和后向函数都设置为 :class:`torch.autograd.Function` ，
以创建一个很好的Python绑定。后向函数涉及的内容稍微多一些，因此我们将不再深入研究代码
(如果您感兴趣，`Alex Graves' thesis <http://www.cs.toronto.edu/~graves/phd.pdf>`_ 
是一个很好的读物，以获得更多有关这方面的信息):

.. code-block:: cpp

  // tanh'(z) = 1 - tanh^2(z)
  at::Tensor d_tanh(at::Tensor z) {
    return 1 - z.tanh().pow(2);
  }

  // elu'(z) = relu'(z) + { alpha * exp(z) if (alpha * (exp(z) - 1)) < 0, else 0}
  at::Tensor d_elu(at::Tensor z, at::Scalar alpha = 1.0) {
    auto e = z.exp();
    auto mask = (alpha * (e - 1)) < 0;
    return (z > 0).type_as(z) + mask.type_as(z) * (alpha * e);
  }

  std::vector<at::Tensor> lltm_backward(
      at::Tensor grad_h,
      at::Tensor grad_cell,
      at::Tensor new_cell,
      at::Tensor input_gate,
      at::Tensor output_gate,
      at::Tensor candidate_cell,
      at::Tensor X,
      at::Tensor gate_weights,
      at::Tensor weights) {
    auto d_output_gate = at::tanh(new_cell) * grad_h;
    auto d_tanh_new_cell = output_gate * grad_h;
    auto d_new_cell = d_tanh(new_cell) * d_tanh_new_cell + grad_cell;

    auto d_old_cell = d_new_cell;
    auto d_candidate_cell = input_gate * d_new_cell;
    auto d_input_gate = candidate_cell * d_new_cell;

    auto gates = gate_weights.chunk(3, /*dim=*/1);
    d_input_gate *= d_sigmoid(gates[0]);
    d_output_gate *= d_sigmoid(gates[1]);
    d_candidate_cell *= d_elu(gates[2]);

    auto d_gates =
        at::cat({d_input_gate, d_output_gate, d_candidate_cell}, /*dim=*/1);

    auto d_weights = d_gates.t().mm(X);
    auto d_bias = d_gates.sum(/*dim=*/0, /*keepdim=*/true);

    auto d_X = d_gates.mm(weights);
    const auto state_size = grad_h.size(1);
    auto d_old_h = d_X.slice(/*dim=*/1, 0, state_size);
    auto d_input = d_X.slice(/*dim=*/1, state_size);

    return {d_old_h, d_input, d_weights, d_bias, d_old_cell};
  }

绑定到 Python
^^^^^^^^^^^^^^^^^

使用C++和ATen库编写操作(OPs)之后，可以使用 pybind11 以非常简单的方式将C++函数或类绑定到Python中。
有关PyTorch C++ 扩展的这一部分的问题或问题将主要由 
`pybind11 文档 <http://pybind11.readthedocs.io/en/master/>`_ 
解决。

对于我们的扩展, 必要的绑定代码只有四行:

.. code-block:: cpp

  PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &lltm_forward, "LLTM forward");
    m.def("backward", &lltm_backward, "LLTM backward");
  }

这里要注意的一点是宏 ``TORCH_EXTENSION_NAME`` 。torch 扩展构建将其定义为您在 ``setup.py`` 脚本中给出的扩展名。
在这种情况下，``TORCH_EXTENSION_NAME`` 的值将是 “lltm” 。这是为了避免在两个地方(构建脚本和C++代码)维护扩展的名称，
因为两者之间的不匹配可能导致棘手和难以跟踪的问题。

使用你的扩展
^^^^^^^^^^^^^^^^^^^^

我们现在准备在PyTorch中导入我们的扩展。此时，您的目录结构可能如下所示::

  pytorch/
    lltm-extension/
      lltm.cpp
      setup.py

现在, 运行 ``python setup.py install`` 来构建和安装你的扩展. 这应该看起来是这样子的::

  running install
  running bdist_egg
  running egg_info
  writing lltm.egg-info/PKG-INFO
  writing dependency_links to lltm.egg-info/dependency_links.txt
  writing top-level names to lltm.egg-info/top_level.txt
  reading manifest file 'lltm.egg-info/SOURCES.txt'
  writing manifest file 'lltm.egg-info/SOURCES.txt'
  installing library code to build/bdist.linux-x86_64/egg
  running install_lib
  running build_ext
  building 'lltm' extension
  gcc -Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes -fPIC -I~/local/miniconda/lib/python3.6/site-packages/torch/lib/include -I~/local/miniconda/lib/python3.6/site-packages/torch/lib/include/TH -I~/local/miniconda/lib/python3.6/site-packages/torch/lib/include/THC -I~/local/miniconda/include/python3.6m -c lltm.cpp -o build/temp.linux-x86_64-3.6/lltm.o -DTORCH_EXTENSION_NAME=lltm -std=c++11
  cc1plus: warning: command line option ‘-Wstrict-prototypes’ is valid for C/ObjC but not for C++
  g++ -pthread -shared -B ~/local/miniconda/compiler_compat -L~/local/miniconda/lib -Wl,-rpath=~/local/miniconda/lib -Wl,--no-as-needed -Wl,--sysroot=/ build/temp.linux-x86_64-3.6/lltm.o -o build/lib.linux-x86_64-3.6/lltm.cpython-36m-x86_64-linux-gnu.so
  creating build/bdist.linux-x86_64/egg
  copying build/lib.linux-x86_64-3.6/lltm_cuda.cpython-36m-x86_64-linux-gnu.so -> build/bdist.linux-x86_64/egg
  copying build/lib.linux-x86_64-3.6/lltm.cpython-36m-x86_64-linux-gnu.so -> build/bdist.linux-x86_64/egg
  creating stub loader for lltm.cpython-36m-x86_64-linux-gnu.so
  byte-compiling build/bdist.linux-x86_64/egg/lltm.py to lltm.cpython-36.pyc
  creating build/bdist.linux-x86_64/egg/EGG-INFO
  copying lltm.egg-info/PKG-INFO -> build/bdist.linux-x86_64/egg/EGG-INFO
  copying lltm.egg-info/SOURCES.txt -> build/bdist.linux-x86_64/egg/EGG-INFO
  copying lltm.egg-info/dependency_links.txt -> build/bdist.linux-x86_64/egg/EGG-INFO
  copying lltm.egg-info/top_level.txt -> build/bdist.linux-x86_64/egg/EGG-INFO
  writing build/bdist.linux-x86_64/egg/EGG-INFO/native_libs.txt
  zip_safe flag not set; analyzing archive contents...
  __pycache__.lltm.cpython-36: module references __file__
  creating 'dist/lltm-0.0.0-py3.6-linux-x86_64.egg' and adding 'build/bdist.linux-x86_64/egg' to it
  removing 'build/bdist.linux-x86_64/egg' (and everything under it)
  Processing lltm-0.0.0-py3.6-linux-x86_64.egg
  removing '~/local/miniconda/lib/python3.6/site-packages/lltm-0.0.0-py3.6-linux-x86_64.egg' (and everything under it)
  creating ~/local/miniconda/lib/python3.6/site-packages/lltm-0.0.0-py3.6-linux-x86_64.egg
  Extracting lltm-0.0.0-py3.6-linux-x86_64.egg to ~/local/miniconda/lib/python3.6/site-packages
  lltm 0.0.0 is already the active version in easy-install.pth

  Installed ~/local/miniconda/lib/python3.6/site-packages/lltm-0.0.0-py3.6-linux-x86_64.egg
  Processing dependencies for lltm==0.0.0
  Finished processing dependencies for lltm==0.0.0

关于编译器的一个小提示：由于ABI版本控制问题，用于构建C++扩展的编译器必须是ABI兼容的编译器，PyTorch是用它构建的。
实际上，这意味着您必须在Linux上使用GCC版本4.9及以上版本。对于Ubuntu16.04和其他最近的Linux发行版，
这应该已经是默认的编译器了。在MacOS上，您必须使用clang(它没有任何ABI版本控制问题)。
在最坏的情况下，您可以使用编译器从源代码构建PyTorch，然后使用相同的编译器构建扩展。

一旦构建了扩展，就可以使用 ``setup.py`` 脚本中指定的名称在Python中导入它。只需确保先 ``import torch`` ，
因为这将解析动态链接器必须看到的一些符号::

  In [1]: import torch
  In [2]: import lltm
  In [3]: lltm.forward
  Out[3]: <function lltm.PyCapsule.forward>

如果我们在函数或模块上调用 ``help()`` ，我们可以看到它的签名与我们的C++代码匹配::

  In[4] help(lltm.forward)
  forward(...) method of builtins.PyCapsule instance
      forward(arg0: at::Tensor, arg1: at::Tensor, arg2: at::Tensor, arg3: at::Tensor, arg4: at::Tensor) -> List[at::Tensor]

      LLTM forward

由于我们现在可以从Python调用我们的C++函数，所以我们可以用 :class:`torch.autograd.Function` 
和 :class:`torch.nn.Module`  封装它们，使它们成为PyTorch的头等公民::

  import math
  import torch

  # 导入模型!
  import lltm

  class LLTMFunction(torch.autograd.Function):
      @staticmethod
      def forward(ctx, input, weights, bias, old_h, old_cell):
          outputs = lltm.forward(input, weights, bias, old_h, old_cell)
          new_h, new_cell = outputs[:2]
          variables = outputs[1:] + [weights]
          ctx.save_for_backward(*variables)

          return new_h, new_cell

      @staticmethod
      def backward(ctx, grad_h, grad_cell):
          outputs = lltm.backward(
              grad_h.contiguous(), grad_cell.contiguous(), *ctx.saved_variables)
          d_old_h, d_input, d_weights, d_bias, d_old_cell = outputs
          return d_input, d_weights, d_bias, d_old_h, d_old_cell


  class LLTM(torch.nn.Module):
      def __init__(self, input_features, state_size):
          super(LLTM, self).__init__()
          self.input_features = input_features
          self.state_size = state_size
          self.weights = torch.nn.Parameter(
              torch.empty(3 * state_size, input_features + state_size))
          self.bias = torch.nn.Parameter(torch.empty(3 * state_size))
          self.reset_parameters()

      def reset_parameters(self):
          stdv = 1.0 / math.sqrt(self.state_size)
          for weight in self.parameters():
              weight.data.uniform_(-stdv, +stdv)

      def forward(self, input, state):
          return LLTMFunction.apply(input, self.weights, self.bias, *state)

性能比较
**********************

现在我们可以使用和调用PyTorch的C++代码了，我们可以运行一个小的基准测试，看看我们用C++重写我们的OP获得了多少性能。
我们将前后运行LLTM几次，并测量持续时间::

  import torch

  batch_size = 16
  input_features = 32
  state_size = 128

  X = torch.randn(batch_size, input_features)
  h = torch.randn(batch_size, state_size)
  C = torch.randn(batch_size, state_size)

  rnn = LLTM(input_features, state_size)

  forward = 0
  backward = 0
  for _ in range(100000):
      start = time.time()
      new_h, new_C = rnn(X, (h, C))
      forward += time.time() - start

      start = time.time()
      (new_h.sum() + new_C.sum()).backward()
      backward += time.time() - start

  print('Forward: {:.3f} us | Backward {:.3f} us'.format(forward * 1e6/1e5, backward * 1e6/1e5))

如果我们在这篇文章的开头使用我们用纯Python编写的原始LLTM运行这段代码，我们将得到以下数字(在我的机器上)::

  Forward: 506.480 us | Backward 444.694 us

使用我们的新的C++版本::

  Forward: 349.335 us | Backward 443.523 us

我们已经可以看到一个显著的加速前向函数(超过30%)。对于后向函数，加速比是可见的，尽管不是主要的加速比。
我上面写的回传并不是特别优化的，而且肯定可以改进。此外，PyTorch的自动微分引擎可以实现计算图的自动并行化，
总体上可以使用更高效的运算流程，也可以在C++中实现，因此具有较快的实现速度。不管咋样，这是一个良好的开端。

在GPU设备上的性能
**************************

关于PyTorch的ATen后端的一个很棒的事实是，它抽象了您正在运行的计算设备。
这意味着我们为CPU编写的代码也可以在GPU上运行，各个操作将相应地分派给GPU优化的实现。
对于某些操作，如矩阵乘法(如 ``mm`` or ``admm`` )，这是一个巨大的胜利。
让我们来看看使用CUDA张量运行C++代码可以获得多少性能。
不需要对实现进行任何更改，只需将张量放在Python的GPU内存中，
或者在创建时添加 ``device=cuda_device`` 参数，
或者在创建后使用 ``.to(cuda_device)`` ::

  import torch

  assert torch.cuda.is_available()
  cuda_device = torch.device("cuda")  # device object representing GPU

  batch_size = 16
  input_features = 32
  state_size = 128

  # Note the device=cuda_device arguments here
  X = torch.randn(batch_size, input_features, device=cuda_device)
  h = torch.randn(batch_size, state_size, device=cuda_device)
  C = torch.randn(batch_size, state_size, device=cuda_device)

  rnn = LLTM(input_features, state_size).to(cuda_device)

  forward = 0
  backward = 0
  for _ in range(100000):
      start = time.time()
      new_h, new_C = rnn(X, (h, C))
      torch.cuda.synchronize()
      forward += time.time() - start

      start = time.time()
      (new_h.sum() + new_C.sum()).backward()
      torch.cuda.synchronize()
      backward += time.time() - start

  print('Forward: {:.3f} us | Backward {:.3f} us'.format(forward * 1e6/1e5, backward * 1e6/1e5))

再一次将我们普通的PyTorch代码与我们的C++版本(现在都运行在CUDA设备上)进行比较，我们再次看到性能的提高。
对于 Python/PyTorch ::

  Forward: 187.719 us | Backward 410.815 us

对于 C++/ATen::

  Forward: 149.802 us | Backward 393.458 us

与non-CUDA代码相比，这是一个很好的整体加速。但是，通过编写定制的CUDA内核，
我们可以从C++代码中获得更高的性能，我们很快就会深入研究这个问题。
在此之前，让我们讨论另一种构建C++扩展的方法。

JIT 编译扩展
^^^^^^^^^^^^^^^^^^^^^^^^

之前，我提到了构建C++扩展的两种方法：使用 :mod:`setuptools` 或 just in time(JIT)。在讨论了前者之后，让我们详细讨论后者。
JIT编译机制为您提供了一种动态编译和加载扩展的方法，方法是调用PyTorch的API中的一个名为 
:func:`torch.utils.cpp_extension.load` 的简单函数。对于 LLTM 来说，这看起来很简单::

  from torch.utils.cpp_extension import load

  lltm = load(name="lltm", sources=["lltm.cpp"])

在这里，我们为函数提供了与 :mod:`setuptools` 相同的信息。在此背景下，这将执行以下操作:

1. 创建一个临时目录 ``/tmp/torch_extensions/lltm`` ,
2. 将一个 `Ninja <https://ninja-build.org/>`_ 构建文件发送到该临时目录 ,
3. 将源文件编译到共享库中 ,
4. 将此共享库导入为Python模块。

实际上，如果您向 :func:`cpp_extension.load` 传递 ``verbose=True`` ，您将被告知这个过程 ::

  Using /tmp/torch_extensions as PyTorch extensions root...
  Creating extension directory /tmp/torch_extensions/lltm...
  Emitting ninja build file /tmp/torch_extensions/lltm/build.ninja...
  Building extension module lltm...
  Loading extension module lltm...

生成的Python模块将与setuptools生成的模块完全相同，但消除了必须维护单独的 ``setup.py`` 构建文件的要求。
如果您的设置更复杂，并且确实需要 :mod:`setuptools` 的全部功能，那么您可以编写自己的 ``setup.py`` -但在许多情况下，
这种JIT技术会做得很好。当您第一次运行这一行时，需要一些时间，因为扩展是在后台编译的。
由于我们使用Ninja构建系统来构建您的源代码，所以重新编译是增量式的，
因此当您第二次运行Python模块时重新加载扩展是快速的，如果不更改扩展名的源文件，则开销很低。

写一个混合的C++/CUDA扩展
----------------------------------

为了将我们的实现提升到下一个层次，我们可以用定制的CUDA内核手动编写前后传递的部分内容。
对于LLTM，这具有特别有效的前景，因为有大量的逐点操作，所有这些操作都可以在一个CUDA内核中进行融合和并行化。
让我们看看如何编写这样一个CUDA内核，并使用这种扩展机制将其与PyTorch集成。

编写CUDA扩展的一般策略是首先编写一个C++文件，该文件定义将从Python调用的函数，并使用 pybind11 将这些函数绑定到Python。
此外，该文件还将声明CUDA(``.cu``)文件中定义的函数。然后C++函数将进行一些检查，并最终将其调用转发给CUDA函数。
在CUDA文件中，我们编写了实际的CUDA内核。:mod:`cpp_extension` 将负责使用C++编译器(如GCC)编译C++源代码，
以及使用NVIDIA公司的NVCC编译器编译CUDA源代码。这确保了每个编译器都能处理它所知道的最适合编译的文件。
最终，它们将链接到一个共享库中，我们可以从Python代码中获得该库。

我们将从一个C++文件开始，此文件被称为 ``lltm_cuda.cpp``, 如下:

.. code-block:: cpp

  #include <torch/torch.h>

  #include <vector>

  // CUDA forward declarations

  std::vector<at::Tensor> lltm_cuda_forward(
      at::Tensor input,
      at::Tensor weights,
      at::Tensor bias,
      at::Tensor old_h,
      at::Tensor old_cell);

  std::vector<at::Tensor> lltm_cuda_backward(
      at::Tensor grad_h,
      at::Tensor grad_cell,
      at::Tensor new_cell,
      at::Tensor input_gate,
      at::Tensor output_gate,
      at::Tensor candidate_cell,
      at::Tensor X,
      at::Tensor gate_weights,
      at::Tensor weights);

  // C++ interface

  #define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
  #define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
  #define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

  std::vector<at::Tensor> lltm_forward(
      at::Tensor input,
      at::Tensor weights,
      at::Tensor bias,
      at::Tensor old_h,
      at::Tensor old_cell) {
    CHECK_INPUT(input);
    CHECK_INPUT(weights);
    CHECK_INPUT(bias);
    CHECK_INPUT(old_h);
    CHECK_INPUT(old_cell);

    return lltm_cuda_forward(input, weights, bias, old_h, old_cell);
  }

  std::vector<at::Tensor> lltm_backward(
      at::Tensor grad_h,
      at::Tensor grad_cell,
      at::Tensor new_cell,
      at::Tensor input_gate,
      at::Tensor output_gate,
      at::Tensor candidate_cell,
      at::Tensor X,
      at::Tensor gate_weights,
      at::Tensor weights) {
    CHECK_INPUT(grad_h);
    CHECK_INPUT(grad_cell);
    CHECK_INPUT(input_gate);
    CHECK_INPUT(output_gate);
    CHECK_INPUT(candidate_cell);
    CHECK_INPUT(X);
    CHECK_INPUT(gate_weights);
    CHECK_INPUT(weights);

    return lltm_cuda_backward(
        grad_h,
        grad_cell,
        new_cell,
        input_gate,
        output_gate,
        candidate_cell,
        X,
        gate_weights,
        weights);
  }

  PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &lltm_forward, "LLTM forward (CUDA)");
    m.def("backward", &lltm_backward, "LLTM backward (CUDA)");
  }

正如您所看到的，它主要是样板代码、检查和转发到我们将在CUDA文件中定义的函数。我们将命名这个文件 
``lltm_cuda_kernel.cu`` (注意 ``.cu`` 扩展名!)。NVCC可以合理地编译C++11，因此我们仍然可以使用ATen和C++标准库(但不是 ``torch.h``)。
请注意，:mod:`setuptools` 不能处理同名但扩展名不同的文件，因此如果使用 ``setup.py`` 方法而不是JIT方法，
则必须给CUDA文件指定一个与C++文件不同的名称(对于JIT方法，``lltm.cpp`` 和 ``lltm.cu`` 可以正常工作)。
让我们看一下这个文件会是什么样子:

.. code-block:: cpp

  #include <ATen/ATen.h>

  #include <cuda.h>
  #include <cuda_runtime.h>

  #include <vector>

  template <typename scalar_t>
  __device__ __forceinline__ scalar_t sigmoid(scalar_t z) {
    return 1.0 / (1.0 + exp(-z));
  }

这里我们看到了我刚才描述的标题，以及我们使用的是CUDA特定的声明，比如 ``__device__`` 和 ``__forceinline__``  
以及像  ``exp`` 这样的函数。让我们继续讨论一些我们需要的辅助函数:

.. code-block:: cpp

  template <typename scalar_t>
  __device__ __forceinline__ scalar_t d_sigmoid(scalar_t z) {
    const auto s = sigmoid(z);
    return (1.0 - s) * s;
  }

  template <typename scalar_t>
  __device__ __forceinline__ scalar_t d_tanh(scalar_t z) {
    const auto t = tanh(z);
    return 1 - (t * t);
  }

  template <typename scalar_t>
  __device__ __forceinline__ scalar_t elu(scalar_t z, scalar_t alpha = 1.0) {
    return fmax(0.0, z) + fmin(0.0, alpha * (exp(z) - 1.0));
  }

  template <typename scalar_t>
  __device__ __forceinline__ scalar_t d_elu(scalar_t z, scalar_t alpha = 1.0) {
    const auto e = exp(z);
    const auto d_relu = z < 0.0 ? 0.0 : 1.0;
    return d_relu + (((alpha * (e - 1.0)) < 0.0) ? (alpha * e) : 0.0);
  }

要真正实现一个函数，我们需要两样东西：一种是执行我们不希望手动显式编写并调用CUDA内核的操作的函数，
然后是我们想要加快的部分的实际CUDA内核。对于前向传递，第一个函数应该如下所示 :

.. code-block:: cpp

  std::vector<at::Tensor> lltm_cuda_forward(
      at::Tensor input,
      at::Tensor weights,
      at::Tensor bias,
      at::Tensor old_h,
      at::Tensor old_cell) {
    auto X = at::cat({old_h, input}, /*dim=*/1);
    auto gates = at::addmm(bias, X, weights.transpose(0, 1));

    const auto batch_size = old_cell.size(0);
    const auto state_size = old_cell.size(1);

    auto new_h = at::zeros_like(old_cell);
    auto new_cell = at::zeros_like(old_cell);
    auto input_gate = at::zeros_like(old_cell);
    auto output_gate = at::zeros_like(old_cell);
    auto candidate_cell = at::zeros_like(old_cell);

    const int threads = 1024;
    const dim3 blocks((state_size + threads - 1) / threads, batch_size);

    AT_DISPATCH_FLOATING_TYPES(gates.type(), "lltm_forward_cuda", ([&] {
      lltm_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
          gates.data<scalar_t>(),
          old_cell.data<scalar_t>(),
          new_h.data<scalar_t>(),
          new_cell.data<scalar_t>(),
          input_gate.data<scalar_t>(),
          output_gate.data<scalar_t>(),
          candidate_cell.data<scalar_t>(),
          state_size);
    }));

    return {new_h, new_cell, input_gate, output_gate, candidate_cell, X, gates};
  }

这里的主要关注点是 ``AT_DISPATCH_FLOATING_TYPES`` 宏和内核启动(由 ``<<<...>>>`` 指示)。
虽然 ATen 抽象出我们所处理的张量的设备和数据类型，但是在运行时，张量仍将由具体设备上的具体类型的内存支持。
因此，我们需要在运行时确定张量是什么类型，然后有选择地调用具有相应的正确类型签名的函数。
手动完成这一操作(概念上)如下所示:

.. code-block:: cpp

  switch (tensor.type().scalarType()) {
    case at::ScalarType::Double:
      return function<double>(tensor.data<double>());
    case at::ScalarType::Float:
      return function<float>(tensor.data<float>());
    ...
  }

``AT_DISPATCH_FLOATING_TYPES`` 的目的是为我们处理这个调度。它需要一个类型(在本例中是 ``gates.type()`` 、
一个名称(用于错误消息)和一个 lambda 函数。在这个 lambda 函数中，类型别名 ``scalar_t`` 是可用的，
并且被定义为张量在运行时在该上下文中实际存在的类型。因此，如果我们有一个模板函数(我们的CUDA内核将是这样)，
我们可以使用这个 ``scalar_t`` 别名实例化它，并调用正确的函数。
在本例中，我们还希望检索张量的数据指针作为 ``scalar_t`` 类型的指针。
如果您希望对所有类型进行调度，而不仅仅是浮点类型(``Float`` 和 ``Double``)，
则可以使用 ``AT_DISPATCH_ALL_TYPES`` 。

注意，我们使用普通 ATen 执行一些操作。这些操作仍将在GPU上运行，但使用ATen的默认实现。
这是有意义的，因为ATen将使用高度优化的例程来处理矩阵乘法(例如  ``addmm`` )或卷积，
这些我们自己很难实现和改进。

至于内核启动本身，我们在这里指定每个CUDA块将有1024个线程，并且整个GPU网格被分割成
很多具有 ``1 x 1024`` 线程的块(blocks) ,这些块的数量应相当于一个组件用一个线程进行填充矩阵所需的数量。
例如，如果我们的状态大小为2048，批处理大小为4，那么我们将启动总共 ``4 x 2 = 8`` 个块，每个块1024个线程。
如果您以前从未听说过CUDA的“块(blocks)”或“栅格(grids)”，
这个 `CUDA介绍 <https://devblogs.nvidia.com/even-easier-introduction-cuda>`_ 可能会有帮助。

实际的 CUDA kernel 是相当简单的 (如果你之前从未在GPU上做过编程):

.. code-block:: cpp

  template <typename scalar_t>
  __global__ void lltm_cuda_forward_kernel(
      const scalar_t* __restrict__ gates,
      const scalar_t* __restrict__ old_cell,
      scalar_t* __restrict__ new_h,
      scalar_t* __restrict__ new_cell,
      scalar_t* __restrict__ input_gate,
      scalar_t* __restrict__ output_gate,
      scalar_t* __restrict__ candidate_cell,
      size_t state_size) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int index = blockIdx.y * state_size + column;
    const int gates_row = blockIdx.y * (state_size * 3);
    if (column < state_size) {
      input_gate[index] = sigmoid(gates[gates_row + column]);
      output_gate[index] = sigmoid(gates[gates_row + state_size + column]);
      candidate_cell[index] = elu(gates[gates_row + 2 * state_size + column]);
      new_cell[index] =
          old_cell[index] + candidate_cell[index] * input_gate[index];
      new_h[index] = tanh(new_cell[index]) * output_gate[index];
    }
  }

这里最有趣的是，我们能够完全并行地计算所有这些逐点运算，对于我们的门矩阵(gate matrices)中的每个单独的分量。
如果你想象要用一个巨型的 ``for`` 循环来完成这个任务的话，你就会明白为什么这样做的速度要快得多。

反向传递遵循的是相同的模式，我将不再对此作进一步的阐述 :

.. code-block:: cpp

  template <typename scalar_t>
  __global__ void lltm_cuda_backward_kernel(
      scalar_t* __restrict__ d_old_cell,
      scalar_t* __restrict__ d_gates,
      const scalar_t* __restrict__ grad_h,
      const scalar_t* __restrict__ grad_cell,
      const scalar_t* __restrict__ new_cell,
      const scalar_t* __restrict__ input_gate,
      const scalar_t* __restrict__ output_gate,
      const scalar_t* __restrict__ candidate_cell,
      const scalar_t* __restrict__ gate_weights,
      size_t state_size) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int index = blockIdx.y * state_size + column;
    const int gates_row = blockIdx.y * (state_size * 3);
    if (column < state_size) {
      const auto d_output_gate = tanh(new_cell[index]) * grad_h[index];
      const auto d_tanh_new_cell = output_gate[index] * grad_h[index];
      const auto d_new_cell =
          d_tanh(new_cell[index]) * d_tanh_new_cell + grad_cell[index];


      d_old_cell[index] = d_new_cell;
      const auto d_candidate_cell = input_gate[index] * d_new_cell;
      const auto d_input_gate = candidate_cell[index] * d_new_cell;


      const auto input_gate_index = gates_row + column;
      const auto output_gate_index = gates_row + state_size + column;
      const auto candidate_cell_index = gates_row + 2 * state_size + column;

      d_gates[input_gate_index] =
          d_input_gate * d_sigmoid(gate_weights[input_gate_index]);
      d_gates[output_gate_index] =
          d_output_gate * d_sigmoid(gate_weights[output_gate_index]);
      d_gates[candidate_cell_index] =
          d_candidate_cell * d_elu(gate_weights[candidate_cell_index]);
    }
  }

  std::vector<at::Tensor> lltm_cuda_backward(
      at::Tensor grad_h,
      at::Tensor grad_cell,
      at::Tensor new_cell,
      at::Tensor input_gate,
      at::Tensor output_gate,
      at::Tensor candidate_cell,
      at::Tensor X,
      at::Tensor gate_weights,
      at::Tensor weights) {
    auto d_old_cell = at::zeros_like(new_cell);
    auto d_gates = at::zeros_like(gate_weights);

    const auto batch_size = new_cell.size(0);
    const auto state_size = new_cell.size(1);

    const int threads = 1024;
    const dim3 blocks((state_size + threads - 1) / threads, batch_size);

    AT_DISPATCH_FLOATING_TYPES(X.type(), "lltm_backward_cuda", ([&] {
      lltm_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
          d_old_cell.data<scalar_t>(),
          d_gates.data<scalar_t>(),
          grad_h.contiguous().data<scalar_t>(),
          grad_cell.contiguous().data<scalar_t>(),
          new_cell.contiguous().data<scalar_t>(),
          input_gate.contiguous().data<scalar_t>(),
          output_gate.contiguous().data<scalar_t>(),
          candidate_cell.contiguous().data<scalar_t>(),
          gate_weights.contiguous().data<scalar_t>(),
          state_size);
    }));

    auto d_weights = d_gates.t().mm(X);
    auto d_bias = d_gates.sum(/*dim=*/0, /*keepdim=*/true);

    auto d_X = d_gates.mm(weights);
    auto d_old_h = d_X.slice(/*dim=*/1, 0, state_size);
    auto d_input = d_X.slice(/*dim=*/1, state_size);

    return {d_old_h, d_input, d_weights, d_bias, d_old_cell, d_gates};
  }

C++/CUDA操作与PyTorch的集成
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CUDA支持的OP与PyTorch的集成也非常简单。如果您想要编写 ``setup.py`` 脚本，它可能如下所示::

  from setuptools import setup
  from torch.utils.cpp_extension import BuildExtension, CUDAExtension

  setup(
      name='lltm',
      ext_modules=[
          CUDAExtension('lltm_cuda', [
              'lltm_cuda.cpp',
              'lltm_cuda_kernel.cu',
          ])
      ],
      cmdclass={
          'build_ext': BuildExtension
      })

我们现在使用的是 :func:`CUDAExtension` ，而不是 :func:`CppExtension` 。
我们只需指定 ``.cu`` 文件和 ``.cpp`` 文件-
这个库负责处理这给您带来的所有麻烦。JIT机制甚至更简单::

  from torch.utils.cpp_extension import load

  lltm = load(name='lltm', sources=['lltm_cuda.cpp', 'lltm_cuda_kernel.cu'])

性能比较
**********************

我们希望并行化和融合代码的逐点操作和CUDA将提高LLTM的性能。让我们看看这是否成立。
我们可以运行我前面列出的代码来运行基准测试。我们之前最快的版本是基于CUDA的C++代码::

  Forward: 149.802 us | Backward 393.458 us


然后现在使用我们自定义的 CUDA kernel::

  Forward: 129.431 us | Backward 304.641 us

获得了更多的性能提升!

总结
----------

现在，您应该能够很好地了解PyTorch的C++扩展机制以及使用它们的动机。
您可以在 `这里 <https://github.com/pytorch/extension-cpp>`_ 找到代码示例。
如果您有问题，请使用 `论坛 <https://discuss.pytorch.org>`_ 。
此外，一定要检查我们的 `常见问题 <https://pytorch.org/cppdocs/notes/faq.html>`_ ，
以防你遇到任何问题。
