使用Pytorch将深度学习用于NLP
**********************************
**Author**: `Robert Guthrie <https://github.com/rguthrie3/DeepLearningForNLPInPytorch>`_
**翻译者**: `Antares <http://www.studyai.com/antares>`_

本教程将向您介绍如何使用PYTORCH进行深度学习编程的关键思想。
许多概念(例如计算图抽象和自动梯度)并不是Pytorch独有的，
而且与任何深度学习工具包都相关。

我编写本教程是为了专门针对那些从未在任何深度学习框架
(例如，TensorFlow、Theano、Keras、Dynet)中编写代码的人编写NLP。
它假定了核心NLP问题的相关知识：part-of-speech tagging, language modeling 等.
它还假设在介绍AI类(如Russel和Norvig书中的一个)的水平上熟悉神经网络。
这些课程通常涉及前馈神经网络的基本反向传播算法，并指出它们是线性和非线性的组合链。
本教程旨在让您开始编写深度学习代码，如果您有这一先决条件的知识。

注意，这是关于模型(*models*)的，而不是数据。对于所有的模型，
我只是创建了一些小维度的测试示例，这样您就可以看到在训练过程中权重是如何变化的。
如果你有一些真实的数据，你想尝试，你应该能够从这个notebook上取出任何models，并使用它们。


.. toctree::
    :maxdepth: 1

    /beginner/nlp/pytorch_tutorial
    /beginner/nlp/deep_learning_tutorial
    /beginner/nlp/word_embeddings_tutorial
    /beginner/nlp/sequence_models_tutorial
    /beginner/nlp/advanced_tutorial


.. galleryitem:: /beginner/nlp/pytorch_tutorial.py
    :intro: All of deep learning is computations on tensors, which are generalizations of a matrix that can be

.. galleryitem:: /beginner/nlp/deep_learning_tutorial.py
    :intro: Deep learning consists of composing linearities with non-linearities in clever ways. The introduction of non-linearities allows

.. galleryitem:: /beginner/nlp/word_embeddings_tutorial.py
    :intro: Word embeddings are dense vectors of real numbers, one per word in your vocabulary. In NLP, it is almost always the case that your features are

.. galleryitem:: /beginner/nlp/sequence_models_tutorial.py
    :intro: At this point, we have seen various feed-forward networks. That is, there is no state maintained by the network at all.

.. galleryitem:: /beginner/nlp/advanced_tutorial.py
    :intro: Dynamic versus Static Deep Learning Toolkits. Pytorch is a *dynamic* neural network kit.


.. raw:: html

    <div style='clear:both'></div>
