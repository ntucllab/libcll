Datasets
========

`libcll` has 9 synthetic complementary-label datasets, including 
**MNIST**, **KMNIST**, **FMNIST**, **CIFAR10**, and **CIFAR20** imported from PyTorch, alongside **Yeast**, **Control**, **Dermatology**, and **Texture** imported from OpenML.
Also, `libcll` provides 2 real-world datasets, **CLCIFAR10** and **CLCIFAR20**.

+-------------------+-------------------+-------------+---------------------------------------------------------------------------------------------------+
| Dataset           | Number of Classes | Input Size  | Description                                                                                       |
+===================+===================+=============+===================================================================================================+
| MNIST             | 10                |  28 x 28    | Grayscale images of handwritten digits (0 to 9)                                                   |
+-------------------+-------------------+-------------+---------------------------------------------------------------------------------------------------+
| FMNIST            | 10                |  28 x 28    | Grayscale images of fashion items                                                                 |
+-------------------+-------------------+-------------+---------------------------------------------------------------------------------------------------+
| KMNIST            | 10                |  28 x 28    | Grayscale images of cursive Japanese characters                                                   |
+-------------------+-------------------+-------------+---------------------------------------------------------------------------------------------------+
| Yeast             | 10                |  8          | Features of different localization sites of protein                                               |
+-------------------+-------------------+-------------+---------------------------------------------------------------------------------------------------+
| Texture           | 11                |  40         | Features of different textures                                                                    |
+-------------------+-------------------+-------------+---------------------------------------------------------------------------------------------------+
| Dermatology       | 6                 |  130        | Clinical Attributes of different diseases                                                         |
+-------------------+-------------------+-------------+---------------------------------------------------------------------------------------------------+
| Control           | 6                 |  60         | Synthetic Control Chart Time Series                                                               |
+-------------------+-------------------+-------------+---------------------------------------------------------------------------------------------------+
| Micro ImageNet10  | 10                | 3 x 64 x 64 | Contains images of 10 classes designed for computer vision research                               |
+-------------------+-------------------+-------------+---------------------------------------------------------------------------------------------------+
| Micro ImageNet20  | 20                | 3 x 64 x 64 | Contains images of 20 classes designed for computer vision research                               |
+-------------------+-------------------+-------------+---------------------------------------------------------------------------------------------------+
| CIFAR10           | 10                | 3 x 32 x 32 | Colored images of distinct objects                                                                |
+-------------------+-------------------+-------------+---------------------------------------------------------------------------------------------------+
| CIFAR20           | 20                | 3 x 32 x 32 | Colored images of distinct objects                                                                |
+-------------------+-------------------+-------------+---------------------------------------------------------------------------------------------------+
| CLMicro ImageNet10| 10                | 3 x 64 x 64 | Containing images of 10 classes designed for computer vision research                             |
|                   |                   |             |                                                                                                   |
|                   |                   |             | paired with complementary labels annotated by humans                                              |
+-------------------+-------------------+-------------+---------------------------------------------------------------------------------------------------+
| CLMicro ImageNet20| 20                | 3 x 64 x 64 | Containing images of 20 classes designed for computer vision research                             |
|                   |                   |             |                                                                                                   |
|                   |                   |             | paired with complementary labels annotated by humans                                              |
+-------------------+-------------------+-------------+---------------------------------------------------------------------------------------------------+
| CLCIFAR10         | 10                | 3 x 32 x 32 | Colored images of distinct objects paired with complementary labels annotated by humans           |
+-------------------+-------------------+-------------+---------------------------------------------------------------------------------------------------+
| CLCIFAR10         | 10                | 3 x 32 x 32 | Colored images of distinct objects paired with complementary labels annotated by humans           |
+-------------------+-------------------+-------------+---------------------------------------------------------------------------------------------------+

Custom complementary-label dataset
----------------------------------

We provide base class to easily create complementary-label dataset not included in `libcll`.
Users can effortlessly generate custom dataset inherited from ``libcll.datasets.CLBaseDataset`` and redefine ``__get_item__()`` if needed.

.. code-block:: python

    import torch
    from PIL import Image
    import numpy as np
    import torchvision.transforms as transforms
    from libcll.datasets import CLBaseDataset
    from torchvision.datasets import SVHN

    train_set = SVHN(root="./data/svhn", split="train", download=True)
    X_train = train_set.data
    Y_train = torch.from_numpy(train_set.labels)
    class CLSVHN(CLBaseDataset):
        def __getitem__(self, index):
            img, target = self.data[index], self.targets[index]
            img = Image.fromarray(np.transpose(img, (1, 2, 0)))
            transform = transforms.ToTensor()
            img = transform(img)
            return img, target
    train_set = CLSVHN(
        X=X_train, 
        Y=Y_train, 
        num_classes=10
    )
    train_set.gen_complementary_target()


Generate complementary labels using transition matrix
-----------------------------------------------------

`libcll` provides 4 types of commonly-used transition matrices for complementary-label generation from ``libcll.datasets.utils.get_transition_matrix(transition_matrix, num_classes)``.
Notice that ``weak``, ``strong``, and ``noise`` transition matrices are designed specifically for datasets containing 10 classes.
Users can generate complementary labels based on their desired distribution by passing transition matrix to ``CLBaseDataset.gen_complementary_target()``.


+-------------------+------------------------------------------------------------------------------------------------------------------------------------+
| Transition matrix |      Description                                                                                                                   |
+===================+====================================================================================================================================+
| ``uniform``       | a uniform transition matrix where the diagonal elements are zero,                                                                  |
|                   |                                                                                                                                    |
|                   | and all non-diagonal elements are equal to :math:`\frac{1}{K - 1}`, :math:`K` representing ``num_classes``                         |
+-------------------+------------------------------------------------------------------------------------------------------------------------------------+
|    ``weak``       | a biased transition matrix simulate milder deviation from uniform distribution,                                                    |
|                   |                                                                                                                                    |
|                   | where the diagonal elements are zero and all non-diagonal elements randomly set.                                                   |
+-------------------+------------------------------------------------------------------------------------------------------------------------------------+
|    ``strong``     | a biased transition matrix simulate stronger deviation from uniform distribution,                                                  |
|                   |                                                                                                                                    |
|                   | where the diagonal elements are zero and all non-diagonal elements randomly set.                                                   |
+-------------------+------------------------------------------------------------------------------------------------------------------------------------+
|    ``noisy``      | a noisy transition matrix where the diagonal elements are not necessary zero,                                                      |
|                   |                                                                                                                                    |
|                   | and equals to :math:`(1-\lambda)T_{\text{strong}}+\lambda\frac{1}{K}1_{K}`, :math:`\lambda` representing the weight of noise.      |
+-------------------+------------------------------------------------------------------------------------------------------------------------------------+

.. code-block:: python

    from libcll.datasets import CLMNIST
    from libcll.datasets.utils import get_transition_matrix

    train_set = CLMNIST(root="./data/mnist", train=True)
    transition_matrix = get_transition_matrix(
        transition_matrix="weak", 
        num_classes=train_set.num_classes
    )
    train_set.gen_complementary_target(transition_matrix)

Multiple complementary-label dataset
------------------------------------

`libcll` offers two types of multiple complementary-label learning settings by the parameter num_cl, which specifies the number of complementary labels for each instance.
When set to zero, ``num_cl`` triggers random sampling of the number of complementary labels per data instance before actual complementary-label sampling.

Since each data has multiple complementary labels, batch decomposition is necessary before passing it to the learner.
We provide two different collate function in ``libcll.datasets.utils`` for dataloader, ``collate_fn_multi_label`` duplicates image inputs to align with target lengths, while ``collate_fn_one_hot`` uses one-hot vectors to store multiple labels.


.. code-block:: python
    
    from torch.utils.data import random_split, DataLoader
    from libcll.datasets import CLMNIST
    from libcll.datasets.utils import collate_fn_multi_label

    train_set = CLMNIST(root="./data/mnist", train=True)
    test_set = CLMNIST(root="./data/mnist", train=False)
    train_set.gen_complementary_target(num_cl=3)
    input_dim = train_set.input_dim
    num_classes = train_set.num_classes

    batch_size = 256
    train_set, valid_set = random_split(train_set, [0.9, 0.1])
    train_loader = DataLoader(train_set, batch_size=batch_size, collate_fn=collate_fn_multi_label, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, collate_fn=collate_fn_multi_label, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)