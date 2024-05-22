API for Datasets
================

Base Dataset Module
-------------------

Dataset define the data format and provide helpers for generating complementary labels.

.. autoclass:: libcll.datasets.CLBaseDataset
    :members:
    :undoc-members:
    :show-inheritance:

Complementary-label Dataset Module
----------------------------------

Here, we exclusively showcase CLMNIST and CLCIFAR10 as an illustration for synthetic and real-world datasets respectively, given that other datasets in libcll share the same attributes, functions, and initialization procedures.

.. autoclass:: libcll.datasets.CLMNIST
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: libcll.datasets.CLCIFAR10
    :members:
    :undoc-members:
    :show-inheritance:

Complementary-label Dataset Utilities
-------------------------------------

.. autofunction:: libcll.datasets.get_transition_matrix

.. autofunction:: libcll.datasets.collate_fn_multi_label

.. autofunction:: libcll.datasets.collate_fn_one_hot
