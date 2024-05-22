Libcll
==================================
What is Libcll?
---------------------
`libcll` is an extendable Python toolkit designed for conducting complementary-label learning process in a standardized way. 
This library provides synthetic complementary-label datasets based on specified distributions, 
state-of-the-art complementary-label learning methods, 
metrics for complementary-label learning evaluation, and support for multiple complementary-label learning approaches.


Getting Started
---------------

In this session, we will showcase an example to kickstart `libcll` and guide you through the complementary-label learning process 
to demonstrate the simplicity of CLL implementation with `libcll`.
For the beginners, we recommend SCL-NL method due to its simplicity and speed, along with CLMNIST dataset for its quick learning curve.

Installation
````````````

- Python version >= 3.8, <= 3.10
- Pytorch version >= 1.11, <= 2.0
- Pytorch Lightning version >= 2.0
- To install `libcll` and develop locally:

.. code-block:: shell

   git clone git@github.com:ntucllab/libcll.git
   cd libcll
   pip install -e .

Data Preparation
````````````````

First, the CLMNIST dataset can be directly imported from `libcll`. 
However, in the aid of diverse complementary-label distribution settings, 
the user need to execute function ``libcll.datasets.CLBaseDataset.gen_complementary_target(num_cl, Q)``, 
where ``num_cl`` and ``Q`` represent number of complementary labels for each instance and class transition probability matrix respectively.

.. code-block:: python

   from torch.utils.data import random_split, DataLoader
   from libcll.datasets import CLMNIST
   from libcll.datasets.utils import collate_fn_multi_label

   train_set = CLMNIST(root="./data/mnist", train=True)
   test_set = CLMNIST(root="./data/mnist", train=False)
   train_set.gen_complementary_target()
   input_dim = train_set.input_dim
   num_classes = train_set.num_classes

   batch_size = 256
   train_set, valid_set = random_split(train_set, [0.9, 0.1])
   train_loader = DataLoader(train_set, batch_size=batch_size, collate_fn=collate_fn_multi_label, shuffle=True, num_workers=4)
   valid_loader = DataLoader(valid_set, batch_size=batch_size, collate_fn=collate_fn_multi_label, shuffle=False, num_workers=4)
   test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

Build Model and Strategy
````````````````````````

`libcll` provides easy access to well-known complementary-label learning methods. 
To mimic real-world scenarios, 
we assume the validation set contains only complementary labels and thus use SCEL loss instead of accuracy as the validation metric.

.. code-block:: python

   from libcll.models import MLP
   from libcll.strategies import SCL

   model = MLP(input_dim, 512, num_classes)
   strategy = SCL(
      model=model,
      valid_type="SCEL", 
      num_classes=num_classes,
      type="NL", 
      lr=1e-4, 
   )

Start Training
``````````````

To train the model, we leverage PyTorch Lightning for easy-to-build training pipeline, GPU acceleration and callbacks such as early stopping, model checkpointing, and logging to TensorBoard. 

.. code-block:: python

   import pytorch_lightning as pl

   trainer = pl.Trainer(
      max_epochs=300, 
      accelerator="gpu",
   )
   trainer.fit(
      strategy,
      train_dataloaders=train_loader,
      val_dataloaders=valid_loader,
   )
   trainer.test(
      dataloaders=test_loader, 
   )


----------

.. toctree::
   :caption: User Guide
   :maxdepth: 2

   user-guide/Datasets
   user-guide/Models
   user-guide/Strategies

.. toctree::
   :caption: API Reference
   :maxdepth: 1

   api/Datasets
   api/Models
   api/Strategies