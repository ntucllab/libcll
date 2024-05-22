Models
======

`libcll` supports 4 commonly-used deep learning models, as shown in the table below.

+-------------------+------------------------------------------------------------------------------------------------------------------------------------+
| Models            |      Description                                                                                                                   |
+===================+====================================================================================================================================+
|   ``Linear``      | Single-layer neural network (input_dim-num_classes) for simple datasets.                                                           |
+-------------------+------------------------------------------------------------------------------------------------------------------------------------+
|  ``MLP``          | Multi-layer neural network (input_dim-hidden_dim-num_classes) for simple datasets.                                                 |
+-------------------+------------------------------------------------------------------------------------------------------------------------------------+
|    ``ResNet``     | Residual neural network for hard datasets.                                                                                         |
+-------------------+------------------------------------------------------------------------------------------------------------------------------------+
|  ``DenseNet``     | Densely connected convolutional network for hard datasets.                                                                         |
+-------------------+------------------------------------------------------------------------------------------------------------------------------------+

Loading a Model
---------------

To build a model, `libcll` requires input dimension and the number of classes from dataset and hidden dimension if needed.

.. code-block:: python

    from libcll.models import build_model

    model = build_model(
        model="MLP", 
        input_dim=input_dim, 
        hidden_dim=512, 
        num_classes=num_classes,
    )
