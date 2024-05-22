Strategies
==========

`libcll` supports 14 different complementary-label learning algorithems. 
Notice that some of the methods need transition matrix or class priors (URE) to calculate loss. 

+---------+---------+------------------------------------------------------------------------------------------------------------------------------------+
| Stategy |   Type  |      Description                                                                                                                   |
+=========+=========+====================================================================================================================================+
| SCL     |   NL    |      Surrogate Complementary Loss with negative log loss                                                                           |
|         +---------+------------------------------------------------------------------------------------------------------------------------------------+
|         |   EXP   |      Surrogate Complementary Loss with exponential loss                                                                            |
+---------+---------+------------------------------------------------------------------------------------------------------------------------------------+
| URE     |   NN    |      Unbiased risk estimator with uniform transition matrix                                                                        |
|         +---------+------------------------------------------------------------------------------------------------------------------------------------+
|         |   TNN   |      Unbiased risk estimator with true transition matrix                                                                           |
|         +---------+------------------------------------------------------------------------------------------------------------------------------------+
|         |   GA    |      Unbiased risk estimator with uniform transition matrix and gradient ascent                                                    |
|         +---------+------------------------------------------------------------------------------------------------------------------------------------+
|         |   TGA   |      Unbiased risk estimator with true transition matrix and gradient ascent                                                       |
+---------+---------+------------------------------------------------------------------------------------------------------------------------------------+
| FWD     |   None  |      Forward Correction with true transition matrix                                                                                |
+---------+---------+------------------------------------------------------------------------------------------------------------------------------------+
| DM      |   None  |      Discriminative Models with Weighted Loss                                                                                      |
+---------+---------+------------------------------------------------------------------------------------------------------------------------------------+
| CPE     |   I     |      Complementary probability estimates                                                                                           |
|         +---------+------------------------------------------------------------------------------------------------------------------------------------+
|         |   F     |      Complementary probability estimates with true transition matrix                                                               |
|         +---------+------------------------------------------------------------------------------------------------------------------------------------+
|         |   T     |      Complementary probability estimates with trainable transition matrix                                                          |
+---------+---------+------------------------------------------------------------------------------------------------------------------------------------+
| MCL     |   MAE   |      Multiple Complementary Loss with mean absolute error                                                                          |
|         +---------+------------------------------------------------------------------------------------------------------------------------------------+
|         |   EXP   |      Multiple Complementary Loss with exponential loss                                                                             |
|         +---------+------------------------------------------------------------------------------------------------------------------------------------+
|         |   LOG   |      Multiple Complementary Loss with negative log loss                                                                            |
+---------+---------+------------------------------------------------------------------------------------------------------------------------------------+



Custom complementary-label strategy
-----------------------------------

For the furture development of complementary-label learning, 
we offer the base class ``libcll.strategies.Strategy`` to allow users implement their learning methods in `libcll`.
``libcll.strategies.Strategy`` has already defined validation and testing step and stored transition matrix in ``self.Q``, users only need to customize ``training_step`` to calculate loss.

.. code-block:: python
    
    import torch
    import torch.nn.functional as F
    from libcll.strategies.Strategy import Strategy
    from libcll.models import MLP
    
    class FWD(Strategy):
        def training_step(self, batch, batch_idx):
            x, y = batch
            out = self.model(x)
            p = torch.mm(F.softmax(out, dim=1), self.Q) + 1e-6
            loss = F.nll_loss(p.log(), y.long())
            self.log("Train_Loss", loss)
            return loss
    model = MLP(input_dim, 512, num_classes)
    strategy = SCL(
        model=model,
        valid_type="SCEL", 
        num_classes=num_classes,
        type="NL", 
        lr=1e-4, 
    )

Change evaluation metric
------------------------

The ``Strategy`` class provides three distinct evaluation metrics. 
We recommend to use ``SCEL`` metric to avoid the need for preparing the validation dataset with true labels.
Given that ``URE`` metric requires calculating the pseudo-inverse of the transition matrix, 
it is unstable under the non-uniform distribution.

+--------------------+------------------------------------------------------------------------------------------------------------------------------------+
| Evaluation metrics |      Description                                                                                                                   |
+====================+====================================================================================================================================+
|    ``URE``         | Unbiased risk estimator of the zero-one loss.                                                                                      |
+--------------------+------------------------------------------------------------------------------------------------------------------------------------+
|    ``SCEL``        | The log loss of the complementary probability estimates induced by the probability                                                 |
|                    |                                                                                                                                    |
|                    | estimated on the ordinary label space.                                                                                             |
+--------------------+------------------------------------------------------------------------------------------------------------------------------------+
| ``Accuracy``       | Accuracy in classification.                                                                                                        |
|                    |                                                                                                                                    |
|                    | This metric is applicable only to datasets containing true labels.                                                                 |
+--------------------+------------------------------------------------------------------------------------------------------------------------------------+

If the users want to append new method to evaluate their training performance, they can modify the ``validation_step`` in ``Strategy`` like the following example:

.. code-block:: python
    
    class CPE(Strategy):
        def validation_step(self, batch, batch_idx):
            # Calculate the validation metric
            # Remember to append the value to self.val_loss
            self.val_loss.append(val_loss)
            return {"val_loss": val_loss}