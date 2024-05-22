import numpy as np
import copy
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


class CLBaseDataset(Dataset):
    """
    libcll dataset object

    Parameters
    ----------
    X : Tensor
        the feature of sample set.

    Y : Tensor
        the ground-true labels for corresponding sample.

    num_classes : int
        the number of classes

    Attributes
    ----------
    data : Tensor
        the feature of sample set.

    targets : Tensor
        the complementary labels for corresponding sample.

    true_targets : Tensor
        the ground-truth labels for corresponding sample.

    num_classes : int
        the number of classes.

    input_dim : int
        the feature space after data compressed into a 1D dimension.

    """

    def __init__(self, X, Y, num_classes):
        self.data = X
        self.targets = Y
        self.num_classes = num_classes
        self.input_dim = 1
        for i in range(len(self.data[0].shape)):
            self.input_dim *= self.data[0].shape[i]

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.targets)

    def gen_complementary_target(self, num_cl=1, Q=None):
        """
        Generate complementary labels for each data.

        Parameters
        ----------
        num_cl : int
            the number of complementary labels of each data. Randomly sampled from (0, num_classes) if set to 0.

        Q : Tensor, shape = (num_classes, num_classes)
            Desired Transition matrix to sample the complementary labels.

        """
        if Q == None:
            Q = torch.ones(self.num_classes, self.num_classes) - torch.eye(
                self.num_classes
            )
            Q = Q / (self.num_classes - 1)
        if getattr(self, "true_targets", None) == None:
            self.true_targets = copy.deepcopy(self.targets)
        self.targets = []
        if num_cl == 0:
            p_s = torch.ones(self.num_classes - 1)
            p_s[0] = self.num_classes - 1
            for i in range(1, p_s.shape[0]):
                p_s[i] = p_s[i - 1] * (p_s.shape[0] - i) / (i + 1)
            p_s /= 2 ** (self.num_classes - 1) - 1
        for i in range(len(self.true_targets)):
            if num_cl == 0:
                nc = np.random.choice(
                    np.arange(1, self.num_classes), 1, p=p_s.numpy(), replace=False
                )
            else:
                nc = num_cl
            cl = np.random.choice(
                np.arange(self.num_classes),
                nc,
                p=Q[self.true_targets[i].long()].numpy(),
                replace=False,
            )
            self.targets.append(torch.tensor(cl, dtype=self.true_targets.dtype))
