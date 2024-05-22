import torch
import torch.nn as nn


class Linear(nn.Module):
    """
    Parameters
    ----------
    input_dim : int
        the feature space after data compressed into a 1D dimension.
    num_classes : int
        the number of classes.
    """

    def __init__(self, input_dim, num_classes=10):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, num_classes),
        )

    def forward(self, x):
        if len(x.shape) == 4:
            x = x.view(x.shape[0], -1)
        if len(x.shape) == 3:
            x = x.view(-1)
        return self.layer(x)
