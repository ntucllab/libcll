import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResNet34(nn.Module):
    """
    Parameters
    ----------
    num_classes : int
        the number of classes.
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.resnet = models.resnet34(num_classes=num_classes)
        self.resnet.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        nn.init.kaiming_normal_(
            self.resnet.conv1.weight, mode="fan_out", nonlinearity="relu"
        )

    def forward(self, x):
        return self.resnet(x)

class ResNet18(nn.Module):
    """
    Parameters
    ----------
    num_classes : int
        the number of classes.
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.resnet = models.resnet18(num_classes=num_classes)
        self.resnet.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        nn.init.kaiming_normal_(
            self.resnet.conv1.weight, mode="fan_out", nonlinearity="relu"
        )

    def forward(self, x):
        return self.resnet(x)
