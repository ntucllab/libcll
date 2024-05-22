import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math


class DenseLayer(nn.Module):
    def __init__(self, input_features, expansion=4, growthRate=12, drop_rate=0):
        super(DenseLayer, self).__init__()
        self.drop_rate = drop_rate
        self.dense_layer = nn.Sequential(
            nn.BatchNorm2d(input_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=input_features,
                out_channels=expansion * growthRate,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(expansion * growthRate),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=expansion * growthRate,
                out_channels=growthRate,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
        )
        self.dropout = nn.Dropout(p=self.drop_rate)

    def forward(self, x):
        y = self.dense_layer(x)
        if self.drop_rate != 0:
            y = self.dropout(y)
        return torch.cat([x, y], dim=1)


class TransitionLayer(nn.Module):
    def __init__(self, input, output):
        super(TransitionLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(input),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=input,
                out_channels=output,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.AvgPool2d(kernel_size=2),
        )

    def forward(self, x):
        return self.layer(x)


class DenseNet(nn.Module):
    """
    Parameters
    ----------
    num_layers : list
        the number of layers of each dense block.
    dropRate : int
        the drop rate of the dropout layers.
    growthRate : int
        the additional number of channels for each layer.
    compressionRate : int
        the ratio by which the number of feature maps is reduced at transition layers.
    num_classes : int
        the number of classes.
    """

    def __init__(
        self,
        num_layers=[16, 16, 16],
        dropRate=0,
        num_classes=10,
        growthRate=12,
        compressionRate=2,
    ):
        super().__init__()
        self.features = 2 * growthRate
        self.in_layer = nn.Conv2d(3, self.features, 3, padding=1, bias=False)
        self.depth = len(num_layers)
        for ind in range(self.depth):
            block = [
                DenseLayer(self.features + i * growthRate, growthRate=growthRate)
                for i in range(num_layers[ind])
            ]
            setattr(self, f"dense{ind+1}", nn.Sequential(*block))
            self.features += num_layers[ind] * growthRate

            if ind != self.depth - 1:
                setattr(
                    self,
                    f"trans{ind+1}",
                    TransitionLayer(self.features, self.features // compressionRate),
                )
                self.features //= compressionRate
        self.bn = nn.BatchNorm2d(self.features)
        self.linear = nn.Linear(self.features, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.in_layer(x)
        for ind in range(self.depth):
            d = getattr(self, f"dense{ind+1}")
            x = d(x)
            if ind != self.depth - 1:
                t = getattr(self, f"trans{ind+1}")
                x = t(x)
        x = F.avg_pool2d(F.relu(self.bn(x)), 8)
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return x
