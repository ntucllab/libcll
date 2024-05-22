import torch
import torchvision
from libcll.datasets.cl_base_dataset import CLBaseDataset
import pickle
import gdown
import os


def _cifar100_to_cifar20(target):
    # obtained from cifar_test script
    _dict = {
        0: 4,
        1: 1,
        2: 14,
        3: 8,
        4: 0,
        5: 6,
        6: 7,
        7: 7,
        8: 18,
        9: 3,
        10: 3,
        11: 14,
        12: 9,
        13: 18,
        14: 7,
        15: 11,
        16: 3,
        17: 9,
        18: 7,
        19: 11,
        20: 6,
        21: 11,
        22: 5,
        23: 10,
        24: 7,
        25: 6,
        26: 13,
        27: 15,
        28: 3,
        29: 15,
        30: 0,
        31: 11,
        32: 1,
        33: 10,
        34: 12,
        35: 14,
        36: 16,
        37: 9,
        38: 11,
        39: 5,
        40: 5,
        41: 19,
        42: 8,
        43: 8,
        44: 15,
        45: 13,
        46: 14,
        47: 17,
        48: 18,
        49: 10,
        50: 16,
        51: 4,
        52: 17,
        53: 4,
        54: 2,
        55: 0,
        56: 17,
        57: 4,
        58: 18,
        59: 17,
        60: 10,
        61: 3,
        62: 2,
        63: 12,
        64: 12,
        65: 16,
        66: 12,
        67: 1,
        68: 9,
        69: 19,
        70: 2,
        71: 10,
        72: 0,
        73: 1,
        74: 16,
        75: 12,
        76: 9,
        77: 13,
        78: 15,
        79: 13,
        80: 16,
        81: 19,
        82: 2,
        83: 4,
        84: 6,
        85: 19,
        86: 5,
        87: 5,
        88: 8,
        89: 19,
        90: 18,
        91: 1,
        92: 2,
        93: 15,
        94: 6,
        95: 0,
        96: 17,
        97: 8,
        98: 14,
        99: 13,
    }

    return _dict[target]


class CLCIFAR20(torchvision.datasets.CIFAR100, CLBaseDataset):
    """

    Parameters
    ----------
    root : str
        path to store dataset file.

    train : bool
        training set if True, else testing set.

    transform : callable, optional
        a function/transform that takes in a PIL image and returns a transformed version.

    target_transform : callable, optional
        a function/transform that takes in the target and transforms it.

    download : bool
        if true, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again.

    num_cl : int
        the number of real-world complementary labels of each data chosen from [1, 3].

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

    def __init__(
        self,
        root="./data/cifar20",
        train=True,
        transform=None,
        target_transform=None,
        download=True,
        num_cl=1,
    ):
        if train:
            dataset_path = f"{root}/clcifar20.pkl"
            if download and not os.path.exists(dataset_path):
                os.makedirs(root, exist_ok=True)
                gdown.download(
                    id="1PhZsyoi1dAHDGlmB4QIJvDHLf_JBsFeP", output=dataset_path
                )
            with open(dataset_path, "rb") as f:
                data = pickle.load(f)
            self.data = data["images"]
            self.true_targets = torch.Tensor(data["ord_labels"]).view(-1)
            self.targets = torch.Tensor(data["cl_labels"])[:, :num_cl]
            self.transform = transform
            self.target_transform = target_transform
        else:
            super(CLCIFAR20, self).__init__(
                root, train, transform, target_transform, download
            )
            self.targets = [_cifar100_to_cifar20(i) for i in self.targets]
            self.targets = torch.Tensor(self.targets)

        self.num_classes = 20
        self.input_dim = 3 * 32 * 32
