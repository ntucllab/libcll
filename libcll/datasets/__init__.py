import torch
import torchvision.transforms as transforms
from torch.utils.data import random_split, Sampler, DataLoader
import numpy as np
from .cl_base_dataset import CLBaseDataset
from .cl_cifar10 import CLCIFAR10
from .cl_cifar20 import CLCIFAR20
from .cl_yeast import CLYeast
from .cl_texture import CLTexture
from .cl_control import CLControl
from .cl_dermatology import CLDermatology
from .cl_fmnist import CLFMNIST
from .cl_kmnist import CLKMNIST
from .cl_mnist import CLMNIST
from .cl_tiny_imagenet10 import CLTiny_ImageNet10
from .cl_tiny_imagenet20 import CLTiny_ImageNet20
from .utils import get_transition_matrix, collate_fn_multi_label, collate_fn_one_hot


class IndexSampler(Sampler):
    def __init__(self, index):
        self.index = index

    def __iter__(self):
        ind = torch.randperm(len(self.index))
        return iter(self.index[ind].tolist())

    def __len__(self):
        return len(self.index)


def prepare_dataloader(
    dataset,
    batch_size=256,
    valid_split=0.1,
    valid_type="URE",
    one_hot=False,
    transition_matrix=None,
    num_cl=1,
    augment=False,
    noise=0.1,
    seed=1126,
):

    if dataset == "mnist":
        train_set = CLMNIST(
            root="./data/mnist",
            train=True,
        )
        test_set = CLMNIST(root="./data/mnist", train=False)

        Q = get_transition_matrix(transition_matrix, train_set.num_classes, noise, seed)
        train_set.gen_complementary_target(num_cl, Q)

    elif dataset == "fmnist":
        train_set = CLFMNIST(
            root="./data/fmnist",
            train=True,
        )
        test_set = CLFMNIST(root="./data/fmnist", train=False)

        Q = get_transition_matrix(transition_matrix, train_set.num_classes, noise, seed)
        train_set.gen_complementary_target(num_cl, Q)

    elif dataset == "kmnist":
        train_set = CLKMNIST(
            root="./data/kmnist",
            train=True,
        )
        test_set = CLKMNIST(root="./data/kmnist", train=False)

        Q = get_transition_matrix(transition_matrix, train_set.num_classes, noise, seed)
        train_set.gen_complementary_target(num_cl, Q)

    elif dataset == "cifar10":
        if augment:
            train_transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, padding=4),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.4914, 0.4822, 0.4465], [0.247, 0.2435, 0.2616]
                    ),
                ]
            )
        else:
            train_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.4914, 0.4822, 0.4465], [0.247, 0.2435, 0.2616]
                    ),
                ]
            )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.2435, 0.2616]),
            ]
        )
        train_set = CLCIFAR10(
            root="./data/cifar10",
            train=True,
            transform=train_transform,
        )
        test_set = CLCIFAR10(
            root="./data/cifar10",
            train=False,
            transform=test_transform,
        )

        Q = get_transition_matrix(transition_matrix, train_set.num_classes, noise, seed)
        train_set.gen_complementary_target(num_cl, Q)

    elif dataset == "cifar20":
        if augment:
            train_transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, padding=4),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.4914, 0.4822, 0.4465], [0.247, 0.2435, 0.2616]
                    ),
                ]
            )
        else:
            train_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.4914, 0.4822, 0.4465], [0.247, 0.2435, 0.2616]
                    ),
                ]
            )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.2435, 0.2616]),
            ]
        )
        train_set = CLCIFAR20(
            root="./data/cifar20",
            train=True,
            transform=train_transform,
        )
        test_set = CLCIFAR20(
            root="./data/cifar20",
            train=False,
            transform=test_transform,
        )

        Q = get_transition_matrix(transition_matrix, train_set.num_classes, noise, seed)
        train_set.gen_complementary_target(num_cl, Q)

    elif dataset == "yeast":
        train_set = CLYeast(
            root="./data/yeast",
            train=True,
        )
        test_set = CLYeast(
            root="./data/yeast",
            train=False,
        )

        Q = get_transition_matrix(transition_matrix, train_set.num_classes, noise, seed)
        train_set.gen_complementary_target(num_cl, Q)

    elif dataset == "texture":
        train_set = CLTexture(
            root="./data/texture",
            train=True,
        )
        test_set = CLTexture(
            root="./data/texture",
            train=False,
        )

        Q = get_transition_matrix(transition_matrix, train_set.num_classes, noise, seed)
        train_set.gen_complementary_target(num_cl, Q)

    elif dataset == "dermatology":
        train_set = CLDermatology(
            root="./data/dermatology",
            train=True,
        )
        test_set = CLDermatology(
            root="./data/dermatology",
            train=False,
        )

        Q = get_transition_matrix(transition_matrix, train_set.num_classes, noise, seed)
        train_set.gen_complementary_target(num_cl, Q)

    elif dataset == "control":
        train_set = CLControl(
            root="./data/control",
            train=True,
        )
        test_set = CLControl(
            root="./data/control",
            train=False,
        )

        Q = get_transition_matrix(transition_matrix, train_set.num_classes, noise, seed)
        train_set.gen_complementary_target(num_cl, Q)

    elif dataset == "clcifar10":
        if augment:
            train_transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, padding=4),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.4914, 0.4822, 0.4465], [0.247, 0.2435, 0.2616]
                    ),
                ]
            )
        else:
            train_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.4914, 0.4822, 0.4465], [0.247, 0.2435, 0.2616]
                    ),
                ]
            )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.2435, 0.2616]),
            ]
        )
        train_set = CLCIFAR10(
            root="./data/cifar10",
            train=True,
            transform=train_transform,
            num_cl=num_cl,
        )
        test_set = CLCIFAR10(
            root="./data/cifar10",
            train=False,
            transform=test_transform,
            num_cl=num_cl,
        )

    elif dataset == "clcifar20":
        if augment:
            train_transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, padding=4),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.4914, 0.4822, 0.4465], [0.247, 0.2435, 0.2616]
                    ),
                ]
            )
        else:
            train_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.4914, 0.4822, 0.4465], [0.247, 0.2435, 0.2616]
                    ),
                ]
            )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.2435, 0.2616]),
            ]
        )
        train_set = CLCIFAR20(
            root="./data/cifar20",
            train=True,
            transform=train_transform,
            num_cl=num_cl,
        )
        test_set = CLCIFAR20(
            root="./data/cifar20",
            train=False,
            transform=test_transform,
            num_cl=num_cl,
        )

    elif dataset == "tiny_imagenet10":
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(64, padding=8),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        train_set = CLTiny_ImageNet10(
            root="/home/maitanha/Tiny_CLL_Data/tinyImageNet10_complementary",
            train=True,
            transform=train_transform,
            num_cl=num_cl,
        )
        test_set = CLTiny_ImageNet10(
            root="/home/maitanha/Tiny_CLL_Data/tinyImageNet10_complementary",
            train=False,
            transform=test_transform,
            num_cl=num_cl,
        )
        Q = get_transition_matrix(transition_matrix, train_set.num_classes, noise, seed)
        train_set.gen_complementary_target(num_cl, Q)

    elif dataset == "tiny_imagenet20":
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(64, padding=8),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        train_set = CLTiny_ImageNet20(
            root="/home/maitanha/Tiny_CLL_Data/tinyImageNet20_complementary",
            train=True,
            transform=train_transform,
            num_cl=num_cl,
        )
        test_set = CLTiny_ImageNet20(
            root="/home/maitanha/Tiny_CLL_Data/tinyImageNet20_complementary",
            train=False,
            transform=test_transform,
            num_cl=num_cl,
        )
        Q = get_transition_matrix(transition_matrix, train_set.num_classes, noise, seed)
        train_set.gen_complementary_target(num_cl, Q)

    elif dataset == "cltiny_imagenet10":
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(64, padding=8),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        train_set = CLTiny_ImageNet10(
            root="/home/maitanha/Tiny_CLL_Data/tinyImageNet10_complementary",
            train=True,
            transform=train_transform,
            num_cl=num_cl,
        )
        test_set = CLTiny_ImageNet10(
            root="/home/maitanha/Tiny_CLL_Data/tinyImageNet10_complementary",
            train=False,
            transform=test_transform,
            num_cl=num_cl,
        )

    elif dataset == "cltiny_imagenet20":
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(64, padding=8),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        train_set = CLTiny_ImageNet20(
            root="/home/maitanha/Tiny_CLL_Data/tinyImageNet20_complementary",
            train=True,
            transform=train_transform,
            num_cl=num_cl,
        )
        test_set = CLTiny_ImageNet20(
            root="/home/maitanha/Tiny_CLL_Data/tinyImageNet20_complementary",
            train=False,
            transform=test_transform,
            num_cl=num_cl,
        )

    else:
        raise NotImplementedError

    idx = np.arange(len(train_set))
    np.random.shuffle(idx)
    train_idx = idx[: int(len(train_set) * (1 - valid_split))]
    valid_idx = idx[int(len(train_set) * (1 - valid_split)) :]
    train_sampler = IndexSampler(train_idx)
    valid_sampler = IndexSampler(valid_idx)
    if valid_type == "Accuracy":
        for i in valid_idx:
            train_set.targets[i] = train_set.true_targets[i].view(1)
    train_loader = DataLoader(
        train_set,
        sampler=train_sampler,
        batch_size=batch_size,
        collate_fn=(
            collate_fn_multi_label
            if not one_hot
            else lambda batch: collate_fn_one_hot(
                batch, num_classes=train_set.num_classes
            )
        ),
        shuffle=False,
        num_workers=4,
    )
    valid_loader = DataLoader(
        train_set,
        sampler=valid_sampler,
        batch_size=batch_size,
        collate_fn=collate_fn_multi_label,
        shuffle=False,
        num_workers=4,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=4
    )
    Q = torch.zeros((train_set.num_classes, train_set.num_classes))
    for idx in train_idx:
        Q[train_set.true_targets[idx].long()][train_set.targets[idx].long()] += 1
    class_priors = Q.sum(dim=0)
    Q = Q / Q.sum(dim=1).view(-1, 1)
    if transition_matrix == "noisy":
        Q = get_transition_matrix("strong", train_set.num_classes, seed)
    return (
        train_loader,
        valid_loader,
        test_loader,
        train_set.input_dim,
        train_set.num_classes,
        Q,
        class_priors,
    )
