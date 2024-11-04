from .cl_dataset_module import CLDataModule
from .cl_cifar10 import CLCIFAR10
from .cl_cifar20 import CLCIFAR20
from .cl_yeast import CLYeast
from .cl_texture import CLTexture
from .cl_control import CLControl
from .cl_dermatology import CLDermatology
from .cl_fmnist import CLFMNIST
from .cl_kmnist import CLKMNIST
from .cl_mnist import CLMNIST
from .cl_micro_imagenet10 import CLMicro_ImageNet10
from .cl_micro_imagenet20 import CLMicro_ImageNet20

D_LIST = {
    "mnist": CLMNIST, 
    "kmnist": CLKMNIST, 
    "fmnist": CLFMNIST, 
    "cifar10": CLCIFAR10, 
    "cifar20": CLCIFAR20, 
    "yeast": CLYeast, 
    "control": CLControl, 
    "dermatology": CLDermatology, 
    "texture": CLTexture, 
    "micro_imagenet10": CLMicro_ImageNet10, 
    "micro_imagenet20": CLMicro_ImageNet20, 
    "clcifar10": CLCIFAR10, 
    "clcifar20": CLCIFAR20, 
    "clmicro_imagenet10": CLMicro_ImageNet10, 
    "clmicro_imagenet20": CLMicro_ImageNet20, 
}

def prepare_cl_data_module(
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

    dataset_class = D_LIST[dataset]
    cl_data_module = CLDataModule(
        dataset, 
        dataset_class,
        batch_size=batch_size,
        valid_split=valid_split,
        valid_type=valid_type,
        one_hot=one_hot,
        transition_matrix=transition_matrix,
        num_cl=num_cl,
        augment=augment,
        noise=noise,
        seed=seed,
    )
    return cl_data_module