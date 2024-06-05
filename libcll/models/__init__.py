import inspect
from .Linear import Linear
from .MLP import MLP
from .ResNet import ResNet18, ResNet34
from .DenseNet import DenseNet

MODEL_LIST = {
    "Linear": Linear,
    "MLP": MLP,
    "ResNet18": ResNet18,
    "ResNet34": ResNet34,
    "DenseNet": DenseNet,
}


def build_model(model, input_dim=None, hidden_dim=None, num_classes=None):
    if model not in MODEL_LIST:
        raise ValueError(f"Model must be chosen from {list(MODEL_LIST.keys())}.")
    args = {
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "num_classes": num_classes,
    }
    model = MODEL_LIST[model]
    model_args = inspect.getargspec(model.__init__).args
    return model(**{arg: args[arg] for arg in args if arg in model_args})
