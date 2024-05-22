from .Strategy import Strategy
from .SCL import SCL
from .URE import URE
from .MCL import MCL
from .FWD import FWD
from .DM import DM
from .CPE import CPE

STRATEGY_LIST = {
    "SCL": SCL,
    "URE": URE,
    "MCL": MCL,
    "FWD": FWD,
    "DM": DM,
    "CPE": CPE,
}


def build_strategy(strategy, **args):
    if strategy not in STRATEGY_LIST:
        raise ValueError(f"Strategy must be chosen from {list(STRATEGY_LIST.keys())}.")
    return STRATEGY_LIST[strategy](**args)
