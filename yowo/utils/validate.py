from typing import get_args
import torch

def validate_literal_types(value, types):
    if value not in get_args(types):
        raise ValueError(f"{value} is not supported.\nSupported value are {get_args(types)}")

def deprecated_verbose_scheduler(verbose: str):
    version = torch.__version__
    if version <= "2.1.2":
        return True if verbose.lower() == "true" else False
    return verbose