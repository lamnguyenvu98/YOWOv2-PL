from typing import get_args

def validate_literal_types(value, types):
    if value not in get_args(types):
        raise ValueError(f"{value} is not supported.\nSupported value are {get_args(types)}")