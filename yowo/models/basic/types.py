from typing import Literal


ACTIVATION = Literal['relu', 'lrelu', 'mish', 'silu']
NORM = Literal['BN', 'GN']
PADDING_MODE = Literal['ZERO', 'SAME']
NORM_3D = Literal['BN', 'IN']