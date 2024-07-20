import collections
import functools
import itertools
from typing import Callable, Literal, Optional, OrderedDict, Type
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_2_t


def _infer_device(module: nn.Module) -> Optional[torch.device]:
    """Attempt to infer a module's device by inspecting its parameters and buffers."""
    try:
        p = next(itertools.chain(module.parameters(), module.buffers()))
    except StopIteration:
        return None
    else:
        return p.device


def _recursive_add_module(
        module: nn.Module,
        name_dicts: OrderedDict[nn.Module, list[tuple[nn.Module, str]]]
):
    for name, child in module.named_children():
        if child not in name_dicts:
            name_dicts[child] = []
            _recursive_add_module(child, name_dicts)
        name_dicts[child].append((module, name))


def check_modules(module: nn.Module, list_modules: list[Type[nn.Module]]) -> bool:
    return all([isinstance(module, m) for m in list_modules])


def replace_module_classes(
        module: nn.Module,
        policies: dict[Type[nn.Module], Callable]
):
    replace_pairs = {}
    child_name_map: OrderedDict[nn.Module,
                                list[tuple[nn.Module, str]]] = collections.OrderedDict()
    _recursive_add_module(module, child_name_map)
    indices: dict[Type[nn.Module], int] = {c: 0 for c, _ in policies.items()}

    while len(child_name_map) > 0:
        child, parents = child_name_map.popitem(last=False)
        for policy_class, replacement_fn in policies.items():
            if not isinstance(child, policy_class):
                continue

            # module_index = indices.get(policy_class)
            replacement_module = replacement_fn(child)
            indices[policy_class] += 1
            if replacement_module is not None:
                assert child not in replace_pairs
                device = _infer_device(child)
                if device is not None:
                    replacement_module = replacement_module.to(device)
                replace_pairs[child] = replacement_module
                for parent, name in parents:
                    setattr(parent, name, replacement_module)

    return replace_pairs


def create_blur_filter_2d(filter_size: int = 3) -> torch.Tensor:
    if filter_size == 1:
        a = torch.tensor([1.0, ])
    elif filter_size == 2:
        a = torch.tensor([1.0, 1.0])
    elif filter_size == 3:
        a = torch.tensor([1.0, 2.0, 1.0])
    elif filter_size == 4:
        a = torch.tensor([1.0, 3.0, 3.0, 1.0])
    elif filter_size == 5:
        a = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0])
    elif filter_size == 6:
        a = torch.tensor([1.0, 5.0, 10.0, 10.0, 5.0, 1.0])
    elif filter_size == 7:
        a = torch.tensor([1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0])
    else:
        raise ValueError(
            'Maximum filter size is 7, but get size of %s instead' % filter_size)

    blur_filter = torch.einsum("i,j->ij", a, a)
    blur_filter = blur_filter / torch.sum(blur_filter)
    return blur_filter


def _padding_filter_2d_same(blur_filter: torch.Tensor) -> tuple[int, int]:
    h, w = blur_filter.shape
    if h % 2 == 0:
        raise IndexError(f'Filter must have odd height; got {h}')
    if w % 2 == 0:
        raise IndexError(f'Filter must have odd width; got {w}')

    return int(torch.div(h, 2)), int(torch.div(w, 2))


def _pad_filter_2d_input(
        x: torch.Tensor,
        pad_type: Literal['reflect', 'replicate', 'zero'] = 'reflect',
        filter_size: int = 3
) -> torch.Tensor:
    pad_sizes = (
        int(1.0 * (filter_size - 1) / 2),
        int(np.ceil(1.0 * (filter_size - 1) / 2)),
        int(1.0 * (filter_size - 1) / 2),
        int(np.ceil(1.0 * (filter_size - 1) / 2)),
    )

    if pad_type == "reflect":
        PadLayer = nn.ReflectionPad2d(pad_sizes)
    elif pad_type == "replicate":
        PadLayer = nn.ReplicationPad2d(pad_sizes)
    elif pad_type == "zero":
        PadLayer = nn.ZeroPad2d(pad_sizes)
    else:
        raise ValueError("Pad type [%s] not recognized" % pad_type)

    return PadLayer(x)


def blur_2d(
        x: torch.Tensor,
        blur_filter: torch.Tensor,
        channels: int = -1,
        stride: _size_2_t = 1,
        padding: bool = False,
) -> torch.Tensor:
    # blur_filter = create_blur_filter_2d(filter_size)
    if channels < 1:
        _, channels, h, w = x.shape

    filter_size = blur_filter.size(-1)
    _blur_filter = blur_filter[None, None, ...].repeat(channels, 1, 1, 1)
    if filter_size == 1:
        if padding:
            x = _pad_filter_2d_input(
                x,
                pad_type="reflect",
                filter_size=filter_size
            )
        return x[..., ::stride, ::stride] if isinstance(stride, int) else x[..., ::stride[0], ::stride[1]]
    else:
        return F.conv2d(
            input=_pad_filter_2d_input(
                x,
                pad_type="reflect",
                filter_size=filter_size
            ),
            weight=_blur_filter,
            stride=stride,
            groups=x.size(1),
        )


def blur_max_pool2d(
        x: torch.Tensor,
        blur_filter: torch.Tensor,
        kernel_size: _size_2_t = (2, 2),
        stride: _size_2_t = 2,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        ceil_mode: bool = False
) -> torch.Tensor:
    out = F.max_pool2d(
        x,
        kernel_size=kernel_size,
        padding=padding,
        stride=1,
        dilation=dilation,
        ceil_mode=ceil_mode
    )
    return blur_2d(
        out,
        blur_filter=blur_filter,
        channels=-1,
        stride=stride,
        padding=False
    )


class BlurMaxPool2d(nn.Module):
    def __init__(
            self,
            kernel_size: _size_2_t,
            filter_size: int = 3,
            stride: _size_2_t = 2,
            padding: _size_2_t = 0,
            dilation: _size_2_t = 1,
            ceil_mode: bool = False,
    ):
        super(BlurMaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if (stride is not None) else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        blur_filter = create_blur_filter_2d(filter_size)
        self.register_buffer('blur_filter', blur_filter)

    def forward(self, x: torch.Tensor):
        return blur_max_pool2d(
            x,
            blur_filter=self.blur_filter,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            ceil_mode=self.ceil_mode
        )

    @staticmethod
    def from_max_pool_2d(
            module: nn.MaxPool2d
    ) -> 'BlurMaxPool2d':
        return BlurMaxPool2d(
            kernel_size=module.kernel_size,
            filter_size=3,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            ceil_mode=module.ceil_mode
        )


class BlurConv2d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: _size_2_t,
            filter_size: int = 3,
            stride: _size_2_t = None,
            padding: _size_2_t = 0,
            dilation: _size_2_t = 1,
            groups: int = 1,
            bias: bool = True,
            blur_first: bool = True,
    ):

        super(BlurConv2d, self).__init__()
        self.blur_first = blur_first

        if self.blur_first:
            assert stride is not None
            conv_stride = stride
            self.blur_stride = 1
            self.blur_channels = in_channels
        else:
            conv_stride = 1
            self.blur_stride = kernel_size if (stride is None) else stride
            self.blur_channels = out_channels

        self.conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=conv_stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        con_attrs = {
            k: v for k, v in self.conv.__dict__.items()
            if not k.startswith('_')
        }
        self.__dict__.update(con_attrs)
        # self.in_channels = self.conv.in_channels
        # self.out_channels = self.conv.out_channels
        # self.kernel_size = self.conv.kernel_size
        # self.stride = self.conv.stride
        # self.padding = self.conv.padding
        # self.dilation = self.conv.dilation
        # self.groups = self.conv.groups
        self.bias = self.conv.bias
        self.weight = self.conv.weight

        self.conv._already_blur_pooled = True
        blur_filter_conv = create_blur_filter_2d(filter_size)
        self.register_buffer(
            "blur_filter_conv",
            blur_filter_conv
        )

    def forward(self, x: torch.Tensor):
        if self.blur_first:
            blurred = blur_2d(
                x,
                blur_filter=self.blur_filter_conv,
                channels=self.blur_channels,
                stride=self.blur_stride
            )
            return self.conv.forward(blurred)
        else:
            activations = self.conv.forward(x)
            return blur_2d(
                activations,
                channels=self.blur_channels,
                blur_filter=self.blur_filter_conv,
                stride=self.blur_stride
            )

    @staticmethod
    def from_conv_2d(
            module: nn.Conv2d,
            blur_first: bool = True
    ):
        has_bias = module.bias is not None and module.bias is not False
        blur_conv = BlurConv2d(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=has_bias,
            blur_first=blur_first,
        )
        with torch.no_grad():
            blur_conv.conv.weight.copy_(module.weight)
            if has_bias:
                blur_conv.conv.bias.copy_(module.bias)

        return blur_conv


class BlurPool2d(nn.Module):
    """This module just calls :func:`.blur_2d` in ``forward`` using the provided arguments."""

    def __init__(
            self,
            channels: int = 0,
            filter_size: int = 3,
            stride: _size_2_t = 2,
            padding: _size_2_t = 1
    ) -> None:
        super(BlurPool2d, self).__init__()
        self.channels = channels
        self.stride = stride
        self.padding = padding
        # self.register_buffer('blur_filter', create_blur_filter_2d(filter_size))
        self.blur_filter = create_blur_filter_2d(filter_size)
        if self.channels > 0:
            self.blur_filter = self.blur_filter.repeat(channels, 1, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return blur_2d(x, channels=self.channels, stride=self.stride, blur_filter=self.blur_filter)


def apply_blurpool(
        model: torch.nn.Module,
        replace_conv: bool = True,
        replace_max_pool: bool = True,
        blur_first: bool = True,
        min_channels: int = 16
) -> None:
    transforms: dict[Type[nn.Module], Callable] = {}
    if replace_max_pool:
        transforms[torch.nn.MaxPool2d] = BlurMaxPool2d.from_max_pool_2d
    if replace_conv:
        transforms[torch.nn.Conv2d] = functools.partial(
            _maybe_replace_stride_conv2d,
            blur_first=blur_first,
            min_channels=min_channels,
        )
    replace_module_classes(model, policies=transforms)


def _maybe_replace_stride_conv2d(
        module: torch.nn.Conv2d,
        # module_index: int,
        blur_first: bool,
        min_channels: int = 16,
):
    already_blur_pooled = hasattr(
        module, '_already_blur_pooled') and module._already_blur_pooled
    if np.max(module.stride) > 1 and module.in_channels >= min_channels and not already_blur_pooled:
        return BlurConv2d.from_conv_2d(module, blur_first=blur_first)
    return None
