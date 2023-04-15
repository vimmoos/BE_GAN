import torch.nn as nn
from typing import List, Callable


def conv_block(
    in_channels: int,
    out_channels: int,
    closing_ops: Callable[[], List[nn.Module]] = lambda in_ch, out_ch: [],
    act_fun: Callable[[], nn.Module] = lambda: nn.ELU(True),
):
    return nn.Sequential(
        nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        ),
        act_fun(),
        nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        ),
        act_fun(),
        *closing_ops(in_channels, out_channels),
    )


def conv_block_downsampling(
    in_channels: int,
    out_channels: int,
    act_fun: Callable[[], nn.Module] = lambda: nn.ELU(True),
):
    return conv_block(
        in_channels,
        out_channels,
        closing_ops=lambda in_ch, out_ch: [
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2),
        ],
        act_fun=act_fun,
    )


def conv_block_upsampling(
    in_channels: int,
    out_channels: int,
    act_fun: Callable[[], nn.Module] = lambda: nn.ELU(True),
):
    return conv_block(
        in_channels,
        out_channels,
        act_fun=act_fun,
        closing_ops=lambda in_ch, out_ch: [
            nn.UpsamplingNearest2d(scale_factor=2)
        ],
    )
