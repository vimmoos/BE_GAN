import torch.nn as nn
from typing import List, Callable, Optional


def conv_block(
    in_channels: int,
    out_channels: int,
    closing_ops: Callable[[], List[nn.Module]] = lambda in_ch, out_ch: [],
    act_fun: Callable[[], nn.Module] = lambda: nn.ELU(inplace=True),
    out_chs_firstconv: Optional[int] = None,
    out_chs_secondconv: Optional[int] = None,
    in_chs_firstconv: Optional[int] = None,
    in_chs_secondconv: Optional[int] = None,
    in_chs_closing: Optional[int] = None,
):
    out_chs_firstconv = out_chs_firstconv or in_channels
    out_chs_secondconv = out_chs_secondconv or in_channels
    in_chs_secondconv = in_chs_secondconv or in_channels
    in_chs_closing = in_chs_closing or in_channels
    in_chs_firstconv = in_chs_firstconv or in_channels

    return nn.Sequential(
        nn.Conv2d(
            in_chs_firstconv,
            out_chs_firstconv,
            kernel_size=3,
            stride=1,
            padding=1,
        ),
        act_fun(),
        nn.Conv2d(
            in_chs_secondconv,
            out_chs_secondconv,
            kernel_size=3,
            stride=1,
            padding=1,
        ),
        act_fun(),
        *closing_ops(in_chs_closing, out_channels),
    )


def conv_block_downsampling(
    in_channels: int,
    out_channels: int,
    act_fun: Callable[[], nn.Module] = lambda: nn.ELU(inplace=True),
    out_chs_firstconv: Optional[int] = None,
    out_chs_secondconv: Optional[int] = None,
    in_chs_secondconv: Optional[int] = None,
    in_chs_closing: Optional[int] = None,
):
    return conv_block(
        in_channels,
        out_channels,
        closing_ops=lambda in_ch, out_ch: [
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2),
        ],
        act_fun=act_fun,
        out_chs_firstconv=out_chs_firstconv,
        out_chs_secondconv=out_chs_secondconv,
        in_chs_secondconv=in_chs_secondconv,
        in_chs_closing=in_chs_closing,
    )


def conv_block_upsampling(
    in_channels: int,
    out_channels: int,
    act_fun: Callable[[], nn.Module] = lambda: nn.ELU(inplace=True),
    out_chs_firstconv: Optional[int] = None,
    out_chs_secondconv: Optional[int] = None,
    in_chs_secondconv: Optional[int] = None,
    in_chs_closing: Optional[int] = None,
):
    return conv_block(
        in_channels,
        out_channels,
        act_fun=act_fun,
        closing_ops=lambda in_ch, out_ch: [
            nn.UpsamplingNearest2d(scale_factor=2)
        ],
        out_chs_firstconv=out_chs_firstconv,
        out_chs_secondconv=out_chs_secondconv,
        in_chs_secondconv=in_chs_secondconv,
        in_chs_closing=in_chs_closing,
    )
