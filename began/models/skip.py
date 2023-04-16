import torch.nn as nn
import torch
from began.models.utils import (
    conv_block,
)
from began.models.standard import Encoder


class SkipDecoder(nn.Module):
    def __init__(
        self,
        n_filters: int = 32,
        out_channels: int = 3,
        hidden_size: int = 64,
    ):
        super().__init__()
        self.n_filters = n_filters
        self.hidden_size = hidden_size
        self.embedding = nn.Linear(hidden_size, n_filters * 8 * 8)
        self.conv1 = conv_block(n_filters, n_filters)
        self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up_h0 = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv2 = conv_block(
            n_filters, n_filters, in_chs_firstconv=n_filters * 2
        )
        self.up2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv3 = conv_block(
            n_filters,
            3,
            in_chs_firstconv=n_filters * 2,
            closing_ops=lambda in_ch, out_ch: [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                nn.Tanh(),
            ],
        )

    def forward(self, x):
        h0 = self.embedding(x)
        h0 = h0.view(h0.size(0), self.n_filters, 8, 8)

        out = self.conv1(h0)
        out = self.up1(torch.cat([out, h0], dim=1))

        h0 = self.up_h0(h0)

        out = self.conv2(out)
        out = self.up2(torch.cat([out, h0], dim=1))

        return self.conv3(out)


class SkipDiscriminator(nn.Module):
    def __init__(
        self,
        n_filters: int = 32,
        input_ch: int = 3,
        hidden_size: int = 64,
    ):
        super().__init__()
        self.encoder = Encoder(n_filters, input_ch, hidden_size)
        self.decoder = SkipDecoder(n_filters, input_ch, hidden_size)

    def forward(self, x):
        return self.decoder(self.encoder(x))
