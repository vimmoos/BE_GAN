import torch.nn as nn
from began.models.utils import (
    conv_block,
    conv_block_upsampling,
    conv_block_downsampling,
)


class Encoder(nn.Module):
    def __init__(
        self,
        n_filters: int = 32,
        input_ch: int = 3,
        hidden_size: int = 64,
    ):
        super().__init__()
        self.n_filters = n_filters
        self.embedding = nn.Linear(8 * 8 * 3 * n_filters, hidden_size)
        self.layers = nn.Sequential(
            nn.Conv2d(input_ch, n_filters, kernel_size=3, stride=1, padding=1),
            nn.ELU(True),
            conv_block_downsampling(n_filters, 2 * n_filters),
            conv_block_downsampling(2 * n_filters, 3 * n_filters),
            conv_block(3 * n_filters, 3 * n_filters),
        )

    def forward(self, x):
        out = self.layers(x)
        return self.embedding(
            out.view(out.size(0), self.n_filters * 3 * 8 * 8)
        )


class Decoder(nn.Module):
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
        self.layers = nn.Sequential(
            conv_block_upsampling(n_filters, n_filters),
            conv_block_upsampling(n_filters, n_filters),
            conv_block(
                n_filters,
                3,
                closing_ops=lambda in_ch, out_ch: [
                    nn.Conv2d(
                        in_ch, out_ch, kernel_size=3, stride=1, padding=1
                    ),
                    nn.Tanh(),
                ],
            ),
        )

    def forward(self, x):
        out = self.embedding(x)
        return self.layers(out.view(out.size(0), self.n_filters, 8, 8))


class Discriminator(nn.Module):
    def __init__(
        self,
        n_filters: int = 32,
        input_ch: int = 3,
        hidden_size: int = 64,
    ):
        super().__init__()
        self.encoder = Encoder(n_filters, input_ch, hidden_size)
        self.decoder = Decoder(n_filters, input_ch, hidden_size)

    def forward(self, x):
        return self.decoder(self.encoder(x))
