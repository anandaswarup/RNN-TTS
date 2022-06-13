"""Tacotron2 Post Processing Network"""

import torch.nn as nn
import torch.nn.functional as F
from tacotron2.layers.common import ConvBatchNorm


class PostNet(nn.Module):
    """Post Processing Network
    """

    def __init__(self, n_mels, n_conv_layers, n_conv_filters, conv_filter_size, conv_dropout):
        """Instantiate the PostNet
        """
        super().__init__()

        # Convolutional layers
        conv_filters = [n_mels] + [n_conv_filters] * (n_conv_layers - 1) + [n_mels]
        conv_activations = [nn.Tanh()] * (n_conv_layers - 1) + [None]
        self.convs = nn.ModuleList(
            [
                ConvBatchNorm(
                    in_filters, out_filters, kernel_size=conv_filter_size, dropout=conv_dropout, activation=act
                )
                for in_filters, out_filters, act in zip(conv_filters, conv_filters[1:], conv_activations)
            ]
        )

    def forward(self, x):
        """Forward pass

            Args:
                x: [B, n_mels, T]
                returns: [B, n_mels, T]
        """
        for conv in self.convs:
            x = conv(x)

        return x
