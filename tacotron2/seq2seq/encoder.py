"""Tacotron2 seq2seq Encoder"""

import torch.nn as nn
import torch.nn.functional as F
from tacotron2.layers.common import ConvBatchNorm


class Encoder(nn.Module):
    """Text Encoder
    """

    def __init__(self, embedding_dim, n_conv_layers, n_conv_filters, conv_filter_size, conv_dropout, blstm_size):
        """Instantiate the Encoder
        """
        super().__init__()

        # Convolutional layers
        conv_filters = [embedding_dim] + [n_conv_filters] * n_conv_layers
        conv_activations = [nn.ReLU()] * n_conv_layers
        self.convs = nn.ModuleList(
            [
                ConvBatchNorm(
                    in_filters,
                    out_filters,
                    kernel_size=conv_filter_size,
                    bias=True,
                    dropout=conv_dropout,
                    activation=act,
                    w_init_gain="relu",
                )
                for in_filters, out_filters, act in zip(conv_filters, conv_filters[1:], conv_activations)
            ]
        )

        # BLSTM layer
        self.blstm = nn.LSTM(n_conv_filters, blstm_size // 2, num_layers=1, batch_first=True, bidirectional=True)

    def forward(self, x):
        """Forward pass

            Args:
                x: [B, T, embedding_dim]
                returns: [B, T, blstm_size]
        """
        # Convolutional layers
        x = x.transpose(1, 2).contiguous()
        for conv in self.convs:
            x = conv(x)
        x = x.transpose(1, 2).contiguous()

        # BLSTM layer
        outputs, _ = self.blstm(x)

        return outputs
