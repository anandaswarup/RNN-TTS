"""Common layers"""

import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Module):
    """Linear layer with Xavier Uniform initialization
    """

    def __init__(self, in_features, out_features, bias=True, w_init_gain="linear"):
        """Instantiate the layer
        """
        super().__init__()

        self.linear_layer = nn.Linear(in_features, out_features, bias=bias)
        nn.init.xavier_uniform_(self.linear_layer.weight, gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        """Forward pass
        """
        return self.linear_layer(x)


class BatchNormConv(nn.Module):
    """Convolution layer + Batch Normalization
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
        w_init_gain="linear",
    ):
        """Instantiate the layeer
        """
        super().__init__()

        if padding is None:
            assert kernel_size % 2 == 1
            padding = dilation * (kernel_size - 1) // 2

        # Convolutional layer
        self.conv_layer = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        nn.init.xavier_uniform_(self.conv_layer.weight, gain=nn.init.calculate_gain(w_init_gain))

        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        """Forward pass
        """
        return self.batch_norm(self.conv_layer(x))

