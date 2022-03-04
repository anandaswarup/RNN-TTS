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


class PreNet(nn.Module):
    """Prenet (used in the decoder as an information bottleneck)
    """

    def __init__(self, in_dim, prenet_layers, prenet_dropout):
        """Instantiate the PreNet
        """
        super().__init__()

        layer_sizes = [in_dim] + prenet_layers
        self.layers = nn.ModuleList(
            [
                Linear(in_size, out_size, bias=True, w_init_gain="relu")
                for in_size, out_size in zip(layer_sizes, layer_sizes[1:])
            ]
        )

        self.dropout = prenet_dropout

    def forward(self, x):
        """Forward pass
        """
        for layer in self.layers:
            x = F.dropout(F.relu(layer(x)), p=self.dropout, training=True)

        return x


class ConvBatchNorm(nn.Module):
    """Convolution layer + Batch Normalization + Activation + Dropout
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
        dropout=0.5,
        activation=None,
        w_init_gain="linear",
    ):
        """Instantiate the layer
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

        # Activation
        self.activation = activation

        # Dropout
        self.dropout = dropout

    def forward(self, x):
        """Forward pass
        """
        x = self.activation(self.batch_norm(self.conv_layer(x)))
        x = F.dropout(x, p=self.dropout, training=self.training)

        return x

