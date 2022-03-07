"""Common layers"""

import torch.nn as nn
import torch.nn.functional as F


class PreNet(nn.Module):
    """Prenet (used in the decoder as an information bottleneck)

        Args:
            in_dim (int): Size of the input tensor
            prenet_layers (list): Prenet layer sizes
            prenet_dropout (float): Dropout probability to apply to each prenet layer
    """

    def __init__(self, in_dim, prenet_layers, prenet_dropout):
        """Instantiate the PreNet
        """
        super().__init__()

        # Prenet layers
        layer_sizes = [in_dim] + prenet_layers
        self.layers = nn.ModuleList(
            [nn.Linear(in_size, out_size, bias=True) for in_size, out_size in zip(layer_sizes, layer_sizes[1:])]
        )

        self.dropout = prenet_dropout

    def forward(self, x):
        """Forward pass

            Shapes:
                x: [B, in_dim]
                returns: [B, prenet_layers[-1]]
        """
        for layer in self.layers:
            x = F.dropout(F.relu(layer(x)), p=self.dropout, training=True)

        return x


class ConvBatchNorm(nn.Module):
    """Convolution layer + Batch Normalization + Activation + Dropout

        Args:
            in_channels (int): Size of the input tensor
            out_channels (int): Size of the output tensor
            dropout (float): Dropout probabilty to apply
            kernel_size (int): Convolutional kernel size
    """

    def __init__(self, in_channels, out_channels, kernel_size, dropout=0.5, activation=None):
        """Instantiate the layer
        """
        super().__init__()

        assert (kernel_size - 1) % 2 == 0
        padding = (kernel_size - 1) // 2

        # Convolutional layer
        self.conv_layer = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)

        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(out_channels, momentum=0.1, eps=1e-5)

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

