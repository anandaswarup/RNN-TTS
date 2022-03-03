"""Dynamic Convolution Attention"""

import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import betabinom
import numpy as np
import torch


class DynamicConvolutionAttention(nn.Module):
    """Dynamic Convolution Attention
    """

    def __init__(
        self,
        query_dim,
        attn_dim,
        n_static_filters,
        static_kernel_size,
        n_dynamic_filters,
        dynamic_kernel_size,
        prior_filter_len,
        alpha,
        beta,
    ):
        """Instantiate the attention layer
        """
        super().__init__()

        self.prior_filter_len = prior_filter_len
        self.n_dynamic_filters = n_dynamic_filters
        self.dynamic_kernel_size = dynamic_kernel_size

        prior_filter = betabinom.pmf(np.arange(prior_filter_len), prior_filter_len - 1, alpha, beta)
        self.register_buffer("prior_filter", torch.FloatTensor(prior_filter).flip(0))

        self.W = nn.Linear(query_dim, attn_dim)
        self.V = nn.Linear(attn_dim, n_dynamic_filters * dynamic_kernel_size, bias=False)

        self.F = nn.Conv1d(1, n_static_filters, static_kernel_size, padding=(static_kernel_size - 1) // 2, bias=False)

        self.U = nn.Linear(n_static_filters, attn_dim, bias=False)
        self.T = nn.Linear(n_dynamic_filters, attn_dim)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def compute_energies(self, query, prev_alignment):
        """Compute energies from the query
        """
        p = F.conv1d(F.pad(prev_alignment.unsqueeze(1), (self.prior_filter_len - 1, 0)), self.prior.view(1, 1, -1))
        p = torch.log(p.clamp_min_(1e-6)).squeeze(1)

        G = self.V(torch.tanh(self.W(query)))
        g = F.conv1d(
            prev_alignment.unsqueeze(0),
            G.view(-1, 1, self.dynamic_kernel_size),
            padding=(self.dynamic_kernel_size - 1) // 2,
            groups=query.size(0),
        )
        g = g.view(query.size(0), self.n_dynamic_filters, -1).transpose(1, 2).contiguous()

        f = self.F(prev_alignment.unsqueeze(1)).transpose(1, 2).contiguous()

        energy = self.v(torch.tanh(self.U(f) + self.T(g))).squeeze(-1) + p

        return energy

    def normalize_energies(self, energies):
        """Normalize the energies by converting to probabilities
        """
        alignment = F.softmax(energies, dim=1)

        return alignment

    def forward(self, query, prev_alignment):
        """Forward pass

            Args:
                query: [B, query_dim]
                prev_alignment: [B, T_enc]
        """
        alignment = self.compute_energies(query, prev_alignment)
        alignment = self.normalize_energies(alignment)

        return alignment
