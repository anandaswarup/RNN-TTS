"""Dynamic Convolution Attention"""

import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import betabinom
import numpy as np
import torch


class DynamicConvolutionAttention(nn.Module):
    """Dynamic Convolution Attention

        Args:
            query_dim (int): Size of the query tensor
            attn_dim (int): Size of the attention hidden state
            n_static_filters (int): Number of static filters
            static_filter_size (int): Size of the each static filter
            n_dynamic_filters (int): Number of dynamic filters
            dynamic_filter_size (int): Size of each dynamic filter
            prior_filter_len (int): Size of the prior filter
            alpha (float): Prior filter parameter
            beta (float): Prior filter parameter
    """

    def __init__(
        self,
        query_dim,
        attn_dim,
        n_static_filters,
        static_filter_size,
        n_dynamic_filters,
        dynamic_filter_size,
        prior_filter_len,
        alpha,
        beta,
    ):
        """Instantiate the attention layer
        """
        super().__init__()

        self.prior_filter_len = prior_filter_len
        self.n_dynamic_filters = n_dynamic_filters
        self.dynamic_filter_size = dynamic_filter_size

        self.attention_weights = None

        # Prior filter
        prior_filter = betabinom.pmf(np.arange(prior_filter_len), prior_filter_len - 1, alpha, beta)
        self.register_buffer("prior_filter", torch.FloatTensor(prior_filter).flip(0))

        # Key and Query layers
        self.W = nn.Linear(query_dim, attn_dim)
        self.V = nn.Linear(attn_dim, n_dynamic_filters * dynamic_filter_size, bias=False)

        # Static filter computation
        self.F = nn.Conv1d(1, n_static_filters, static_filter_size, padding=(static_filter_size - 1) // 2, bias=False)

        # Static filter layer
        self.U = nn.Linear(n_static_filters, attn_dim, bias=False)

        # Dynamic filter layer
        self.T = nn.Linear(n_dynamic_filters, attn_dim)

        # Score layer
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def _init_attention(self, memory):
        """Initialize the attention
        """
        B, T_enc, _ = memory.size()

        self.attention_weights = torch.zeros([B, T_enc], device=memory.device)
        self.attention_weights[:, 0] = 1.0

    def forward(self, query, memory):
        """Forward pass

            Shapes:
                query: [B, query_dim]
                memory: [B, T_enc, memory_dim]
                returns: [B, memory_dim]
        """
        # Compute prior filters
        p = F.conv1d(
            F.pad(self.attention_weights.unsqueeze(1), (self.prior_filter_len - 1, 0)), self.prior_filter.view(1, 1, -1)
        )
        p = torch.log(p.clamp_min_(1e-6)).squeeze(1)

        G = self.V(torch.tanh(self.W(query)))

        # Compute dynamic filters
        g = F.conv1d(
            self.attention_weights.unsqueeze(0),
            G.view(-1, 1, self.dynamic_filter_size),
            padding=(self.dynamic_filter_size - 1) // 2,
            groups=query.size(0),
        )
        g = g.view(query.size(0), self.n_dynamic_filters, -1).transpose(1, 2).contiguous()

        # Compute static filters
        f = self.F(self.attention_weights.unsqueeze(1)).transpose(1, 2).contiguous()

        # Compute attention weights (normalized energies)
        energy = self.v(torch.tanh(self.U(f) + self.T(g))).squeeze(-1) + p
        attention_weights = F.softmax(energy, dim=-1)
        self.attention_weights = attention_weights

        # Compute context
        context = torch.bmm(attention_weights.unsqueeze(1), memory).squeeze(1)

        return context
