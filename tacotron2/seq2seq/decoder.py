"""Tacotron2 seq2seq Decoder"""

import torch.nn as nn
import torch.nn.functional as F
import torch
from tacotron2.layers.common import PreNet, Linear
from tacotron2.layers.dca import DynamicConvolutionAttention


class Decoder(nn.Module):
    """Autoregressive Decoder
    """

    def __init__(
        self,
        n_mels,
        memory_dim,
        prenet,
        prenet_dropout,
        attn_lstm_size,
        attn_dim,
        n_static_filters,
        static_filter_size,
        n_dynamic_filters,
        dynamic_filter_size,
        prior_filter_len,
        alpha,
        beta,
        decoder_lstm_size,
        lstm_dropout,
        reduction_factor,
    ):
        """Instantiate the autoregressive decoder
        """
        super().__init__()

        self.n_mels = n_mels
        self.memory_dim = memory_dim
        self.attn_lstm_size = attn_lstm_size
        self.decoder_lstm_size = decoder_lstm_size
        self.lstm_dropout = lstm_dropout
        self.reduction_factor = reduction_factor

        # PreNet
        self.prenet = PreNet(n_mels, prenet, prenet_dropout)

        # Attention LSTM
        self.attn_lstm = nn.LSTMCell(prenet[-1] + memory_dim, attn_lstm_size)

        # Attention Mechanism
        self.attention = DynamicConvolutionAttention(
            attn_lstm_size,
            attn_dim,
            n_static_filters,
            static_filter_size,
            n_dynamic_filters,
            dynamic_filter_size,
            prior_filter_len,
            alpha,
            beta,
        )

        # Decoder LSTM
        self.decoder_lstm = nn.LSTMCell(attn_lstm_size + memory_dim, decoder_lstm_size)

        # Output Layer
        self.acoustic_layer = Linear(decoder_lstm_size + memory_dim, n_mels * reduction_factor, bias=True)

    def _init_states(self, memory):
        """Initialize the decoder states
        """
        B, T_enc, _ = memory.size()

        self.alignment = F.one_hot(torch.zeros([B], dtype=torch.long, device=memory.device), T_enc).float()
        self.attention_context = torch.zeros([B, self.memory_dim], device=memory.device)

        self.attn_lstm_hx = [
            torch.zeros([B, self.attn_lstm_size], device=memory.device),
            torch.zeros([B, self.attn_lstm_size], device=memory.device),
        ]

        self.decoder_lstm_hx = [
            torch.zeros([B, self.decoder_lstm_size], device=memory.device),
            torch.zeros([B, self.decoder_lstm_size], device=memory.device),
        ]

    def _get_go_frame(self, memory):
        """Get all zeros go frame to start the decoding
        """
        B = memory.size(0)

        go_frame = torch.zeros([B, self.n_mels], device=memory.device)

        return go_frame

    def step(self, mel_frame, memory):
        """Decoder step
        """
        B, N = mel_frame.size()
        assert N == self.n_mels

        # Prenet
        mel_frame = self.prenet(mel_frame)

        # Attention LSTM (Attention LSTM generates the query to compute the alignment for the current step)
        attn_lstm_input = torch.cat((self.attention_context, mel_frame), dim=-1)
        self.attn_lstm_hx[0], self.attn_lstm_hx[1] = self.attn_lstm(
            attn_lstm_input, (self.attn_lstm_hx[0], self.attn_lstm_hx[1])
        )
        self.attn_lstm_hx[0] = F.dropout(self.attn_lstm_hx[0], p=self.lstm_dropout, training=self.training)

        # Attention mechanism computes the alignment for the current timestep using the query and previous timestep
        # alignment
        alignment = self.attention(self.attn_lstm_hx[0], self.alignment)
        self.attention_context = torch.matmul(alignment.unsqueeze(1), memory).squeeze(1)
        self.alignment = alignment

        # Decoder LSTM
        decoder_lstm_input = torch.cat((self.attn_lstm_hx[0], self.attention_context), dim=-1)
        self.decoder_lstm_hx[0], self.decoder_lstm_hx[1] = self.decoder_lstm(
            decoder_lstm_input, (self.decoder_lstm_hx[0], self.decoder_lstm_hx[1])
        )
        self.decoder_lstm_hx[0] = F.dropout(self.decoder_lstm_hx[0], p=self.lstm_dropout, training=self.training)

        # Output Layers
        decoder_lstm_output = torch.cat((self.decoder_lstm_hx[0], self.attention_context), dim=-1)
        output_mel_frame = self.acoustic_layer(decoder_lstm_output).view(B, N, self.reduction_factor)

        return output_mel_frame, alignment

    def forward(self, mels, memory):
        """Forward pass (Training in teacher forcing mode: previous timestep ground-truth is used as input while 
        decoding the current timstep)
        """
        T_mels = mels.size(-1)
        mels = mels.unbind(-1)

        self._init_states(memory)
        go_frame = self._get_go_frame(memory)

        output_mels, alignments = [], []
        for t in range(0, T_mels, self.reduction_factor):
            mel_frame = mels[t - 1] if t > 0 else go_frame
            output_mel_frame, alignment = self.step(mel_frame, memory)

            output_mels.append(output_mel_frame)
            alignments.append(alignment)

        output_mels = torch.cat(output_mels, dim=-1)
        alignments = torch.stack(alignments, dim=2)

        return output_mels, alignments

    def inference(self, memory, max_length=10000, stop_threshold=-0.2):
        """Inference mode (Generates mels using greedy decoding)
        """
        self._init_states(memory)
        go_frame = self._get_go_frame(memory)

        output_mels, alignments = [], []
        for t in range(0, max_length, self.reduction_factor):
            mel_frame = output_mels[-1][:, :, -1] if t > 0 else go_frame
            output_mel_frame, alignment = self.step(mel_frame, memory)

            if torch.all(mel_frame[:, :, -1] > stop_threshold):
                break

            output_mels.append(output_mel_frame)
            alignments.append(alignment)

        output_mels = torch.cat(output_mels, dim=-1)
        alignments = torch.stack(alignments, dim=2)

        return output_mels, alignments

