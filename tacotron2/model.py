"""Tacotron2 model"""

from math import sqrt

import config as cfg
import torch.nn as nn
from text.en.processor import symbol_to_id

from tacotron2.seq2seq.decoder import Decoder
from tacotron2.seq2seq.encoder import Encoder
from tacotron2.seq2seq.postnet import PostNet


class Tacotron2(nn.Module):
    """Tacotron2 model
    """

    def __init__(self):
        """Instantiate the model
        """
        super().__init__()

        # Embedding layer
        self.embedding_layer = nn.Embedding(
            len(symbol_to_id), cfg.tts_model["embedding_dim"], padding_idx=symbol_to_id["_PAD_"]
        )
        std = sqrt(2.0 / (len(symbol_to_id) + cfg.tts_model["embedding_dim"]))
        val = sqrt(3.0) * std
        self.embedding_layer.weight.data.uniform_(-val, val)

        # Encoder
        self.encoder = Encoder(
            cfg.tts_model["embedding_dim"],
            cfg.tts_model["encoder"]["n_conv_layers"],
            cfg.tts_model["encoder"]["n_conv_filters"],
            cfg.tts_model["encoder"]["conv_filter_size"],
            cfg.tts_model["encoder"]["conv_dropout"],
            cfg.tts_model["encoder"]["blstm_size"],
        )

        # Decoder
        self.decoder = Decoder(
            cfg.audio["n_mels"],
            cfg.tts_model["encoder"]["blstm_size"],
            cfg.tts_model["decoder"]["prenet"],
            cfg.tts_model["decoder"]["prenet_dropout"],
            cfg.tts_model["decoder"]["attn_lstm_size"],
            cfg.tts_model["attention"]["attn_dim"],
            cfg.tts_model["attention"]["n_static_filters"],
            cfg.tts_model["attention"]["static_filter_size"],
            cfg.tts_model["attention"]["n_dynamic_filters"],
            cfg.tts_model["attention"]["dynamic_filter_size"],
            cfg.tts_model["attention"]["prior_filter_len"],
            cfg.tts_model["attention"]["alpha"],
            cfg.tts_model["attention"]["beta"],
            cfg.tts_model["decoder"]["decoder_lstm_size"],
            cfg.tts_model["decoder"]["lstm_dropout"],
            cfg.tts_model["decoder"]["reduction_factor"],
        )

        # Post Processing Network
        self.postnet = PostNet(
            cfg.audio["n_mels"],
            cfg.tts_model["postnet"]["n_conv_layers"],
            cfg.tts_model["postnet"]["n_conv_filters"],
            cfg.tts_model["postnet"]["conv_filter_size"],
            cfg.tts_model["postnet"]["conv_dropout"],
        )

    def forward(self, texts, mels):
        """Forward pass
        """
        # Embedding layer
        embedded_texts = self.embedding_layer(texts)

        # Text Encoder
        memory = self.encoder(embedded_texts)

        # Autoregressive Decoder
        output_mels, alignments = self.decoder(mels, memory)

        # Post Processing Network
        postnet_output_mels = self.postnet(output_mels)
        postnet_output_mels = output_mels + postnet_output_mels

        return postnet_output_mels, output_mels, alignments

    def generate(self, texts):
        """Generate mels from text
        """
        # Embedding layer
        embedded_texts = self.embedding_layer(texts)

        # Text Encoder
        memory = self.encoder(embedded_texts)

        # Autoregressive Decoder
        output_mels, alignments = self.decoder.inference(memory)

        # Post Processing Network
        postnet_output_mels = self.postnet(output_mels)
        postnet_output_mels = output_mels + postnet_output_mels

        return postnet_output_mels, output_mels, alignments

