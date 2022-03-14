"""Tacotron2 model"""

from math import sqrt

import config as cfg
import torch.nn as nn
from text.en.processor import _symbol_to_id

from tacotron2.seq2seq.decoder import Decoder
from tacotron2.seq2seq.encoder import Encoder


class Tacotron2(nn.Module):
    """Tacotron2 model
    """

    def __init__(self):
        """Instantiate the model
        """
        super().__init__()

        # Embedding layer
        self.embedding_layer = nn.Embedding(
            len(_symbol_to_id), cfg.tts_model["embedding_dim"], padding_idx=_symbol_to_id["_PAD_"]
        )
        std = sqrt(2.0 / (len(_symbol_to_id) + cfg.tts_model["embedding_dim"]))
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

    def forward(self, texts, mels):
        """Forward pass
        """
        embedded_texts = self.embedding_layer(texts)
        memory = self.encoder(embedded_texts)
        output_mels, alignments = self.decoder(mels, memory)

        return output_mels, alignments

    def generate(self, texts):
        """Generate mels from text
        """
        embedded_texts = self.embedding_layer(texts)
        memory = self.encoder(embedded_texts)
        output_mels, alignments = self.decoder.inference(memory)

        return output_mels, alignments

