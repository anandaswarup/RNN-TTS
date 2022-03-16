"""Configuration Parameters"""

dataset = "LJSpeech"  # The name of the dataset

# Audio processing configuration
audio = {
    "sampling_rate": 22050,
    "max_db": 100,
    "ref_db": 20,
    "n_fft": 2048,
    "win_length": 1100,  # 50 ms window length
    "hop_length": 275,  # 12.5 ms frame shift
    "n_mels": 80,
    "fmin": 50,
    "n_bits": 10,  # The bit depth of the signal
}

# TTS configuration
tts_model = {
    "embedding_dim": 512,
    # Encoder
    "encoder": {
        "n_conv_layers": 3,
        "n_conv_filters": 512,
        "conv_filter_size": 5,
        "conv_dropout": 0.5,
        "blstm_size": 512,
    },
    # Dynamic convolutional attention
    "attention": {
        "attn_dim": 128,
        "n_static_filters": 8,
        "static_filter_size": 21,
        "n_dynamic_filters": 8,
        "dynamic_filter_size": 21,
        "prior_filter_len": 11,
        "alpha": 0.1,
        "beta": 0.9,
    },
    # Autoregressive decoder
    "decoder": {
        "prenet": [256, 256],
        "prenet_dropout": 0.5,
        "attn_lstm_size": 1024,
        "decoder_lstm_size": 1024,
        "lstm_dropout": 0.1,
        "reduction_factor": 2,
    },
}

tts_training = {
    "batch_size": 128,
    "bucket_size_multiplier": 5,
    "n_steps": 250000,
    "checkpoint_interval": 10000,
    "lr": 1e-3,
    "lr_scheduler_milestones": [20000, 40000, 100000, 150000, 200000],
    "lr_scheduler_gamma": 0.5,
    "clip_grad_norm": 0.05,
}

# Vocoder configuration
vocoder_model = {
    "audio_embedding_dim": 256,
    "conditioning_rnn_size": 128,
    "rnn_size": 896,
    "fc_size": 1024,
}

vocoder_training = {
    "batch_size": 32,
    "n_steps": 250000,
    "sample_frames": 24,
    "lr": 4e-4,
    "lr_scheduler_step_size": 25000,
    "lr_scheduler_gamma": 0.5,
    "checkpoint_interval": 10000,
}
