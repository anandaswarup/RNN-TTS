"""Audio processing utilities used in the TTS and Vocoder pipelines
"""

import librosa
import numpy as np
import scipy

import config as cfg


def _sound_norm(wav):
    """Normalize the volume of the audio signal
    """
    return wav / np.abs(wav).max() * 0.95


def _trim_silence(wav):
    """Trim silences from the begining and ending of the signal
    """
    margin = int(cfg.audio["sampling_rate"] * 0.01)
    wav = wav[margin:-margin]

    return librosa.effects.trim(
        wav, top_db=cfg.audio["trim_db"], frame_length=cfg.audio["win_length"], hop_length=cfg.audio["hop_length"]
    )[0]


def load_wav(wavpath):
    """Load the wav file from disk
    """
    wav, _ = librosa.load(wavpath, sr=cfg.audio["sampling_rate"])
    wav = _trim_silence(wav)
    wav = _sound_norm(wav)

    return wav


def save_wav(wav, wavpath):
    """Write a signal to disk as a wav file
    """
    wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))

    scipy.io.wavfile.write(wavpath, cfg.audio["sampling_rate"], wav_norm.astype(np.int16))


def mulaw_compression(wav):
    """Compress the waveform using mu-law compression
    """
    wav = np.pad(wav, (cfg.audio["win_length"] // 2,), mode="reflect")
    wav = wav[: ((wav.shape[0] - cfg.audio["win_length"]) // cfg.audio["hop_length"] + 1) * cfg.audio["hop_length"]]

    wav = 2 ** (cfg.audio["n_bits"] - 1) + librosa.mu_compress(wav, mu=2 ** cfg.audio["n_bits"] - 1)

    return wav


def inv_mulaw_compression(wav):
    """Reverse mu-law compression and regain the full waveform
    """
    wav = librosa.mu_expand(wav - 2 ** (cfg.audio["n_bits"] - 1), mu=2 ** cfg.audio["n_bits"] - 1)

    return wav


def compute_melspectrogram(wav):
    """Compute mel-spectrogram from waveform
    """
    # Apply pre-emphasis
    wav = librosa.effects.preemphasis(wav, coef=0.97)

    # Compute the mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=wav,
        sr=cfg.audio["sampling_rate"],
        hop_length=cfg.audio["hop_length"],
        win_length=cfg.audio["win_length"],
        n_fft=cfg.audio["n_fft"],
        n_mels=cfg.audio["n_mels"],
        fmin=cfg.audio["fmin"],
        norm=1,
        power=1,
    )

    # Convert to log scale
    mel = librosa.core.amplitude_to_db(mel, top_db=None) - cfg.audio["ref_db"]

    # Normalize
    mel = np.maximum(mel, -cfg.audio["max_db"])
    mel = mel / cfg.audio["max_db"]

    return mel
