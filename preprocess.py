"""Preprocess dataset and split into train/val/test splits"""

import argparse
import os

import librosa
import numpy as np

import config as cfg


def _split_dataset(items):
    """Split the dataset items into train/val/test splits
    """
    np.random.seed(1234)

    split_size = int(len(items) * 0.01)

    np.random.shuffle(items)
    train_val_split = items[split_size:]
    test_split = items[:split_size]

    np.random.shuffle(train_val_split)
    train_split = train_val_split[split_size:]
    val_split = train_val_split[:split_size]

    return train_split, val_split, test_split


def _compute_melspectrogram(wav):
    """Compute the mel-spectrogram
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


def _mulaw_compression(wav):
    """Compress the waveform using mu-law compression
    """
    wav = np.pad(wav, (cfg.audio["win_length"] // 2,), mode="reflect")
    wav = wav[: ((wav.shape[0] - cfg.audio["win_length"]) // cfg.audio["hop_length"] + 1) * cfg.audio["hop_length"]]

    wav = 2 ** (cfg.audio["n_bits"] - 1) + librosa.mu_compress(wav, mu=2 ** cfg.audio["n_bits"] - 1)

    return wav


def _process_utterance(wav, filename, mel_dir, qwav_dir):
    """Process a single wav file
    """
    peak = np.abs(wav).max()
    if peak >= 1:
        wav = wav / peak * 0.999

    # Compute mel spectrogram
    mel = _compute_melspectrogram(wav)

    # Quantize the wavform
    qwav = _mulaw_compression(wav)

    # Save to disk
    mel_path = os.path.join(mel_dir, filename + ".npy")
    qwav_path = os.path.join(qwav_dir, filename + ".npy")

    np.save(mel_path, mel)
    np.save(qwav_path, qwav)

    return mel.shape[-1]


def write_metadata(metadata, out_file):
    """Write the metadata to file
    """
    with open(out_file, "w") as file_writer:
        for m in metadata:
            file_writer.write("|".join([str(x) for x in m]) + "\n")


def preprocess_dataset(root_dir, out_dir):
    """Process dataset and write processed dataset to disk
    """
    # Load dataset items from disk
    items = []
    with open(os.path.join(root_dir, "metadata.csv"), "r") as file_reader:
        for line in file_reader:
            parts = line.strip().split("|")
            text = parts[1]
            wav_path = os.path.join(root_dir, "wavs", f"{parts[0]}.wav")
            items.append([text, wav_path])

    # Split into train/val/test sets
    train_items, val_items, test_items = _split_dataset(items)

    # Process the train split
    print("Processing train split")

    train_mel_dir = os.path.join(out_dir, "train", "mel")
    train_qwav_dir = os.path.join(out_dir, "train", "qwav")

    os.makedirs(train_mel_dir, exist_ok=True)
    os.makedirs(train_qwav_dir, exist_ok=True)

    train_metadata = []
    for text, wav_path in train_items:
        # Get filename of file being processed
        filename = os.path.splitext(os.path.basename(wav_path))[0]

        # Load wav file from disk
        wav, _ = librosa.load(wav_path, sr=cfg.audio["sampling_rate"])

        num_frames = _process_utterance(wav, filename, train_mel_dir, train_qwav_dir)
        train_metadata.append((filename, text, num_frames))
    write_metadata(train_metadata, os.path.join(out_dir, "train/metadata.csv"))

    # Process the val split
    print("Processing val split")

    val_mel_dir = os.path.join(out_dir, "val", "mel")
    val_qwav_dir = os.path.join(out_dir, "val", "qwav")

    os.makedirs(val_mel_dir, exist_ok=True)
    os.makedirs(val_qwav_dir, exist_ok=True)

    val_metadata = []
    for text, wav_path in val_items:
        # Get filename of file being processed
        filename = os.path.splitext(os.path.basename(wav_path))[0]

        # Load wav file from disk
        wav, _ = librosa.load(wav_path, sr=cfg.audio["sampling_rate"])

        num_frames = _process_utterance(wav, filename, val_mel_dir, val_qwav_dir)
        val_metadata.append((filename, text, num_frames))
    write_metadata(val_metadata, os.path.join(out_dir, "val/metadata.csv"))

    # Process the test split
    print("Processing test split")

    test_dir = os.path.join(out_dir, "test")
    os.makedirs(test_dir, exist_ok=True)

    test_metadata = []
    for text, wav_path in test_items:
        filename = os.path.splitext(os.path.basename(wav_path))[0]
        test_metadata.append((filename, text))
    write_metadata(test_metadata, os.path.join(out_dir, "test/heldout.csv"))


def preprocess(root_dir, out_dir):
    """Preprocess the dataset
    """
    os.makedirs(out_dir, exist_ok=True)

    if cfg.dataset == "LJSpeech":
        preprocess_dataset(root_dir, out_dir)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess dataset and create train and eval splits")

    parser.add_argument("--dataset_dir", help="Path to the root of the downloaded dataset", required=True)

    parser.add_argument("--out_dir", help="Output path to write the processed dataset", required=True)

    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    out_dir = args.out_dir

    preprocess(dataset_dir, out_dir)
