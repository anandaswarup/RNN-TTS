"""Data loading utilities"""

import os
import random

import config as cfg
import numpy as np
import torch
from torch.utils.data import Dataset


def _load_dataset_instances(data_dir):
    """Load the training instances from disk
    """
    with open(os.path.join(data_dir, "metadata.csv"), "r") as file_reader:
        dataset_instances = file_reader.readlines()

    dataset_instances = [instance.strip("\n") for instance in dataset_instances]

    dataset_instances = [instance.split("|") for instance in dataset_instances]

    dataset_instances = [
        [os.path.join(data_dir, "mel", instance[0] + ".npy"), os.path.join(data_dir, "qwav", instance[0] + ".npy")]
        for instance in dataset_instances
    ]

    return dataset_instances


class WaveRNNDataset(Dataset):
    """Vocoder dataset (Loads and returns mel and quantized wav pairs)
    """

    def __init__(self, data_dir):
        """Instantiate the dataset
        """
        self.dataset_instances = _load_dataset_instances(data_dir)

        self.sample_frames = cfg.vocoder_training["sample_frames"]
        self.hop_length = cfg.audio["hop_length"]

    def __len__(self):
        return len(self.dataset_instances)

    def __getitem__(self, index):
        mel_path, qwav_path = self.dataset_instances[index]

        mel = np.load(mel_path)
        qwav = np.load(qwav_path)

        pos = random.randint(0, mel.shape[-1] - self.sample_frames - 1)
        mel = mel[:, pos : pos + self.sample_frames]

        p, q = pos, pos + self.sample_frames
        qwav = qwav[p * self.hop_length : q * self.hop_length + 1]

        return torch.FloatTensor(mel.T), torch.LongTensor(qwav)
