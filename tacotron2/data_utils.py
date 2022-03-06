"""Data loading utilities"""

import math
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data.sampler as samplers
from text.en.processor import _symbol_to_id, text_to_sequence
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def _load_dataset_instances(data_dir):
    """Load the training instances from disk
    """
    with open(os.path.join(data_dir, "metadata.csv"), "r") as file_reader:
        dataset_instances = file_reader.readlines()

    dataset_instances = [instance.strip("\n") for instance in dataset_instances]

    dataset_instances = [instance.split("|") for instance in dataset_instances]

    data_instances = [
        [os.path.join(data_dir, "mel", instance[0] + ".npy"), instance[1]] for instance in dataset_instances
    ]

    instance_lengths = [instance[2] for instance in dataset_instances]

    return data_instances, instance_lengths


class SortedSampler(samplers.Sampler):
    """Adapted from https://github.com/PetrochukM/PyTorch-NLP
       Copyright (c) James Bradbury and Soumith Chintala 2016,
       All rights reserved.
    """

    def __init__(self, data, sort_key):
        super().__init__(data)

        self.data = data
        self.sort_key = sort_key

        zip_ = [(i, self.sort_key(row)) for i, row in enumerate(self.data)]
        zip_ = sorted(zip_, key=lambda r: r[1], reverse=True)
        self.sorted_indexes = [item[0] for item in zip_]

    def __iter__(self):
        return iter(self.sorted_indexes)

    def __len__(self):
        return len(self.data)


class BucketBatchSampler(samplers.BatchSampler):
    """Adapted from https://github.com/PetrochukM/PyTorch-NLP
       Copyright (c) James Bradbury and Soumith Chintala 2016,
       All rights reserved.
    """

    def __init__(self, sampler, batch_size, drop_last, sort_key, bucket_size_multiplier):
        super().__init__(sampler, batch_size, drop_last)

        self.sort_key = sort_key
        self.bucket_sampler = samplers.BatchSampler(
            sampler, min(batch_size * bucket_size_multiplier, len(sampler)), False
        )

    def __iter__(self):
        for bucket in self.bucket_sampler:
            sorted_sampler = SortedSampler(bucket, self.sort_key)
            for batch in samplers.SubsetRandomSampler(
                list(samplers.BatchSampler(sorted_sampler, self.batch_size, self.drop_last))
            ):
                yield [bucket[idx] for idx in batch]

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return math.ceil(len(self.sampler) / self.batch_size)


class TextMelDataset(Dataset):
    """Dataset class
       (1) Loads text, mel pairs
       (2) Normalizes text and returns a sequence of IDs corresponding to symbols present in the text
    """

    def __init__(self, data_dir, reduction_factor):
        """Instantiate the dataset class
        """
        self.reduction_factor = reduction_factor
        self.instances, self.lengths = _load_dataset_instances(data_dir)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        melpath, text = self.instances[idx][0], self.instances[idx][1]

        mel = np.load(melpath)
        text_seq = text_to_sequence(text)

        return (torch.LongTensor(text_seq), torch.FloatTensor(mel).transpose_(0, 1).contiguous())

    def sort_key(self, idx):
        return self.lengths[idx]

    def pad_collate(self, batch):
        """Create padded batches
        """
        texts, mels = zip(*batch)

        texts = list(texts)
        mels = list(mels)

        # if len(mels[0]) is not a multiple of reduction_factor pad so that it becomes a multiple
        if len(mels[0]) % self.reduction_factor != 0:
            mels[0] = F.pad(mels[0], (0, 0, 0, self.reduction_factor - 1))

        mel_lengths = [len(mel) for mel in mels]
        text_lengths = [len(text) for text in texts]

        texts = pad_sequence(texts, batch_first=True, padding_value=_symbol_to_id["_PAD_"])
        mels = pad_sequence(mels, batch_first=True)

        return texts, text_lengths, mels.transpose_(1, 2).contiguous(), mel_lengths
