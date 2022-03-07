"""Train the Tacotron model"""

import argparse
import os
import random

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.sampler as samplers
from torch.utils.data import DataLoader

import config as cfg

from tacotron2.data_utils import BucketBatchSampler, TextMelDataset
from tacotron2.model import Tacotron2

matplotlib.use("Agg")


def save_checkpoint(checkpoint_dir, model, optimizer, scaler, scheduler, step):
    """Save checkpoint to disk
    """
    checkpoint_state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
    }

    checkpoint_path = os.path.join(checkpoint_dir, f"model_step{step:09d}.pth")

    torch.save(checkpoint_state, checkpoint_path)

    print(f"Written checkpoint: {checkpoint_path} to disk")


def load_checkpoint(checkpoint_path, model, optimizer, scaler, scheduler):
    """Load checkpoint from the specified path
    """
    print(f"Loading checkpoint: {checkpoint_path} from disk")
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint["scaler"])
    scheduler.load_state_dict(checkpoint["scheduler"])

    return checkpoint["step"]


def log_alignment(alignment, path):
    """Log the alignment
    """
    _ = plt.figure(figsize=(10, 6))
    plt.imshow(alignment, vmin=0, vmax=0.6, origin="lower")
    plt.xlabel("Decoder steps")
    plt.ylabel("Encoder steps")

    plt.savefig(path)


def prepare_dataloaders(data_dir):
    """Prepare the dataloaders
    """
    # Train dataloader
    train_dataset = TextMelDataset(
        data_dir=os.path.join(data_dir, "train"), reduction_factor=cfg.tts_model["decoder"]["reduction_factor"]
    )
    sampler = samplers.RandomSampler(train_dataset)
    batch_sampler = BucketBatchSampler(
        sampler=sampler,
        batch_size=cfg.tts_training["batch_size"],
        drop_last=True,
        sort_key=train_dataset.sort_key,
        bucket_size_multiplier=cfg.tts_training["bucket_size_multiplier"],
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_sampler=batch_sampler,
        collate_fn=train_dataset.pad_collate,
        num_workers=8,
        pin_memory=True,
    )

    # Validation dataloader
    val_dataset = TextMelDataset(
        data_dir=os.path.join(data_dir, "val"), reduction_factor=cfg.tts_model["decoder"]["reduction_factor"]
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=1,
        collate_fn=val_dataset.pad_collate,
        pin_memory=False,
        drop_last=True,
    )

    return train_dataloader, val_dataloader


def validate(model, val_dataloader, global_step, alignment_dir):
    """Validate the model
    """
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for idx, (texts, text_lengths, mels, mel_lengths) in enumerate(val_dataloader):
            output_mels, alignments = model(texts, mels)
            loss = F.mse_loss(output_mels[:, :, : mels.size(-1)], mels)

            val_loss += loss.item()
        val_loss = val_loss / (idx + 1)
    model.train()

    print(f"Val Loss: {val_loss:.6f}", flush=True)

    # Log alignment
    index = random.randint(0, alignments.size(0) - 1)
    alignment = alignments[
        index, : text_lengths[index], : mel_lengths[index] // cfg.tts_model["decoder"]["reduction_factor"]
    ]
    alignment = alignment.detach().cpu().numpy()
    alignment_path = os.path.join(alignment_dir, f"model_step{global_step:09d}.png")
    log_alignment(alignment, alignment_path)


def train_model(data_dir, checkpoint_dir, alignment_dir, resume_checkpoint_path):
    """Train the model
    """
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)

    # Create directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(alignment_dir, exist_ok=True)

    # Specify the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the model
    model = Tacotron2()
    model = model.to(device)

    # Instantiate the optimizer, scaler (for mixed precision training) and scheduler (for learning rate decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.tts_training["lr"])
    scaler = amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer,
        milestones=cfg.tts_training["lr_scheduler_milestones"],
        gamma=cfg.tts_training["lr_scheduler_gamma"],
    )

    # Instantiate the dataloader
    train_dataloader, val_dataloader = prepare_dataloaders(data_dir)

    # Load checkpoint and resume training from that point (if specified)
    if resume_checkpoint_path is not None:
        global_step = load_checkpoint(resume_checkpoint_path, model, optimizer, scaler, scheduler)
    else:
        global_step = 0

    n_epochs = cfg.tts_training["n_steps"] // len(train_dataloader) + 1
    start_epoch = global_step // len(train_dataloader) + 1

    model.train()

    # Main training loop
    for epoch in range(start_epoch, n_epochs + 1):
        print(f"Epoch: {epoch}", flush=True)
        for _, (texts, _, mels, _) in enumerate(train_dataloader, 1):
            texts, mels = texts.to(device), mels.to(device)

            model.zero_grad()

            # Forward pass and loss computation
            with amp.autocast():
                output_mels, _ = model(texts, mels)
                loss = F.mse_loss(output_mels[:, :, : mels.size(-1)], mels)

            # Gradient computation
            scaler.scale(loss).backward()

            # Gradient clipping
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.tts_training["clip_grad_norm"])

            # Weights update
            scaler.step(optimizer)

            # Scaler state update
            scaler.update()

            # lr scheduler state update
            scheduler.step()

            global_step += 1

            print(
                f"Step: {global_step}, Loss: {loss.item():.6f}, Grad: {grad_norm:.6f}, LR: {scheduler.get_last_lr()}",
                flush=True,
            )

            if global_step % cfg.tts_training["checkpoint_interval"] == 0:
                validate(model, val_dataloader, global_step, alignment_dir)
                save_checkpoint(checkpoint_dir, model, optimizer, scaler, scheduler, global_step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Tacotron model")

    parser.add_argument("--data_dir", help="Path to processed dataset to be used to train the model", required=True)

    parser.add_argument(
        "--checkpoint_dir", help="Path to location where training checkpoints will be saved", required=True
    )

    parser.add_argument("--alignment_dir", help="Path to location where model alignments will be saved", required=True)

    parser.add_argument(
        "--resume_checkpoint_path", help="If specified load checkpoint and resume training from that point"
    )

    args = parser.parse_args()

    data_dir = args.data_dir
    checkpoint_dir = args.checkpoint_dir
    alignment_dir = args.alignment_dir
    resume_checkpoint_path = args.resume_checkpoint_path

    train_model(data_dir, checkpoint_dir, alignment_dir, resume_checkpoint_path)
