"""Text to Speech Synthesis"""

import argparse
import os

import soundfile as sf
import torch

import config as cfg
from tacotron2.model import Tacotron2
from text.en.processor import text_to_sequence
from wavernn.model import WaveRNNModel


def _load_synthesis_instances(synthesis_file):
    """Load the training instances from disk
    """
    with open(synthesis_file, "r") as file_reader:
        synthesis_instances = file_reader.readlines()

    synthesis_instances = [instance.strip("\n") for instance in synthesis_instances]

    synthesis_instances = [instance.split("|") for instance in synthesis_instances]

    return synthesis_instances


def load_trained_model(checkpoint_path, model):
    """Load checkpoint from the specified path
    """
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint["model"])
    model.eval()

    return model, checkpoint["step"]


def synthesize(synthesis_file, tacotron2_checkpoint, waveRNN_checkpoint, out_dir):
    """Synthesize text present in the synthesis file
    """
    # Create directories
    os.makedirs(out_dir, exist_ok=True)

    # Specify the device on which to perform the synthesis
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained Tacotron2 model
    print("Loading trained Tacotron2 model")
    tacotron2_model = Tacotron2()
    tacotron2_model = tacotron2_model.to(device)
    tacotron2_model, tacotron2_step = load_trained_model(tacotron2_checkpoint, tacotron2_model)

    # Load the trained WaveRNN model
    print("Loading trained WaveRNN model")
    waveRNN_model = WaveRNNModel()
    waveRNN_model = waveRNN_model.to(device)
    waveRNN_model, waveRNN_step = load_trained_model(waveRNN_checkpoint, waveRNN_model)

    # Get the synthesis instances
    synthesis_instances = _load_synthesis_instances(synthesis_file)

    # Synthesis loop
    for file_id, text in synthesis_instances:
        print(f"Synthesizing {file_id}", flush=True)

        # Convert text to sequence of IDs corresponding the symbols present in the text
        text_seq = torch.LongTensor(text_to_sequence(text)).unsqueeze(0)
        text_seq = text_seq.to(device)

        # Synthesize audio
        with torch.no_grad():
            mel, _ = tacotron2_model.generate(text_seq)
            wav_hat = waveRNN_model.generate(mel.transpose(1, 2).contiguous())

        # Write the synthesized wav file to disk
        wav_path = os.path.join(out_dir, f"{file_id}-tacotron2_{tacotron2_step}steps-WaveRNN_{waveRNN_step}steps.wav")
        sf.write(wav_path, wav_hat, cfg.audio["sampling_rate"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text to Speech Synthesis")

    parser.add_argument("--synthesis_file", help="Path to the file containing text to be synthesized", required=True)

    parser.add_argument(
        "--tacotron2_checkpoint", help="Path to the trained Tacotron2 model to be used for synthesis", required=True
    )

    parser.add_argument(
        "--waveRNN_checkpoint", help="Path to the trained WaveRNN model to be used for synthesis", required=True
    )

    parser.add_argument(
        "--out_dir", help="Path to where the synthesized waveforms will be written to disk", required=True
    )

    args = parser.parse_args()

    synthesis_file = args.synthesis_file
    tacotron2_checkpoint = args.tacotron2_checkpoint
    waveRNN_checkpoint = args.waveRNN_checkpoint
    out_dir = args.out_dir

    synthesize(synthesis_file, tacotron2_checkpoint, waveRNN_checkpoint, out_dir)
