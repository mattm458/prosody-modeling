import random
from os import path

import librosa
import torch
import torchaudio
from speech_utils.audio.transforms import TacotronMelSpectrogram
from torch.utils.data import Dataset


class ProsodyDataset(Dataset):
    """A class implementing a text-to-speech PyTorch Dataset. It supplies Mel spectrograms and
    textual data for a text-to-speech model."""

    def __init__(
        self,
        filenames,
        features,
        base_dir,
        n_mels=80,
    ):
        super().__init__()

        # Simple assignments
        self.filenames = filenames
        self.features = features
        self.base_dir = base_dir

        # Create a Torchaudio MelSpectrogram generator
        self.melspectrogram = TacotronMelSpectrogram(n_mels=n_mels)
        self.resample = torchaudio.transforms.Resample(orig_freq=48000, new_freq=22050)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, i):
        # Audio preprocessing -----------------------------------------------------------
        # Load the audio file and squeeze it to a 1D Tensor
        wav, _ = librosa.load(
            path.join(self.base_dir, self.filenames[i]),
            sr=self.melspectrogram.sample_rate,
        )
        wav = torch.tensor(wav)
        wav = wav.squeeze(0)

        # Create the Mel spectrogram and save its length
        mel_spectrogram = self.melspectrogram(wav)
        mel_spectrogram_len = torch.IntTensor([len(mel_spectrogram)])

        out_data = {
            "mel_spectrogram": mel_spectrogram,
        }

        out_metadata = {
            "mel_spectrogram_len": mel_spectrogram_len,
            "features": torch.Tensor([self.features[i]]),
        }

        return out_data, out_metadata
