import torch
import torchaudio
from torch import nn


class ProsodyModel(nn.Module):
    def __init__(
        self,
        conv_out_dim,
        rnn_in_dim,
        use_deltas,
        rnn_layers,
        rnn_dropout,
        num_features,
    ):
        super().__init__()

        self.use_deltas = use_deltas
        self.delta = torchaudio.transforms.ComputeDeltas()

        self.conv = nn.Sequential(
            nn.Conv2d(
                3 if use_deltas else 1,
                128,
                (5, 3),
                padding=(2, 1),
            ),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4)),
            nn.Conv2d(128, 256, (5, 3), (1, 1), padding=(2, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, (5, 3), (1, 1), padding=(2, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, (5, 3), (1, 1), padding=(2, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, (5, 3), (1, 1), padding=(2, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, (5, 3), (1, 1), padding=(2, 1)),
            nn.LeakyReLU(),
        )

        self.pre_rnn = nn.Sequential(
            nn.Linear(conv_out_dim, rnn_in_dim), nn.LeakyReLU()
        )

        self.rnn = nn.GRU(
            rnn_in_dim,
            128,
            batch_first=True,
            bidirectional=True,
            num_layers=rnn_layers,
            dropout=rnn_dropout,
        )

        self.frame_weights = nn.Sequential(
            nn.Linear(256, 1), nn.Sigmoid(), nn.Linear(1, 1)
        )

        self.features_out = nn.Sequential(
            nn.Linear(256, 64), nn.LeakyReLU(), nn.Linear(64, num_features)
        )

    def forward(self, mel_spectrogram, mel_spectrogram_len):
        batch_size = mel_spectrogram.shape[0]

        mel_spectrogram = mel_spectrogram.swapaxes(1, 2)
        if mel_spectrogram.shape[2] % 2 == 1:
            mel_spectrogram = torch.cat(
                [
                    mel_spectrogram,
                    torch.zeros(
                        (mel_spectrogram.shape[0], mel_spectrogram.shape[1], 1),
                        device=mel_spectrogram.device,
                    ),
                ],
                2,
            )

        if self.use_deltas:
            d1 = self.delta(mel_spectrogram)
            d2 = self.delta(d1)

            x = torch.cat(
                [mel_spectrogram.unsqueeze(1), d1.unsqueeze(1), d2.unsqueeze(1)], dim=1
            ).swapaxes(2, 3)
        else:
            x = mel_spectrogram.unsqueeze(1).swapaxes(2, 3)

        output = self.conv(x)
        output = output.permute(0, 2, 3, 1).reshape(
            mel_spectrogram.shape[0],
            mel_spectrogram.shape[2],
            -1,  # 256 * 10
        )

        output_low = output

        output = self.pre_rnn(output)

        rnn_input = nn.utils.rnn.pack_padded_sequence(
            output, mel_spectrogram_len.cpu(), batch_first=True, enforce_sorted=False
        )
        output, _ = self.rnn(rnn_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        output_middle = output

        weights = self.frame_weights(output).squeeze(-1)

        mask = torch.arange(mel_spectrogram_len.max(), device=mel_spectrogram.device)
        mask = mask.repeat(batch_size, 1)
        mask = mask < mel_spectrogram_len.unsqueeze(1)

        weights = torch.masked_fill(weights, ~mask, -float("inf"))
        weights = torch.softmax(weights, dim=1)

        output_high = torch.bmm(weights.unsqueeze(1), output).squeeze(1)

        return (
            self.features_out(output_high),
            output_low,
            output_middle,
            output_high,
        )
