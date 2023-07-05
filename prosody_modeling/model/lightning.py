import lightning.pytorch as pl
import torch
import torch.nn.functional as F
import torchmetrics

from prosody_modeling.model.prosody_model import ProsodyModel


class ProsodyModelLightning(pl.LightningModule):
    def __init__(
        self,
        conv_out_dim=2560,
        rnn_in_dim=768,
        use_deltas=True,
        rnn_layers=2,
        rnn_dropout=0.5,
        features=[
            "pitch_mean_log",
            "pitch_range_log",
            "intensity_mean_vcd",
            "nhr_vcd",
            "rate",
        ],
        lr=0.0001,
        weight_decay=0,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.features = features
        self.lr = lr
        self.weight_decay = weight_decay

        self.prosody_predictor = ProsodyModel(
            conv_out_dim=conv_out_dim,
            rnn_in_dim=rnn_in_dim,
            use_deltas=use_deltas,
            rnn_layers=rnn_layers,
            rnn_dropout=rnn_dropout,
            num_features=len(features),
        )

    def forward(self, mel_spectrogram, mel_spectrogram_len):
        out, _, _, _ = self.prosody_predictor(mel_spectrogram, mel_spectrogram_len)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ExponentialLR(
                    optimizer=optimizer, gamma=0.9
                ),
                "interval": "epoch",
                # "scheduler": torch.optim.lr_scheduler.SequentialLR(
                #     optimizer,
                #     schedulers=[
                #         torch.optim.lr_scheduler.LambdaLR(
                #             optimizer=optimizer, lr_lambda=lambda x: 1
                #         ),
                #         torch.optim.lr_scheduler.ExponentialLR(
                #             optimizer=optimizer, gamma=0.9999
                #         ),
                #     ],
                #     milestones=[50_000],
                # ),
                # "interval": "step",
            },
        }

    def predict_step(self, batch, batch_idx):
        data, metadata = batch

        mel_spectrogram = data["mel_spectrogram"]
        mel_spectrogram_len = metadata["mel_spectrogram_len"]
        y = metadata["features"]

        mel_spectrogram_len = mel_spectrogram_len.squeeze(-1)

        pred_features = self(mel_spectrogram, mel_spectrogram_len)

        return pred_features, y

    def validation_step(self, batch, batch_idx):
        data, metadata = batch

        mel_spectrogram = data["mel_spectrogram"]
        mel_spectrogram_len = metadata["mel_spectrogram_len"]
        y = metadata["features"]

        mel_spectrogram_len = mel_spectrogram_len.squeeze(-1)

        pred_features = self(mel_spectrogram, mel_spectrogram_len)

        loss = F.mse_loss(pred_features, y)
        self.log("val_loss", loss, on_epoch=True, on_step=False)

        ccc = torchmetrics.functional.concordance_corrcoef(pred_features, y)
        for feature, p in zip(self.features, ccc):
            self.log(f"val_{feature}_ccc", p, on_epoch=True, on_step=False)

        return loss

    def training_step(self, batch, batch_idx):
        data, metadata = batch

        mel_spectrogram = data["mel_spectrogram"]
        mel_spectrogram_len = metadata["mel_spectrogram_len"]
        y = metadata["features"]

        mel_spectrogram_len = mel_spectrogram_len.squeeze(-1)

        pred_features = self(mel_spectrogram, mel_spectrogram_len)

        loss = F.mse_loss(pred_features, y)
        self.log("train_loss", loss.detach(), on_epoch=True, on_step=True)

        ccc = torchmetrics.functional.concordance_corrcoef(pred_features.detach(), y)
        for feature, p in zip(self.features, ccc):
            self.log(f"train_{feature}_ccc", p, on_epoch=True, on_step=True)

        return loss
