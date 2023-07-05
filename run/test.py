from typing import List, Tuple

import pandas as pd
import torch
import torchmetrics
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from torch import Tensor
from torch.nn import functional as F

from consts import FEATURES, FEATURES_NORM
from datasets.dataloader import ProsodyDataLoader
from datasets.dataset import ProsodyDataset
from prosody_modeling.model.lightning import ProsodyModelLightning


def do_test(
    device: int,
    speech_dir: str,
    test_csv: str,
    checkpoint: str,
):
    print("Loading CSV data...")

    test_df = pd.read_csv(test_csv, engine="c")
    print(test_csv)

    test_dataset = ProsodyDataset(
        filenames=test_df.path.values.tolist(),
        features=test_df[FEATURES_NORM].values.tolist(),
        base_dir=speech_dir,
    )

    del test_df

    test_dataloader = ProsodyDataLoader(
        dataset=test_dataset,
        batch_size=32,
        num_workers=8,
        pin_memory=True,
        shuffle=False,
        persistent_workers=True,
        drop_last=False,
    )

    torch.set_float32_matmul_precision("high")

    model = ProsodyModelLightning.load_from_checkpoint(checkpoint, features=FEATURES)

    trainer = Trainer(
        logger=None,
        devices=[device],
        accelerator="gpu",
    )

    results: List[Tuple[Tensor, Tensor]] = trainer.predict(
        model, dataloaders=test_dataloader
    )

    y_hat_all_lst = []
    y_all_lst = []

    for y_hat, y in results:
        y_hat_all_lst.append(y_hat)
        y_all_lst.append(y)

    y_hat_all = torch.concat(y_hat_all_lst)
    y_all = torch.concat(y_all_lst)

    print("Loss", F.mse_loss(y_hat_all, y_all))

    ccc = torchmetrics.functional.concordance_corrcoef(y_hat_all, y_all)
    for feature, p in zip(FEATURES, ccc):
        print(feature, p)
