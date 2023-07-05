import datetime
import os
from os import path
from typing import Optional

import pandas as pd
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger

from consts import FEATURES_NORM
from datasets.dataloader import ProsodyDataLoader
from datasets.dataset import ProsodyDataset
from prosody_modeling.model.lightning import ProsodyModelLightning


def do_train(
    device: int,
    speech_dir: str,
    train_csv: str,
    val_csv: str,
    results_dir: Optional[str] = None,
    resume_ckpt: Optional[str] = None,
):
    if results_dir is None:
        results_dir = f"results_{datetime.datetime.now()}"
        os.mkdir(results_dir)

    print("Loading CSV data...")
    train_df = pd.read_csv(train_csv, engine="c")
    train_df = train_df[train_df.duration <= 30]

    val_df = pd.read_csv(val_csv, engine="c")
    val_df = val_df[val_df.duration <= 30]

    train_dataset = ProsodyDataset(
        filenames=train_df.path.values.tolist(),
        features=train_df[FEATURES_NORM].values.tolist(),
        base_dir=speech_dir,
    )

    val_dataset = ProsodyDataset(
        filenames=val_df.path.values.tolist(),
        features=val_df[FEATURES_NORM].values.tolist(),
        base_dir=speech_dir,
    )

    del train_df
    del val_df

    train_dataloader = ProsodyDataLoader(
        dataset=train_dataset,
        batch_size=32,
        num_workers=8,
        pin_memory=True,
        shuffle=True,
        persistent_workers=True,
        drop_last=True,
    )

    val_dataloader = ProsodyDataLoader(
        dataset=val_dataset,
        batch_size=32,
        num_workers=8,
        pin_memory=True,
        shuffle=False,
        persistent_workers=True,
        drop_last=False,
    )

    torch.set_float32_matmul_precision("high")

    logger = TensorBoardLogger(
        path.join(results_dir, "lightning_logs"), name="prosody_model"
    )

    model = ProsodyModelLightning(features=FEATURES_NORM)

    model_checkpoint = ModelCheckpoint(
        save_top_k=5,
        monitor="val_loss",
        mode="min",
        filename="checkpoint-{epoch}-{val_loss}",
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=25,
        verbose=False,
        mode="min",
    )

    trainer = Trainer(
        logger=logger,
        devices=[device],
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            model_checkpoint,
            early_stopping,
        ],
        accelerator="gpu",
        precision="16-mixed",
        gradient_clip_val=1.0,
        val_check_interval=0.0625,
        num_sanity_val_steps=0,
        # max_epochs=4,
    )

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=resume_ckpt,
    )
