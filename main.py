from typing import Optional

import click
from click import Context

from run.train import do_train
from run.test import do_test


@click.group()
@click.pass_context
@click.option(
    "--device",
    type=int,
    required=False,
    default=0,
    help="The GPU to use for training or inference. Default 0.",
)
def main(ctx: Context, device: int):
    ctx.obj["device"] = device


@main.command()
@click.pass_context
@click.option(
    "--speech-dir",
    required=True,
    type=str,
    help="A directory containing audio files from the dataset.",
)
@click.option(
    "--train-csv",
    required=True,
    type=str,
    help="A CSV with training data.",
)
@click.option(
    "--val-csv",
    required=True,
    type=str,
    help="A CSV with validation data.",
)
@click.option(
    "--results-dir",
    required=False,
    type=str,
    help="The directory to save results. Defaults to the model configuration name with a timestamp.",
)
@click.option(
    "--resume-ckpt",
    required=False,
    type=str,
    help="Resume training from the given checkpoint.",
)
def train(
    ctx: Context,
    speech_dir: str,
    train_csv: str,
    val_csv: str,
    results_dir: Optional[str] = None,
    resume_ckpt: Optional[str] = None,
):
    do_train(
        device=ctx.obj["device"],
        speech_dir=speech_dir,
        train_csv=train_csv,
        val_csv=val_csv,
        results_dir=results_dir,
        resume_ckpt=resume_ckpt,
    )


@main.command()
@click.pass_context
@click.option(
    "--speech-dir",
    required=True,
    type=str,
    help="A directory containing audio files from the dataset.",
)
@click.option(
    "--test-csv",
    required=True,
    type=str,
    help="A CSV with test data.",
)
@click.option(
    "--checkpoint",
    required=True,
    type=str,
    help="A trained model to test.",
)
def test(
    ctx: Context,
    speech_dir: str,
    test_csv: str,
    checkpoint: str = None,
):
    do_test(
        device=ctx.obj["device"],
        speech_dir=speech_dir,
        test_csv=test_csv,
        checkpoint=checkpoint,
    )


if __name__ == "__main__":
    main(obj={})
