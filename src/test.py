import os
from pathlib import Path

import hydra
from omegaconf import DictConfig
from albumentations import Compose
from pytorch_lightning import LightningModule, LightningDataModule, Callback, Trainer
from pytorch_lightning import seed_everything
import torch

import src.utils.template as utils

LOG = utils.get_logger(__name__)


def load_checkpoint(current_dir: Path, checkpoint_path: Path = "checkpoints"):
    path = os.path.join(current_dir, checkpoint_path)
    checkpoints = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".ckpt")]
    best_checkpoint = checkpoints[0]
    current_epoch = -1
    for filename in checkpoints:
        if filename.endswith("last.ckpt"):
            continue
        LOG.info(f"Loading checkpoint {os.path.basename(filename)}...")
        state = torch.load(filename, map_location="cpu")
        state_epoch = state["epoch"]
        if state_epoch > current_epoch:
            current_epoch = state_epoch
            best_checkpoint = filename
    return best_checkpoint


def test(config: DictConfig, cwd: Path) -> None:
    """Contains the testing pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Args:
        config (DictConfig): Configuration composed by Hydra.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        seed_everything(config.seed)
    # Init augmentations, they require primitive types to be instantiated
    test_augs: Compose = None
    if "augmentations" in config:
        test_augs = Compose(utils.instantiate_list(config.augmentations.valid, group="test augs.", primitive=True))

    # Init Lightning datamodule
    LOG.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule,
                                                              train_transforms=None,
                                                              valid_transforms=test_augs)

    # Init Lightning model
    LOG.info("Loading latest checkpoint")
    last_checkpoint = load_checkpoint(cwd)
    LOG.info(f"Last checkpoint found: {os.path.basename(last_checkpoint)}")

    LOG.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model)
    model = model.load_from_checkpoint(last_checkpoint, map_location=None)
    model.eval()
    model.freeze()

    # Init Lightning trainer
    LOG.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(config.trainer,
                                               gpus=[5],
                                               auto_select_gpus=False,
                                               accelerator=None,
                                               callbacks=None,
                                               logger=None,
                                               _convert_="partial")

    # Evaluate model on test set
    LOG.info("Starting testing...")
    out = trainer.test(model=model, ckpt_path=last_checkpoint, datamodule=datamodule)
    trainer.pred
    print(len(out))

    # Make sure everything closed properly
    LOG.info("Finalizing")
    utils.finish(config=config,
                 model=model,
                 datamodule=datamodule,
                 trainer=trainer)
    LOG.info("Done!")
