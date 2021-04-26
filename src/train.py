from typing import List, Optional

import hydra
from omegaconf import DictConfig
from albumentations import Compose
from pytorch_lightning import LightningModule, LightningDataModule, Callback, Trainer
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.plugins import Plugin
from pytorch_lightning import seed_everything

import src.utils.template as utils

LOG = utils.get_logger(__name__)


def train(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        seed_everything(config.seed)

    # Init augmentations, they require primitive types to be instantiated
    train_augs: Compose = None
    valid_augs: Compose = None
    if "augmentations" in config:
        train_augs = Compose(utils.instantiate_list(config.augmentations.train, group="train augs.", primitive=True))
        valid_augs = Compose(utils.instantiate_list(config.augmentations.valid, group="valid augs.", primitive=True))

    # Init Lightning datamodule
    LOG.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule,
                                                              train_transforms=train_augs,
                                                              valid_transforms=valid_augs)

    # Init Lightning model
    LOG.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model)

    # Init Lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        callbacks = utils.instantiate_list(config.callbacks, group="callback")

    # Init Lightning loggers
    loggers: List[LightningLoggerBase] = []
    if "logger" in config:
        loggers = utils.instantiate_list(config.logger, group="logger")

    # Init trainer plugins
    plugins: List[Plugin] = []
    if "plugin" in config:
        plugins = utils.instantiate_list(config.plugin, group="plugin")

    # Init Lightning trainer
    LOG.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(config.trainer,
                                               callbacks=callbacks,
                                               logger=loggers,
                                               plugins=plugins,
                                               _convert_="partial")

    # Send some parameters from config to all lightning loggers
    LOG.info("Logging hyperparameters")
    utils.log_hyperparameters(config=config,
                              model=model,
                              datamodule=datamodule,
                              trainer=trainer,
                              callbacks=callbacks,
                              logger=loggers)

    # Train the model
    LOG.info("Starting training")
    trainer.fit(model=model, datamodule=datamodule)

    # Evaluate model on test set after training
    if not config.trainer.get("fast_dev_run"):
        LOG.info("Starting testing...")
        trainer.test()

    # Make sure everything closed properly
    LOG.info("Finalizing")
    utils.finish(config=config,
                 model=model,
                 datamodule=datamodule,
                 trainer=trainer,
                 callbacks=callbacks,
                 logger=loggers)

    # Print path to best checkpoint
    LOG.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")

    # Return metric score for Optuna optimization
    optimized_metric = config.get("optimized_metric")
    if optimized_metric:
        return trainer.callback_metrics[optimized_metric]
