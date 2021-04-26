import os
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pytorch_lightning as pl
import rich
import wandb
from hydra.utils import instantiate
from hydra.experimental import initialize_config_dir, compose
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from rich.syntax import Syntax
from rich.tree import Tree


def get_logger(name=__name__, level=logging.INFO):
    """Initializes python logger."""

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


log = get_logger()


def delete_field(config: DictConfig, key: Any) -> DictConfig:
    """Deletes the given key from the OmegaConf configuration.

    Args:
        config (DictConfig): config object
        key (Any): key to delete, typically string

    Returns:
        DictConfig: the same config, without extra value
    """
    with open_dict(config):
        del config[key]
    return config


def load_old_config(path: Path, delete_fields: List[str] = []) -> DictConfig:
    """Loads a previous config from the `.hydra` folder, optionally removing some attributes.

    Args:
        path (Path): path to the previous run
        delete_fields (List[str], optional): List of fields from the previous run to eliminate. Defaults to [].

    Returns:
        DictConfig: OmegaCof configuration
    """
    initialize_config_dir(config_dir=os.path.join(path, ".hydra"))
    config = compose(config_name="config", return_hydra_config=False)
    for field in delete_fields:
        config = delete_field(config, field)
    config.work_dir = os.path.abspath(path)
    return config


def extras(config: DictConfig) -> None:
    """A couple of optional utilities, controlled by main config file.
        - disabling warnings
        - easier access to debug mode
        - forcing debug friendly configuration
        - forcing multi-gpu friendly configuration
    Args:
        config (DictConfig): [description]
    """

    # enable adding new keys to config
    OmegaConf.set_struct(config, False)

    # disable python warnings if <config.disable_warnings=True>
    if config.get("disable_warnings"):
        log.info(f"Disabling python warnings! <config.disable_warnings=True>")
        warnings.filterwarnings("ignore")

    # set <config.trainer.fast_dev_run=True> if <config.debug=True>
    if config.get("debug"):
        log.info("Running in debug mode! <config.debug=True>")
        config.trainer.fast_dev_run = True

    # force debugger friendly configuration if <config.trainer.fast_dev_run=True>
    if config.trainer.get("fast_dev_run"):
        log.info("Forcing debugger friendly configuration! <config.trainer.fast_dev_run=True>")
        # Debuggers don't like GPUs or multiprocessing
        if config.trainer.get("gpus"):
            config.trainer.gpus = 0
        if config.datamodule.get("num_workers"):
            config.datamodule.num_workers = 0

    # force multi-gpu friendly configuration if <config.trainer.accelerator=ddp>
    if config.trainer.get("accelerator") in ["ddp_spawn", "dp", "ddp2"]:
        log.info("Forcing dp friendly configuration! <config.trainer.accelerator=dp|ddp2|ddp_spawn>")
        # ddp doesn't like num_workers>0 or pin_memory=True
        if config.datamodule.get("num_workers"):
            config.datamodule.num_workers = 0
        if config.datamodule.get("pin_memory"):
            config.datamodule.pin_memory = False

    # disable adding new keys to config
    OmegaConf.set_struct(config, True)


def instantiate_list(config: Dict[str, OmegaConf], group: str = None, primitive: bool = False) -> List[Any]:
    """Iterates over the given dictionary and instantiates the given items.

    Args:
        config (Dict[str, OmegaConf]): list ok <name,params> objects to be instantiated
        group (str): name of the group, just ofr logging purposes
        primitive (bool): whether to first convert to primitive types or not

    Returns:
        List[Any]: list of instantiated objects of any kind
    """
    result = []
    for name, conf in config.items():
        if "_target_" in conf:
            log.info(f"Instantiating {group} <{name}:{conf._target_}")
            if primitive:
                data = OmegaConf.to_container(conf, resolve=True)
                conversion = "partial"
            else:
                data = conf
                conversion = None
            result.append(instantiate(data, _convert_=conversion))
    return result


@rank_zero_only
def print_config(
    config: DictConfig,
    fields: Sequence[str] = (
        "trainer",
        "model",
        "datamodule",
        "augmentations",
        "callbacks",
        "logger",
        "seed",
    ),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Config.
        fields (Sequence[str], optional): Determines which main fields from config will be printed
        and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = Tree(f":gear: CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(Syntax(branch_content, "yaml"))

    rich.print(tree)


def empty(*args, **kwargs):
    pass


@rank_zero_only
def log_hyperparameters(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    """This method controls which parameters from Hydra config are saved by Lightning loggers.

    Additionaly saves:
        - sizes of train, val, test dataset
        - number of trainable model parameters
    """

    hparams = {}

    # choose which parts of hydra config will be saved to loggers
    hparams["trainer"] = config["trainer"]
    hparams["model"] = config["model"]
    hparams["datamodule"] = config["datamodule"]
    if "callbacks" in config:
        hparams["callbacks"] = config["callbacks"]

    # save sizes of each dataset
    # (requires calling `datamodule.setup()` first to initialize datasets)
    # datamodule.setup()
    # if hasattr(datamodule, "data_train") and datamodule.data_train:
    #     hparams["datamodule/train_size"] = len(datamodule.data_train)
    # if hasattr(datamodule, "data_val") and datamodule.data_val:
    #     hparams["datamodule/val_size"] = len(datamodule.data_val)
    # if hasattr(datamodule, "data_test") and datamodule.data_test:
    #     hparams["datamodule/test_size"] = len(datamodule.data_test)

    # save number of model parameters
    hparams["model/params_total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params_trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params_not_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)
    # disable logging any more hyperparameters for all loggers
    # (this is just a trick to prevent trainer from logging hparams of model, since we already did that above)
    trainer.logger.log_hyperparams = empty


def finish(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback] = None,
    logger: List[pl.loggers.LightningLoggerBase] = None,
) -> None:
    """Makes sure everything closed properly."""

    # without this sweeps with wandb logger might crash!
    if logger:
        for lg in logger:
            if isinstance(lg, WandbLogger):
                wandb.finish()
