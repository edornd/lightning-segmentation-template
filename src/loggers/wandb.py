import wandb
import logging
from typing import Optional
from pytorch_lightning.loggers import WandbLogger


class SilentWandbLogger(WandbLogger):
    """Wandb logger wrapper that updates the log visibility from the client after initialization.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        level = kwargs.get("log_level", logging.WARNING)
        logging.getLogger(wandb.__name__).setLevel(level)
