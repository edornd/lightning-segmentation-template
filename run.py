import sys
import dotenv
import hydra
from omegaconf import DictConfig
from pathlib import Path
from typing import List

from src.utils.template import get_logger


# only two run modes
# train instantiates a new hydra config, test runs a previous config
RUN_MODES = ("train", "test")
LOG = get_logger(__name__)

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)


def parse_command(args: List[str]) -> str:
    if not len(args) >= 2:
        LOG.warning("Missing run argument, assuming train.")
        return RUN_MODES[0]
    command = args[1]
    if command not in RUN_MODES:
        LOG.warning(f"Invalid command '{command}', assuming train")
        return RUN_MODES[0]
    return command


@hydra.main(config_path="configs/", config_name="config")
def train(config: DictConfig):
    # internal imports to avoid hydra shenanigans
    from src.train import train
    from src.utils.template import extras, print_config
    # A couple of optional utilities:
    # - disabling python warnings
    # - easier access to debug mode
    # - forcing debug friendly configuration
    # - forcing multi-gpu friendly configuration
    extras(config)
    # Pretty print config using Rich library
    if config.get("print_config"):
        print_config(config, resolve=True)
    # Train model
    return train(config)


def test(args: List[str]):
    from src.test import test
    from src.utils.template import load_old_config, print_config
    # initializing a config from a previous run:
    # - load from directory
    # - delete unwanted (and non-resolvable) nodes, such as loggers
    path = Path(args[2]).absolute()
    config = load_old_config(path=path, delete_fields=["logger"])
    # Pretty printing same as train
    print_config(config)
    # Test model
    return test(config, path)


if __name__ == "__main__":
    run_mode = parse_command(sys.argv)
    if run_mode == "train":
        train()
    elif run_mode == "test":
        test(sys.argv)
