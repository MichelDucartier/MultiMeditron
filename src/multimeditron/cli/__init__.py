import click
import logging

logger = logging.getLogger(__name__)

EPILOG = """
This tools is part of the MultiMeditron project,
made by the LiGHT group at EPFL."""

def ensure_config_empty_dict(config, key):
    if key not in config or config[key] is None:
        config[key] = {}


@click.group(epilog=EPILOG)
def main_cli():
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] -- %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

from .preprocess import *
from .train import *
