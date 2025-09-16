from .config import VerlTrainConfig
from .utils import split_host_port

import click
import hydra
import random
import yaml
import ray
import logging
import os

from pydanclick import from_pydantic
from pprint import pprint

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

@main_cli.command(epilog=EPILOG)
@click.option("--output", "-o", type=click.Path(), default=None, help="Path to save the generated configuration file.")
def config_verl_generate(output):
    """
    Generate a default configuration file for training the MultiMeditron model.
    """
    cfg = VerlTrainConfig()
    if output is not None:
        with open(output, "w") as f:
            yaml.dump(cfg.model_dump(), f)
        print(f"Configuration file saved to {output}")
    else:
        print(yaml.dump(cfg.model_dump()))

@main_cli.command(epilog=EPILOG)
@click.option("--config", "-c", type=click.Path(exists=True), multiple=True, help="Path to the configuration file(s) in YAML format.")
@click.option("--trust-remote-code/--no-trust-remote-code", default=False, help="Whether to trust remote code when loading models from HuggingFace.")
@from_pydantic("cfg", VerlTrainConfig)
def train_verl(cfg: VerlTrainConfig, config: tuple[str] = [], trust_remote_code: bool = False):
    """
    Train the MultiMeditron model using the specified configuration file.
    """
    # Load and merge configuration files from YAML
    for cfg_path in config:
        logger.info(f"Loading configuration from {cfg_path}")
        with open(cfg_path, "r") as f:
            file_cfg = yaml.safe_load(f)

            # Merge configurations
            cli_cfg = cfg.model_dump(exclude_unset=True)
            file_cfg.update(cli_cfg)
            cfg = VerlTrainConfig.model_validate(file_cfg, strict=True)

    # TODO(linjunrong.ocss884): this ENV is left for resolving SGLang conflict with ray devices
    # isolation, will solve in the future
    os.environ["ENSURE_CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not ray.is_initialized():
        kwargs = {}
        if cfg.ray.dashboard is not None:
            host, port = split_host_port(cfg.ray.dashboard, default_port=8265)
            kwargs["dashboard_host"] = host
            kwargs["dashboard_port"] = port
            kwargs["include_dashboard"] = True
        else:
            kwargs["include_dashboard"] = False

        ray.init(
            runtime_env={
                "env_vars": {
                    "TOKENIZERS_PARALLELISM": "true",
                    "NCCL_DEBUG": "WARN",
                    "VLLM_LOGGING_LEVEL": "WARN"
                },
            },
            num_cpus=cfg.ray.num_cpus,
            **kwargs
        )
    else:
        logger.warning("Ray is already initialized. Skipping ray.init(), ray configuration will be partially ignored.")

    from multimeditron.verl import TaskRunner
    runner = TaskRunner.remote()
    ray.get(runner.run.remote(cfg, trust_remote_code=trust_remote_code))

