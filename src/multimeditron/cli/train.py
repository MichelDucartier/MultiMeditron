from multimeditron.cli import EPILOG, main_cli
from .config import VerlConfig
from .utils import split_host_port
import yaml
import click
import os
import logging
import ray


logger = logging.getLogger(__name__)


@main_cli.command(epilog=EPILOG)
@click.option("--input", "-i", type=click.Path(exists=True), default=None, help="Path to load an existing configuration file to modify.")
@click.option("--output", "-o", type=click.Path(), default=None, help="Path to save the generated configuration file.")
def config_verl_generate(input, output):
    """
    Generate a default configuration file for training the MultiMeditron model.
    """
    if input is not None:
        with open(input, "r") as f:
            file_cfg = yaml.safe_load(f)
        cfg = VerlConfig.model_validate(file_cfg, strict=True)
    else:
        cfg = VerlConfig()

    if output is not None:
        with open(output, "w") as f:
            yaml.dump(cfg.model_dump(), f)
        print(f"Configuration file saved to {output}")
    else:
        print(yaml.dump(cfg.model_dump()))

@main_cli.command(epilog=EPILOG)
@click.option("--config", "-c", type=click.Path(exists=True), multiple=True, help="Path to the configuration file(s) in YAML format.")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging.")
@click.option("--debug/--no-debug", default=False, help="Enable debug mode.")
@click.option("--trust-remote-code/--no-trust-remote-code", default=False, help="Whether to trust remote code when loading models from HuggingFace.")
def train_verl(config: tuple[str] = [], trust_remote_code: bool = False, verbose: bool = False, debug: bool = False):
    """
    Train the MultiMeditron model using the specified configuration file.
    """
    # Load and merge configuration files from YAML
    cfg = VerlConfig()

    for cfg_path in config:
        logger.info(f"Loading configuration from {cfg_path}")
        with open(cfg_path, "r") as f:
            file_cfg = yaml.safe_load(f)

            # Merge configurations
            cli_cfg = cfg.model_dump(exclude_unset=True)
            cli_cfg.update(file_cfg)
            # file_cfg.update(cli_cfg)
            cfg = VerlConfig.model_validate(cli_cfg, strict=True)

    # TODO(linjunrong.ocss884): this ENV is left for resolving SGLang conflict with ray devices
    # isolation, will solve in the future
    # os.environ["ENSURE_CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not ray.is_initialized():
        kwargs = {
            "runtime_env": {
                "env_vars": {
                    "TOKENIZERS_PARALLELISM": "true",
                    "NCCL_DEBUG": "INFO" if debug else "WARN",
                    "VLLM_LOGGING_LEVEL": "INFO" if debug else "ERROR",
                },
            },
        }

        if cfg.ray.num_cpus is not None:
            kwargs["num_cpus"] = cfg.ray.num_cpus

        if debug:
            logger.info("Ray debug mode is enabled.")
            kwargs["runtime_env"]["env_vars"]["RAY_DEBUG"] = "1"
            kwargs["runtime_env"]["env_vars"]["RAY_DEBUG_POST_MORTEM"] = "1"
        else:
            logger.info("Ray debug mode is disabled.")
            kwargs["runtime_env"]["env_vars"]["RAY_DEBUG"] = "0"

        if cfg.ray.dashboard is not None:
            host, port = split_host_port(cfg.ray.dashboard, default_port=8265)
            kwargs["dashboard_host"] = host
            kwargs["dashboard_port"] = port
            kwargs["include_dashboard"] = True
        else:
            kwargs["include_dashboard"] = False


        ray.init(
            **kwargs
        )
    else:
        logger.warning("Ray is already initialized. Skipping ray.init(), ray configuration will be partially ignored.")

    from multimeditron.verl import TaskRunner
    runner = TaskRunner.remote()
    ray.get(runner.run.remote(cfg, trust_remote_code=trust_remote_code, verbose=verbose))
