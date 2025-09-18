from .config import VerlConfig
from .utils import split_host_port

import click
import torch
import torch.multiprocessing as mp
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
    cfg = VerlConfig()
    if output is not None:
        with open(output, "w") as f:
            yaml.dump(cfg.model_dump(), f)
        print(f"Configuration file saved to {output}")
    else:
        print(yaml.dump(cfg.model_dump()))

@main_cli.command(epilog=EPILOG)
@click.argument("dataset", type=click.Path(exists=True))
@click.argument("output", type=click.Path())
@click.argument("model", type=str)
@click.option("--attachment-token", "-a", type=str, default="<|reserved_special_token_0|>", help="Special token to represent attachments in the text.")
@click.option("--registry-type", "-r", type=click.Choice(["path", "hdf5", "wids"]), default="path", help="Type of the dataset registry to use.")
@click.option("--num-processes", "-n", type=int, default=32, help="Number of processes to use for tokenization.")
def preprocess_ds(dataset,
                  output,
                  model, 
                  attachment_token,
                  registry_type, 
                  num_processes):
    """
    Preprocess and tokenize the dataset for training.
    """
    torch.set_num_threads(1)
    mp.set_sharing_strategy("file_system")
    mp.set_start_method("spawn", force=True)

    # Disable randomness
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Create the base mode with fast tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model, dtype=torch.bfloat16, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    special_tokens = {'additional_special_tokens': [attachment_token]}
    tokenizer.add_special_tokens(special_tokens)

    # Create a model
    torch.set_default_dtype(torch.bfloat16)

    # Load the configuration
    print("Saving dataset to", output)
    from multimeditron.dataset.registry.registry import get_registry
    from multimeditron.model.jsonl_generator import JSONLGenerator
    from multimeditron.dataset.preprocessor.modality_preprocessor import ModalityRetriever
    from datasets import Dataset

    registry_builder = get_registry(registry_type)

    base_path = os.path.dirname(dataset)
    with registry_builder(base_path=base_path) as registry:
        ds = JSONLGenerator(dataset)
        ds = Dataset.from_generator(lambda: ds)

        processor_wrapper = ModalityRetriever(registry)

        ds = ds.map(
            processor_wrapper.merge_modality_with_sample,
            batched=False,
            writer_batch_size=num_processes,
            num_proc=num_processes)
        ds.to_parquet(
            output
        )

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
    os.environ["ENSURE_CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
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

