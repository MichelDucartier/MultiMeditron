from multimeditron.cli import EPILOG, main_cli
import click
import logging
import os


logger = logging.getLogger(__name__)


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
    import torch
    import torch.multiprocessing as mp

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

        # processor_wrapper = ModalityRetriever(registry)

        ds = ds.map(
            make_map_fn("train", os.path.basename(dataset)),
            batched=False,
            writer_batch_size=num_processes,
            num_proc=num_processes,
            with_indices=True
        )

        # ds = ds.map(
        #     processor_wrapper.merge_modality_with_sample,
        #     batched=False,
        #     writer_batch_size=num_processes,
        #     num_proc=num_processes)

        ds.to_parquet(
            output
        )

def make_map_fn(split, data_source):
    def process_fn(i_data, idx):
        o_data = {
            "data_source": data_source,
            "prompt": [
                {
                    "role": "assistant",
                    "content": i_data["prompt"],
                }
            ],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": i_data["response"],
            },
            "extra_info": {
                "split": split,
                "index": idx,
            }
        }
        return o_data
    return process_fn
        
