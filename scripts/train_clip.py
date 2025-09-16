#!/usr/bin/env python
# coding=utf-8
# Copyright 2022 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Training a CLIP like dual encoder models using text and vision encoders in the library.

The script can be used to train CLIP like models for languages other than English by using
a text encoder pre-trained in the desired language. Currently this script supports the following vision
and text models:
Vision models: ViT(https://huggingface.co/models?filter=vit), CLIP (https://huggingface.co/models?filter=clip)
Text models: BERT, ROBERTa (https://huggingface.co/models?filter=fill-mask)
"""

import logging
import os
import numpy as np
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from datasets import load_dataset, DatasetDict, interleave_datasets
from PIL import Image
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import CenterCrop, ConvertImageDtype, Normalize, Resize
from torchvision.transforms.functional import InterpolationMode

import transformers
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    VisionTextDualEncoderModel,
    Trainer,
    TrainingArguments,
    set_seed,
)

from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

#Disable WANDB
os.environ["WANDB_DISABLED"] = "true"

logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
#check_min_version("4.47.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/contrastive-image-text/requirements.txt")


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    image_processor_name: str = field(default=None, metadata={"help": "Name or path of preprocessor config."})
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    freeze_vision_model: bool = field(
        default=False, metadata={"help": "Whether to freeze the vision model parameters or not."}
    )
    freeze_text_model: bool = field(
        default=False, metadata={"help": "Whether to freeze the text model parameters or not."}
    )
    
    vision_model_name: Optional[str] = field(
        default=None,
        metadata={"help": "Vision encoder model name/path (e.g. openai/clip-vit-base-patch32)"}
    )
    text_model_name: Optional[str] = field(
        default=None, 
        metadata={"help": "Text encoder model name/path (e.g. FacebookAI/roberta-base)"}
    )

@dataclass
class DatasetConfig:
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    data_dir: Optional[str] = field(default=None, metadata={"help": "The data directory containing input files."})
    image_column: Optional[str] = field(
        default="modalities",
        metadata={"help": "The name of the column in the datasets containing the full image file paths."},
    )
    caption_column: Optional[str] = field(
        default="text",
        metadata={"help": "The name of the column in the datasets containing the image captions."},
    )
    weight: Optional[float] = field(
        default=1.0, metadata={"help": "The weight to assign to this dataset during training."}
        )
    

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    
    dataset_configs: Optional[List[DatasetConfig]] = field(
        default=None, metadata={"help": "Dataset configuration for training and evaluation."}
    )

    max_seq_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    
dataset_name_mapping = {
    "image_caption_dataset.py": ("image_path", "caption"),
}

# We use torchvision for faster image pre-processing. The transforms are implemented as nn.Module,
# so we jit it to be faster.
class Transform(torch.nn.Module):
    def __init__(self, image_size, mean, std):
        super().__init__()
        self.transforms = torch.nn.Sequential(
            Resize([image_size], interpolation=InterpolationMode.BICUBIC, antialias=True),
            CenterCrop(image_size),
            ConvertImageDtype(torch.float),
            Normalize(mean, std),
        )

    def forward(self, x) -> torch.Tensor:
        """`x` should be an instance of `PIL.Image.Image`"""
        with torch.no_grad():
            x = self.transforms(x)
        return x


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    input_ids = torch.tensor([example["input_ids"] for example in examples], dtype=torch.long)
    attention_mask = torch.tensor([example["attention_mask"] for example in examples], dtype=torch.long)
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "return_loss": True,
    }

def get_combined_dataset(dataset_configs: List[DatasetConfig], model_args: ModelArguments):
    datasets = []
    probabilities = []
    logger.info(f"Loading datasets: {dataset_configs}")
    for config in dataset_configs:
        # Load individual dataset
        if config.get("dataset_name", None) is None:
            assert(len(config) == 1)
            config = config[list(config.keys())[0]]
        config = DatasetConfig(**config)
        if config.dataset_name.endswith(".jsonl"):
            dataset = load_dataset(
                "json",
                config.dataset_config_name,
                cache_dir=model_args.cache_dir,
                keep_in_memory=False,
                data_dir=config.data_dir,
                token=model_args.token,
                trust_remote_code=model_args.trust_remote_code,
                data_files=config.dataset_name,
                )
        else:
            dataset = load_dataset(
                config.dataset_name,
                config.dataset_config_name,
                cache_dir=model_args.cache_dir,
                keep_in_memory=False,
                data_dir=config.data_dir,
                token=model_args.token,
                trust_remote_code=model_args.trust_remote_code,
            )
        
        # For each dataset, assign the image column and caption column to standard names
        if "train" in dataset:
            def find_img_path(row):
                return {config.caption_column: row[config.caption_column], config.image_column: os.path.join(os.path.dirname(config.dataset_name), row[config.image_column][0]["value"])}

            if config.dataset_name.endswith(".jsonl"):
                dataset["train"] = dataset["train"].map(find_img_path)

            dataset["train"] = dataset["train"].rename_column(config.image_column, "image_path").rename_column(config.caption_column, "caption")
            dataset["train"] = dataset["train"].map(lambda x: {"caption": x["caption"].replace("<attachment>","")})
        
        # Repeat dataset according to epochs weight
        probabilities.append(config.weight)
        datasets.append(dataset["train"])
    
    # Normalize weights
    probabilities = np.array(probabilities)
    probabilities = probabilities / np.sum(probabilities)
    
    # Combine all datasets
    combined_dataset = interleave_datasets(datasets, probabilities=probabilities)

    return combined_dataset.train_test_split(test_size=0.1) #DatasetDict({"train": combined_dataset})

def main():
    # 1. Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    """
    if len(sys.argv) == 2: 
        if sys.argv[1].endswith(".json"):
            # If we pass only one argument to the script and it's the path to a json file,
            # let's parse it to get our arguments.
            model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
        elif sys.argv[1].endswith(".yaml"):
            # If we pass only one argument to the script and it's the path to a yaml file,
            # let's parse it to get our arguments.
            model_args, data_args, training_args = parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[1]))
            training_args.bf16 = True
    
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    """
    model_args, data_args, training_args = parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[1]))

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clip", model_args, data_args)

    # 2. Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    
    
    logger.info(f"Training/evaluation parameters {training_args}")

    # 3. Detecting last checkpoint and eventually continue from last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # 4. Load dataset
    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files this script will use the first column for the full image path and the second column for the
    # captions (unless you specify column names for this with the `image_column` and `caption_column` arguments).
    #
    if data_args.dataset_configs is not None:
        # Downloading and loading a dataset from the hub.
        dataset = get_combined_dataset(data_args.dataset_configs, model_args)
    else:
        raise ValueError("Please provide dataset configs")
        
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.

    # 5. Load pretrained model, tokenizer, and image processor
    if model_args.vision_model_name and model_args.text_model_name:
        # Dual encoder path
        logger.info(f"Loading dual encoder with vision model {model_args.vision_model_name} "
                f"and text model {model_args.text_model_name}")
        
        model = VisionTextDualEncoderModel.from_vision_text_pretrained(
            model_args.vision_model_name,
            model_args.text_model_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        ).to(dtype=torch.bfloat16)
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.text_model_name,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            token=model_args.token,
        )
        
        image_processor = AutoImageProcessor.from_pretrained(
            model_args.vision_model_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )
        
    else:
        # Single model path - existing code
        if model_args.tokenizer_name:
            tokenizer = AutoTokenizer.from_pretrained(
                model_args.tokenizer_name,
                cache_dir=model_args.cache_dir,
                use_fast=model_args.use_fast_tokenizer,
                token=model_args.token,
            )
        elif model_args.model_name_or_path:
            tokenizer = AutoTokenizer.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=model_args.cache_dir,
                use_fast=model_args.use_fast_tokenizer,
                token=model_args.token,
            )
        else:
            raise ValueError(
                "You must specify either model_name_or_path or both vision_model_name and text_model_name"
            )

        image_processor = AutoImageProcessor.from_pretrained(
            model_args.image_processor_name or model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
        )

        model = AutoModel.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
        )
    config = model.config

    def _freeze_params(module):
        for param in module.parameters():
            param.requires_grad = False

    if model_args.freeze_vision_model:
        _freeze_params(model.vision_model)

    if model_args.freeze_text_model:
        _freeze_params(model.text_model)

    # set seed for torch dataloaders
    set_seed(training_args.seed)

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = dataset["train"].column_names
    elif training_args.do_eval:
        column_names = dataset["train"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # 6. Get the column names for input/target.
    
    image_column = "image_path"
    caption_column = "caption"

    # 7. Preprocessing the datasets.
    # Initialize torchvision transforms and jit it for faster processing.
    image_transformations = Transform(
        config.vision_config.image_size, image_processor.image_mean, image_processor.image_std
    )
    image_transformations = torch.jit.script(image_transformations)
    

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples):
        captions = list(examples[caption_column])
        text_inputs = tokenizer(captions, max_length=data_args.max_seq_length, padding="max_length", truncation=True)
        examples["input_ids"] = text_inputs.input_ids
        examples["attention_mask"] = text_inputs.attention_mask
        return examples

    def transform_images(examples):
        images = []
        for image_file in examples[image_column]:
            if isinstance(image_file, str):
                # If it's a file path
                image = read_image(image_file, mode=ImageReadMode.RGB)
            else:
                # If it's already a PIL Image
                image = torch.from_numpy(np.array(image_file)).permute(2, 0, 1)
        
            images.append(image)
        
        examples["pixel_values"] = [image_transformations(image) for image in images]
        return examples

    def filter_corrupt_images(examples):
        """remove problematic images"""
        valid_images = []
        for image_file in examples[image_column]:
            if isinstance(image_file, str):
                try:
                    Image.open(image_file)
                    valid_images.append(True)
                except Exception as e:
                    logger.warning(f"Corrupt image found: {image_file}, Error: {str(e)}")
                    valid_images.append(False)
            else: # already loaded image
                valid_images.append(True)
        return valid_images

    if training_args.do_train:
        if "train" not in dataset:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = dataset["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

        logger.info(f"Dataset length: {len(train_dataset)}")
        train_dataset = train_dataset.filter(
            filter_corrupt_images, batched=True, num_proc=data_args.preprocessing_num_workers
        )
        logger.info(f"Dataset length: {len(train_dataset)}")

        train_dataset = train_dataset.map(
            function=tokenize_captions,
            batched=True,
            remove_columns=[col for col in column_names if col != image_column],
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )

        # Transform images on the fly as doing it on the whole dataset takes too much time.
        train_dataset.set_transform(transform_images)

    if training_args.do_eval:
        test_dataset = dataset["test"]
        test_dataset = test_dataset.map(
            function=tokenize_captions,
            batched=True,
            remove_columns=[col for col in column_names if col != image_column],
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on test dataset",
        )

        # Transform images on the fly as doing it on the whole dataset takes too much time.
        test_dataset.set_transform(transform_images)

    # 8. Initialize our trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=test_dataset if training_args.do_eval else None,
        data_collator=collate_fn,
    )

    # 9. Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        
        try:
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
        except RuntimeError as e:
            print(e)
            input()
            train_result = None
        
        trainer.save_model()
        tokenizer.save_pretrained(training_args.output_dir)
        image_processor.save_pretrained(training_args.output_dir)
        if train_result:
            trainer.log_metrics("train", train_result.metrics)
            trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # 10. Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # 11. Write Training Stats and push to hub.
    finetuned_from = model_args.model_name_or_path
    # If from a local directory, don't set `finetuned_from` as this is required to be a valid repo. id on the Hub.

    if finetuned_from is None or os.path.isdir(finetuned_from):
        finetuned_from = None
    kwargs = {"finetuned_from": finetuned_from, "tasks": "contrastive-image-text-modeling"}
    for dataset in data_args.dataset_configs:
        if dataset.get("dataset_name", None) is None:
            assert(len(dataset) == 1)
            dataset = dataset[list(dataset.keys())[0]]
        dataset = DatasetConfig(**dataset)
        if dataset.dataset_name is not None:
            if not hasattr(kwargs, "dataset_tags"):
                kwargs["dataset_tags"] = []
            kwargs["dataset_tags"].append(dataset.dataset_name)

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

if __name__ == "__main__":
    main()