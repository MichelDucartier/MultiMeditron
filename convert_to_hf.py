from src.model.model import MultiModalModelForCausalLM, MultimodalConfig
from src.model.modality_imp.image_modality import ImageConfig 
from src.model.modality_imp.three_d_modality import CTConfig 

import yaml
import os
import torch
import argparse
import safetensors.torch as st
from transformers import AutoTokenizer

def bootstrap(config):
    multimodal_config = MultimodalConfig(
            hidden_size=config["token_size"],
            vocab_size=len(tokenizer),
            attachment_token_idx=attachment_token_idx,
            eos_token_idx=tokenizer.convert_tokens_to_ids(tokenizer.eos_token),
            modalities=[
                ImageConfig(modality_name="image", clip_name=config["modalities"]["image"]["clip_name"]),
                CTConfig(modality_name="image_3d")
                ],
            llm_path=config["base_llm"],
            modality_processing_mode=config["modality_processing_mode"]
            )

    model = MultiModalModelForCausalLM(multimodal_config, dtype=torch.bfloat16, bootstrap=True)
    return model


parser = argparse.ArgumentParser(description="Utility to uncompile a MultiMeditron model. As the MultiMeditron trainer saves compiled model (using torch.compile), loading the model can be cumbersome. This script converts the compiled version of MultiMeditron to a format usable by huggingface `from_pretrained`")

parser.add_argument("config_path")
parser.add_argument("checkpoint", help="Path to the compiled checkpoint")
parser.add_argument("uncompiled", help="Path to the uncompiled output directory")
args = parser.parse_args()

with open(args.config_path, "r") as f:
    config = yaml.safe_load(f)

# Create the base model
ATTACHMENT_TOKEN = config["attachment_token"]
tokenizer = AutoTokenizer.from_pretrained(config["base_llm"], dtype=torch.bfloat16)
tokenizer.pad_token = tokenizer.eos_token
special_tokens = {'additional_special_tokens': [ATTACHMENT_TOKEN]}
tokenizer.add_special_tokens(special_tokens)

attachment_token_idx = tokenizer.convert_tokens_to_ids(ATTACHMENT_TOKEN)

model_name = args.checkpoint

model = bootstrap(config)
model = torch.compile(model)
st.load_model(model, os.path.join(
    model_name, "model.safetensors"), strict=True)


print(model)

model.save_pretrained(args.uncompiled, 
                      safe_serialization=True,
                      torch_dtype=torch.bfloat16)
