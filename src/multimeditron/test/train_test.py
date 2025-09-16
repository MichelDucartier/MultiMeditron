sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from multimeditron.model.data_loader import DataCollatorForMultimodal, MultimodalDataset
from train.trainer import TrainingMode
import numpy as np
from transformers import TrainingArguments
from train.trainer import MultimodalTrainer
from accelerate import Accelerator
from torch.utils.data import DataLoader
from model.modality_imp.three_d_modality import CTModality, CTConfig
from model.model import MultiModalModelForCausalLM, ModalityConfig, MultimodalProcessedInput, MultimodalConfig
from torch.nn import GELU
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
import unittest
import torch
import os
import sys

# Custom Projection Layer for 3D Modality


class SimpleProjectionLayer(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.gelu = GELU()
        self.fc = torch.nn.Linear(input_size, hidden_size)

    def forward(self, x):
        return self.fc(self.gelu(x))


class TestMultimodalProjectionTrainer(unittest.TestCase):
    def setUp(self):
        # Set up the accelerator
        self.accelerator = Accelerator()

        # Load GPT-2 model
        config = GPT2Config(
            vocab_size=50258,
            n_positions=128,
            n_embd=64,  # Smaller hidden size for testing
            n_layer=2,  # Reduced number of layers
            n_head=2  # Fewer attention heads
        )
        self.transformer = GPT2LMHeadModel(config)
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        # Register the attachment token <image> with the tokenizer
        self.attachment_token = "<image>"
        if self.attachment_token not in self.tokenizer.get_vocab():
            self.tokenizer.add_tokens([self.attachment_token])
        self.attachment_token_idx = self.tokenizer.convert_tokens_to_ids(
            self.attachment_token)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # Set up the model (LanguageModelForCausalLM)
        self.hidden_size = config.n_embd
        self.vocab_size = config.vocab_size

        # Add 3D Modality with a simple projection layer
        modality_config = CTConfig(
            modality_name="3DModality",
            num_patches_per_entry=16,
            clip_name="GoodBaiBai88/M3D-CLIP",
        )

        multimodal_config = MultimodalConfig(
            hidden_size=self.hidden_size,
            modalities=[modality_config],
            attachment_token_idx=self.attachment_token_idx,
            pad_token_idx=0,
            vocab_size=self.vocab_size,
        )

        self.model = MultiModalModelForCausalLM(
            multimodal_config,
            self.transformer,
            dtype=torch.bfloat16
        )

        # Sample data
        self.data = [{
            "text": "This <image> is a CT scan",
            "image_3d": "src/test/data/example_00.npy"
        }]

        # Define processors for each modality
        self.modality_processors = self.model.processors()
        self.modality_processors["text"] = self.tokenizer

        self.train_dataset = MultimodalDataset(
            data=self.data,
            tokenizer=self.tokenizer,
            modality_processors=self.modality_processors
        )
        self.collator = DataCollatorForMultimodal(
            tokenizer=self.tokenizer,
            modality_processors=self.modality_processors
        )

    def _create_single_sample_dataset(self):
        # Preprocess the 3D image
        image_path = self.data[0]["image_3d"]
        image_tensor = np.load(image_path)
        image_tensor = torch.tensor(
            image_tensor, dtype=torch.float32).unsqueeze(0)

        sample = {
            "text": self.data[0]["text"],
            "image_3d": image_tensor
        }

        dataset = [sample]*1000
        return dataset

    def test_train_projection_layer(self):
        # Create the trainer
        trainer = MultimodalTrainer(
            model=self.model,
            train_dataset=self.train_dataset,
            data_collator=self.collator,
            tokenizer=self.tokenizer,
            args=TrainingArguments(
                output_dir="tmp_trainer/runs", remove_unused_columns=False, learning_rate=1e-2),
            training_mode=TrainingMode.ALIGNMENT
        )

        # Run a single training step
        print(next(iter(self.model.modalities[0].projection.parameters())))
        trainer.train()
        print(next(iter(self.model.modalities[0].projection.parameters())))

        # Validate that the model is trainable
        for name, param in self.model.named_parameters():
            if "projection_module" in name:
                self.assertTrue(param.requires_grad,
                                f"Parameter {name} should require gradients")


if __name__ == '__main__':
    unittest.main()
