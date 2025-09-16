import warnings
from multimeditron.model.prompt_tokenizers import NUM_EMBEDDINGS_KEY
from ..modality import AbstractModality, ModalityConfig
import torch
from transformers import AutoImageProcessor, CLIPModel, AutoModel
from multimeditron.dataset.registry.registry import ModalityRegistry
import numpy as np
from PIL import Image
import io
from typing import Optional, Dict, Any, List


class ImageConfig(ModalityConfig):
    model_type = "moe_meditron_clip"

    def __init__(
        self,
        modality_name: Optional[str] = None,
        max_batch_size: int = 32,
        use_bias_proj: bool = True,
        expert_clip_names: List[str] = [
            "openai/clip-vit-large-patch14", 
            "openai/clip-vit-large-patch14"
        ],
        top_k_experts: int = 1,
        **kwargs,
    ):
        super().__init__(
            modality_name=modality_name,
            max_batch_size=max_batch_size,
            use_bias_proj=use_bias_proj,
            modality_type="image",
            kwargs=kwargs,
        )

        self.expert_clip_names = expert_clip_names
        self.top_k_experts = top_k_experts


class MoEGatingNetwork(torch.nn.Module):
    def __init__(self, num_experts: int, top_k: int = 1):
        super().__init__()
        self.top_k = top_k
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = AutoImageProcessor.from_pretrained("openai/clip-vit-large-patch14")

        self.input_dim = self.clip_model.vision_embed_dim
        self.gate = torch.nn.Linear(self.input_dim, num_experts)


    def forward(self, x):
        x = self.clip_model.vision_model(x).last_hidden_state[:, 0, :]

        logits = self.gate(x)

        topk_vals, topk_indices = torch.topk(logits, self.top_k, dim=-1)

        weights = torch.nn.functional.softmax(logits, dim=-1)

        return topk_indices, weights



class ImageModality(AbstractModality):
    config_class = ImageConfig

    def __init__(self, config: ImageConfig):
        super().__init__(config)

        self.experts = torch.nn.ModuleList()

        self._embedding_size = None
        for idx, clip_name in enumerate(config.expert_clip_names):
            expert_model = CLIPModel.from_pretrained(clip_name, trust_remote_code=True)

            if self._embedding_size is None:
                self._embedding_size = expert_model.vision_embed_dim

            self.experts.append(expert_model.vision_model)

        self._num_patches_per_entry = (self.experts[0].config.image_size // self.experts[0].config.patch_size) ** 2
        
        self.gating_network = MoEGatingNetwork(len(self.experts), config.top_k_experts)
        self.image_processor = self.gating_network.processor

    def forward(self, inputs) -> torch.FloatTensor:
        device = next(self.experts[0].parameters()).device
        inputs = torch.stack(inputs, dim=0).to(device)

        topk_indices, weights = self.gating_network(inputs)
        
        if self.training:
            # Use all experts
            expert_outputs = []
            for expert in self.experts:
                expert_out = expert(inputs).last_hidden_state[:, 1:, :]

                expert_outputs.append(expert_out)

            # stacked_expert_outputs shape: (num_experts, batch_size, num_patches, embedding_size)
            stacked_expert_outputs = torch.stack(expert_outputs, dim=1)
            weights = weights.unsqueeze(-1).unsqueeze(-1)  # Shape: (batch_size, 1, 1, num_experts)

            weighted_output = (stacked_expert_outputs * weights).sum(dim=1)

            return weighted_output

           
    def modality_to_tensor(self, modality) -> Dict[str, Any]:
        processed_modality = modality.copy()

        image = modality["value"]
        if isinstance(modality["value"], dict):
            image = Image.open(io.BytesIO(image["bytes"]))

        pixel_values = self.image_processor(images=image, return_tensors="pt")["pixel_values"][0]
        processed_modality["value"] = pixel_values
        processed_modality[NUM_EMBEDDINGS_KEY] = self._num_patches_per_entry

        return processed_modality

    @property
    def embedding_size(self) -> int:
        return self._embedding_size

    @classmethod
    def from_dict(cls, config_args, **kwargs):
        return ImageConfig.from_dict(config_args, **kwargs)
