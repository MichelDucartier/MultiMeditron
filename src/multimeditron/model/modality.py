from __future__ import annotations

from transformers import PretrainedConfig, PreTrainedModel
from typing import Any, Optional, OrderedDict, Dict
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from transformers import AutoModel, AutoTokenizer, PretrainedConfig, PreTrainedTokenizer, PreTrainedModel

from multimeditron.dataset.registry.registry import ModalityRegistry


class ModalityConfig(PretrainedConfig):
    """
    Configuration class for defining modality-specific settings and parameters. Inherits from
    `transformers.PretrainedConfig`.

    Attributes:
        modality_name (Optional[str]): The name of the modality (e.g., 'ClipImage', 'ClipAudio'). This allows specifying a unique identifier for the modality.
        max_batch_size (int): The maximum batch size for processing (default is 32). This helps in controlling memory usage during training or inference.
        use_bias_proj (bool): Whether to use bias in the projection layer (default is True). If set to True, the model includes bias terms in the projection layers.
        modality_type (Optional[str]): The type of modality (e.g., 'image', 'audio'). This specifies the general category of the input data being processed, helping downstream tasks apply modality-specific logic.
    """
    def __init__(
            self,
            modality_name: Optional[str] = None,
            max_batch_size: int = 32,
            use_bias_proj: bool = True,
            modality_type: Optional[str] = None,
            **kwargs
            ):
        self.modality_type = modality_type  # e.g., 'image', 'audio'
        self.modality_name = modality_name  # e.g., 'ClipImage', 'ClipAudio'

        self.max_batch_size = max_batch_size
        self.use_bias_proj = use_bias_proj

        super().__init__(**kwargs)


class AbstractModality(ABC, PreTrainedModel):
    """
    Abstract base class for all modalities, providing a structure for defining new modality-specific models.

    Attributes:
        config (ModalityConfig): Configuration object containing modality-specific parameters.
        config_class (Type[ModalityConfig]): Class of the configuration object.
        tokenizer (Optional[PreTrainedTokenizer]): Tokenizer associated with the modality, if applicable.
    """
    def __init__(self, config: ModalityConfig):
        super().__init__(config)

        self.config = config
        self.config_class = ModalityConfig
        self.tokenizer = None

    @property
    @abstractmethod
    def embedding_size(self) -> int:
        """
        Abstract property that must be implemented to return the embedding size of the modality.

        Returns:
            int: The size of the embedding vector.
        """
        ...

    @property
    def num_patches_per_entry(self) -> Optional[int]:
        """
        Property that returns the number of patches per entry, if applicable.

        Returns:
            Optional[int]: Number of patches per entry, or None if not applicable.
        """
        return None

    @abstractmethod
    def modality_to_tensor(self, modality: Dict[str, Any]) -> Dict[str, Any]:
        """
        Abstract method to convert a modality into a tensor representation.

        Args:
            modality (Dict[str, Any]): Input modality data.

        Returns:
            Dict[str, Any]: Tensor representation of the modality.
        """
        ...

    def set_requires_grad(self, value: bool):
        """
        Set the `requires_grad` attribute for all parameters in the model.

        Args:
            value (bool): Whether gradients should be computed for the parameters.
        """
        for params in self.parameters():
            params.requires_grad = value


class ModalityWithProjection(nn.Module):
    """
    A class that combines a modality with a projection layer to transform the modality's embedding into a hidden size.

    Attributes:
        config (ModalityConfig): Configuration object providing modality-specific settings.
        modality (AbstractModality): The modality to be processed.
        hidden_size (int): The hidden size of the projection layer.
        _dtype (torch.dtype): Data type for the projection layers.
        num_patches_per_entry (Optional[int]): Number of patches per entry, if applicable.
    """
    def __init__(self, modality: AbstractModality, hidden_size: int, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.config = modality.config
        self.modality = modality
        self.hidden_size = hidden_size
        self._dtype = dtype

        self.num_patches_per_entry = modality.num_patches_per_entry

        # Define projection layer
        # Neural network for projection
        self.projection = nn.Sequential(
                nn.Linear(modality.embedding_size, modality.embedding_size, dtype=dtype),
                nn.GELU(),
                nn.Linear(modality.embedding_size, self.hidden_size, dtype=dtype),
                nn.GELU(),
                nn.Linear(self.hidden_size, self.hidden_size, dtype=dtype),
        )

    def get_config(self) -> ModalityConfig:
        """
        Retrieve the configuration object associated with the modality.

        Returns:
            ModalityConfig: The configuration object.
        """
        return self.modality.config

    @property
    def name(self) -> str:
        """
        Name of the modality.

        Returns:
            str: The name of the modality.
        """
        return self.modality.modality_name

    def freeze_projection_only(self):
        """
        Freeze the parameters of the projection layers, while keeping the modality trainable.
        """
        for params in self.projection.parameters():
            params.requires_grad = False
        self.modality.set_requires_grad(True)

    def freeze_modality_only(self):
        """
        Freeze the parameters of the modality, while keeping the projection layers trainable.
        """
        for params in self.projection.parameters():
            params.requires_grad = True
        self.modality.set_requires_grad(False)

    def freeze_all(self):
        """
        Freeze all parameters in the model.
        """
        for params in self.parameters():
            params.requires_grad = False

    def unfreeze_all(self):
        """
        Unfreeze all parameters in the model.
        """
        for params in self.parameters():
            params.requires_grad = True

    def modality_to_tensor(self, modality: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert the input modality data into a tensor representation.

        Args:
            modality (Dict[str, Any]): Input modality data.

        Returns:
            Dict[str, Any]: Tensor representation of the modality.
        """
        return self.modality.modality_to_tensor(modality)

    def forward(self, value: Any) -> torch.FloatTensor:
        """
        Forward pass of the model, including modality processing and projection.

        Args:
            value (Any): Input data to be processed by the modality.

        Returns:
            torch.FloatTensor: Projected tensor representation.
        """
        hidden_state = self.modality(value).to(self._dtype)

        # Projection
        projection = self.projection(hidden_state)

        return projection
