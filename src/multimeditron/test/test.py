sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from accelerate import Accelerator
from utils import pad_sequence
from train.trainer import MultimodalTrainer
from model.model import MultiModalModelForCausalLM, ModalityConfig
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Config
import torch.nn as nn
from utils import expand_attachments, pad_sequence
from model.model import MultiModalModelForCausalLM, ModalityConfig, MultimodalProcessedInput
from model.modality_imp.three_d_modality import CTModality
from model.modality_imp.image_modality import ImageModality
from model.modality import AbstractModality, ModalityProjectionOutput, _ModalityWithProjection
import unittest
import torch
import os
import sys


class TestImageModality(unittest.TestCase):
    def setUp(self):
        self.modality = ImageModality()
        self.device = torch.device("cpu")
        self.modality.load_model(device_map=self.device)

    def test_load_model(self):
        self.assertTrue(self.modality.is_loaded)
        self.assertIsNotNone(self.modality.vision_tower)
        self.assertIsNotNone(self.modality.image_processor)

    def test_preprocess(self):
        # Create a dummy image tensor
        from PIL import Image
        import numpy as np

        dummy_image = Image.fromarray(
            np.uint8(np.random.rand(224, 224, 3) * 255))
        processed = self.modality.preprocess(dummy_image)
        self.assertIn("pixel_values", processed)
        self.assertIsInstance(processed["pixel_values"], torch.Tensor)

    def test_forward(self):
        from PIL import Image
        import numpy as np

        dummy_image = Image.fromarray(
            np.uint8(np.random.rand(224, 224, 3) * 255))
        processed = self.modality.preprocess(dummy_image)
        features = self.modality.forward(processed)
        self.assertIsInstance(features, torch.Tensor)
        # Assuming output is [batch_size, feature_dim]
        self.assertEqual(features.dim(), 2)


class TestModalityWithProjection(unittest.TestCase):
    def setUp(self):
        # Create a dummy modality
        class DummyModality(AbstractModality):
            def __init__(self):
                super().__init__()
                self._name = "DummyModality"
                self._output_size = 128

            @property
            def name(self):
                return self._name

            @property
            def embedding_size(self):
                return self._output_size

            def preprocess(self, value):
                return value

            def forward(self, value):
                return torch.randn(value.size(0), self._output_size)

        self.dummy_modality = DummyModality()
        self.projection = _ModalityWithProjection(
            modality=self.dummy_modality,
            hidden_size=64,
            num_patches_per_entry=2,
            dtype=torch.float32,
            use_bias_proj=True
        )

    def test_projection_initialization(self):
        self.assertEqual(self.projection.hidden_size, 64)
        self.assertEqual(self.projection.num_patches_per_entry, 2)
        self.assertIsInstance(self.projection.mm_proj, nn.Linear)
        self.assertEqual(self.projection.mm_proj.in_features,
                         self.dummy_modality.embedding_size)
        self.assertEqual(self.projection.mm_proj.out_features,
                         self.projection.hidden_size * self.projection.num_patches_per_entry)

    def test_forward_projection(self):
        input_tensor = torch.randn(5, self.dummy_modality.embedding_size)
        output = self.projection(input_tensor)
        self.assertIsInstance(output, ModalityProjectionOutput)
        self.assertIsNotNone(output.projection)
        self.assertEqual(output.projection.shape, (5, 64, 2))


class TestLanguageModelForCausalLM(unittest.TestCase):
    def setUp(self):
        # Load a small, functional GPT-2 model for testing
        config = GPT2Config(
            vocab_size=50257,  # GPT-2 vocab size
            n_positions=128,
            n_embd=64,  # Smaller hidden size for testing
            n_layer=2,  # Reduce the number of layers for testing purposes
            n_head=2,  # Fewer attention heads
        )
        self.transformer = GPT2LMHeadModel(config)
        self.hidden_size = config.n_embd
        self.vocab_size = config.vocab_size
        self.dtype = torch.float32
        self.attachment_token_idx = 999  # Dummy token index for attachment
        self.pad_token_idx = 0  # Padding token index

        # Create a dummy image modality
        class DummyImageModality(AbstractModality):
            def __init__(self):
                super().__init__()
                self._name = "DummyImageModality"
                self._output_size = 128

            @property
            def name(self):
                return self._name

            @property
            def embedding_size(self):
                return self._output_size

            def preprocess(self, value):
                return value

            def forward(self, value):
                return torch.randn(value.size(0), self._output_size)

        self.image_modality = DummyImageModality()
        self.modality_config = ModalityConfig(
            modality=self.image_modality,
            num_patches_per_entry=2
        )
        self.model = MultiModalModelForCausalLM(
            model=self.transformer,
            hidden_size=self.hidden_size,
            vocab_size=self.vocab_size,
            dtype=self.dtype,
            modalities_with_config=[self.modality_config],
            attachment_token_idx=self.attachment_token_idx,
            pad_token_idx=self.pad_token_idx
        )

    def test_model_initialization(self):
        self.assertIsNotNone(self.model.lm_head)
        self.assertIsInstance(self.model.modalities, nn.ModuleList)
        self.assertEqual(len(self.model.modalities), 1)
        self.assertIn("DummyImageModality", self.model.modalities_by_type)

    def test_forward_pass_without_modalities(self):
        # Random input ids for GPT-2
        input_ids = torch.randint(0, self.vocab_size, (2, 10))
        outputs = self.model(input_ids=input_ids)
        self.assertIsNotNone(outputs)
        # logits shape should match
        self.assertEqual(outputs[0].shape, (2, 10, self.vocab_size))

    def test_forward_pass_with_modalities(self):
        input_ids = torch.tensor([
            [1, 2, self.attachment_token_idx, 4, 5],
            [1, self.attachment_token_idx, 3, 4, self.pad_token_idx]
        ])
        multimodal_inputs = [
            [MultimodalProcessedInput(
                type="DummyImageModality", value=torch.randn(1, 128))],
            [MultimodalProcessedInput(
                type="DummyImageModality", value=torch.randn(1, 128))]
        ]
        outputs = self.model(input_ids=input_ids,
                             multimodal_inputs=multimodal_inputs)
        self.assertIsNotNone(outputs)
        # logits adjusted for attachments
        self.assertEqual(outputs[0].shape,
                         (2, input_ids.shape[1], self.vocab_size))


class TestUtils(unittest.TestCase):
    def test_expand_attachments(self):
        attachment_points = [1, 4]
        embeds_ids = torch.tensor([1, 3, 2, 4, 5])
        attachment_embeds = [torch.tensor([10, 11]), torch.tensor([20, 21])]
        additional_inputs = [
            (False, torch.tensor([0, 0, 0, 0, 0])),
            (torch.tensor(-1), torch.tensor([0, 0, 0, 0, 0]))
        ]

        new_embeds_ids, new_additional_tensors = expand_attachments(
            attachment_points,
            embeds_ids,
            attachment_embeds,
            additional_inputs
        )

        expected_embeds_ids = torch.tensor([1, 10, 11, 2, 4, 20, 21])
        self.assertTrue(torch.equal(new_embeds_ids, expected_embeds_ids))

        expected_additional_0 = torch.tensor([0, 0, 0, 0, 0, 0, 0])
        expected_additional_1 = torch.tensor([0,  -1, -1, 0, 0, -1, -1])

        self.assertTrue(torch.equal(
            new_additional_tensors[0], expected_additional_0))
        self.assertTrue(torch.equal(
            new_additional_tensors[1], expected_additional_1))

    def test_pad_sequence(self):
        sequences = [
            torch.tensor([1, 2, 3]),
            torch.tensor([4, 5]),
            torch.tensor([6])
        ]
        padded = pad_sequence(sequences, padding_value=0, padding_side="right")
        expected = torch.tensor([
            [1, 2, 3],
            [4, 5, 0],
            [6, 0, 0]
        ])
        self.assertTrue(torch.equal(padded, expected))

        # Test left padding
        padded_left = pad_sequence(
            sequences, padding_value=0, padding_side="left")
        expected_left = torch.tensor([
            [1, 2, 3],
            [0, 4, 5],
            [0, 0, 6]
        ])
        self.assertTrue(torch.equal(padded_left, expected_left))


class TestThreeDModality(unittest.TestCase):
    def setUp(self):
        self.three_d_modality = CTModality()
        self.three_d_modality.load_model()

    def test_preprocess(self):
        # Assuming you have a .npy file for testing
        image_path = "src/test/data/example_00.npy"
        image_tensor = self.three_d_modality.preprocess(image=image_path)
        self.assertEqual(image_tensor.shape, (1, 1, 32, 256, 256))

    def test_forward(self):
        image_path = "src/test/data/example_00.npy"
        image_tensor = self.three_d_modality.preprocess(image_path)
        features = self.three_d_modality.forward(image_tensor)
        self.assertEqual(features.shape[-1],
                         self.three_d_modality.embedding_size)


if __name__ == '__main__':
    unittest.main()
