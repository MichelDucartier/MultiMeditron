import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from multimeditron.model.prompt_tokenizers import Llama3PromptTokenizer
import unittest
import torch
from transformers import AutoTokenizer


class TestLlama3PromptTokenizer(unittest.TestCase):

    def setUp(self):
        # Initialize a simple GPT2 tokenizer for testing
        self.tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Meta-Llama-3.1-8B-Instruct")
        attachment_token = "<|reserved_special_token_1|>"

        # Define a mock modalities_num_embeddings mapping
        self.modalities_num_embeddings = {
            "image": 5,
            "image_3d": 10
        }

        # Instantiate the Llama3PromptTokenizer class
        self.prompt_tokenizer = Llama3PromptTokenizer(
            tokenizer=self.tokenizer,
            modalities_num_embeddings=self.modalities_num_embeddings,
            attachment_token=attachment_token
        )

        # Define default test inputs
        self.prompt = [
            {"role": "user", "content": "Hello, I am sending an image <|reserved_special_token_1|>"},
            {"role": "system", "content": "System message here."},
            {"role": "assistant", "content": "Got the image! Here is the analysis: blabla"},
            {"role": "user", "content": "Sending another <|reserved_special_token_1|>"}
        ]
        self.modalities = [
            {"type": "image", "path": "path/to/image"},
            {"type": "image_3d", "path": "path/to/image_3d"}
        ]

    def test_tokenize_conversation_basic(self):
        """
        Test that the conversation is correctly tokenized and the attachment tokens 
        are replaced by the appropriate modality IDs.
        """
        result = self.prompt_tokenizer.tokenize_conversation(
            self.prompt, self.modalities)

        # Check that the result contains the expected keys
        self.assertIn("input_ids", result)
        self.assertIn("attention_mask", result)
        self.assertIn("labels", result)

        input_ids = result["input_ids"]
        attention_mask = result["attention_mask"]
        labels = result["labels"]

        # Ensure input_ids, attention_mask, and labels have the same size
        self.assertEqual(input_ids.size(), attention_mask.size())
        self.assertEqual(input_ids.size(), labels.size())

        # Ensure that the correct modality tokens are in place for attachments
        # e.g., input_ids should contain 1000 for the first <|reserved_special_token_1|> and 2000 for the second
        attachment_positions = (input_ids == self.prompt_tokenizer.tokenizer.convert_tokens_to_ids(
            "<|reserved_special_token_1|>")).nonzero(as_tuple=True)[0]

        self.assert_attachment_positions(
            attachment_positions, modalities=["image", "image_3d"])

        # Ensure that the attention_mask correctly handles the attachments (they should not be masked)
        self.assertTrue(
            all(attention_mask[attachment_positions].eq(1).tolist()))

    def assert_attachment_positions(self, attachment_positions, modalities):
        attachment_groups = []
        current_group = 0
        last_pos = None
        for pos in attachment_positions:
            if last_pos is not None and last_pos != pos - 1:
                attachment_groups.append(current_group)
                current_group = 0

            current_group += 1
            last_pos = pos

        if current_group != 0:
            attachment_groups.append(current_group)

        self.assertEqual(len(attachment_groups), len(modalities))
        self.assertEqual(
            attachment_groups[0], self.modalities_num_embeddings[modalities[0]])
        self.assertEqual(
            attachment_groups[1], self.modalities_num_embeddings[modalities[1]])

    def test_labels_for_roles(self):
        """
        Test that the labels are set to ignore_index for user and system messages,
        and match the input_ids for assistant messages.
        """
        result = self.prompt_tokenizer.tokenize_conversation(
            self.prompt, self.modalities)

        input_ids = result["input_ids"]
        labels = result["labels"]

        # Tokenize the prompt to separate out user/system/assistant portions
        user_message = self.tokenizer("Hello, I am sending an image", return_tensors="pt")[
            "input_ids"].squeeze()
        system_message = self.tokenizer("System message here.", return_tensors="pt")[
            "input_ids"].squeeze()
        assistant_message = self.tokenizer(
            "Got the image! Here is the analysis:", return_tensors="pt")["input_ids"].squeeze()

        # Find positions corresponding to user, system, and assistant messages in input_ids
        user_length = len(user_message)
        system_length = len(system_message)
        assistant_length = len(assistant_message)

        # User and system message tokens should have labels equal to ignore_index (-100)
        self.assertTrue(all(labels[:user_length].eq(
            self.prompt_tokenizer.ignore_index).tolist()))
        self.assertTrue(all(labels[user_length:user_length + system_length].eq(
            self.prompt_tokenizer.ignore_index).tolist()))

    def test_handle_multiple_attachments(self):
        """
        Test that the tokenizer can handle multiple attachments in the conversation correctly.
        """
        multiple_prompt = [
            {"role": "user", "content": "Here's an image <|reserved_special_token_1|> and another one <|reserved_special_token_1|>"},
            {"role": "assistant", "content": "Got both images!"}
        ]
        multiple_modalities = [
            {"type": "image", "path": "path/to/image1"},
            {"type": "image", "path": "path/to/image2"}
        ]

        result = self.prompt_tokenizer.tokenize_conversation(
            multiple_prompt, multiple_modalities)

        input_ids = result["input_ids"]
        attention_mask = result["attention_mask"]

        # The first <|reserved_special_token_1|> should map to the first modality, the second to the second modality
        attachment_positions = (input_ids == self.tokenizer.convert_tokens_to_ids(
            "<|reserved_special_token_1|>")).nonzero(as_tuple=True)[0]
        self.assert_attachment_positions(
            attachment_positions, modalities=["image", "image"])

        # Attachments should be unmasked
        self.assertTrue(
            all(attention_mask[attachment_positions].eq(1).tolist()))


if __name__ == '__main__':
    unittest.main()
