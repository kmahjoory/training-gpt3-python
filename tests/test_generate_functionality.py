import unittest
import torch
from unittest.mock import patch

class TestGenerateFunctionality(unittest.TestCase):

    def setUp(self):
        # Create a simple GPT model for testing
        from model import GPT, GPTConfig
        config = GPTConfig(
            vocab_size=100, 
            block_size=128, 
            n_layer=2, 
            n_head=2, 
            n_embd=64, 
            dropout=0.1
        )
        self.model = GPT(config)
        self.model.eval()

    def test_generate_basic(self):
        # Test the basic functionality of the generate method
        start_tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)  # Sample input
        max_new_tokens = 10

        # Generate a sequence of tokens
        with torch.no_grad():
            generated_tokens = self.model.generate(start_tokens, max_new_tokens=max_new_tokens)
        
        # Ensure the generated output has the correct shape
        self.assertEqual(generated_tokens.size(1), start_tokens.size(1) + max_new_tokens)

    def test_generate_with_temperature(self):
        # Test generation with different temperatures
        start_tokens = torch.tensor([[1, 2, 3]], dtype=torch.long)

        # Test temperature=1.0 (normal)
        with torch.no_grad():
            generated_normal_temp = self.model.generate(start_tokens, max_new_tokens=5, temperature=1.0)
        
        # Test temperature=0.5 (more deterministic)
        with torch.no_grad():
            generated_low_temp = self.model.generate(start_tokens, max_new_tokens=5, temperature=0.5)
        
        # Test temperature=2.0 (more random)
        with torch.no_grad():
            generated_high_temp = self.model.generate(start_tokens, max_new_tokens=5, temperature=2.0)

        # Check if the outputs are of correct size
        self.assertEqual(generated_normal_temp.size(1), start_tokens.size(1) + 5)
        self.assertEqual(generated_low_temp.size(1), start_tokens.size(1) + 5)
        self.assertEqual(generated_high_temp.size(1), start_tokens.size(1) + 5)

    def test_generate_with_top_k(self):
        # Test generation with top-k sampling
        start_tokens = torch.tensor([[1, 2, 3]], dtype=torch.long)

        with torch.no_grad():
            generated_top_k_10 = self.model.generate(start_tokens, max_new_tokens=5, top_k=10)
            generated_top_k_50 = self.model.generate(start_tokens, max_new_tokens=5, top_k=50)

        # Check if outputs have correct size
        self.assertEqual(generated_top_k_10.size(1), start_tokens.size(1) + 5)
        self.assertEqual(generated_top_k_50.size(1), start_tokens.size(1) + 5)

    @patch('torch.multinomial')
    def test_generate_sampling_behavior(self, mock_multinomial):
        # Mock multinomial to ensure it behaves deterministically for the test
        mock_multinomial.return_value = torch.tensor([[42]], dtype=torch.long)

        start_tokens = torch.tensor([[1, 2, 3]], dtype=torch.long)
        with torch.no_grad():
            generated_sequence = self.model.generate(start_tokens, max_new_tokens=5)
        
        # Ensure the mocked token (42) is used during generation
        self.assertTrue(42 in generated_sequence, "Mocked token 42 should appear in the generated sequence")


if __name__ == '__main__':
    unittest.main()
