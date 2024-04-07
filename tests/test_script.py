import unittest
from ast import literal_eval
import sys
import torch

class TestConfigAndOverrides(unittest.TestCase):

    def test_config_override_with_file(self):
        # Simulate loading a config file (mocking open and exec)
        test_config_file = 'test_config.py'
        open_mock = unittest.mock.mock_open(read_data="batch_size = 64\nlearning_rate = 0.001\n")
        with unittest.mock.patch('builtins.open', open_mock):
            exec(open(test_config_file).read())

        # Check if variables were set correctly
        self.assertEqual(globals().get('batch_size'), 64)
        self.assertEqual(globals().get('learning_rate'), 0.001)

    def test_config_override_with_cli_arguments(self):
        # Simulate CLI arguments: --batch_size=32 --learning_rate=0.0001
        sys.argv = ['train.py', '--batch_size=32', '--learning_rate=0.0001']
        for argument in sys.argv[1:]:
            key, value = argument.split('=', 1)
            key = key[2:]  # Remove '--'
            if key in globals():
                try:
                    parsed_value = literal_eval(value)
                except (SyntaxError, ValueError):
                    parsed_value = value
                self.assertEqual(type(parsed_value), type(globals().get(key)))  # Type match
                globals()[key] = parsed_value

        # Check if overrides worked
        self.assertEqual(globals().get('batch_size'), 32)
        self.assertEqual(globals().get('learning_rate'), 0.0001)

    def test_invalid_cli_key(self):
        # Simulate invalid CLI argument
        sys.argv = ['train.py', '--invalid_key=123']
        with self.assertRaises(ValueError):
            for argument in sys.argv[1:]:
                key, value = argument.split('=', 1)
                key = key[2:]
                if key not in globals():
                    raise ValueError(f"Unknown configuration key: {key}")


class TestDataLoading(unittest.TestCase):

    def test_batch_loading(self):
        # Simulate a simple dataset for loading batches
        dataset_size = 1000
        block_size = 1024
        batch_size = 12

        # Mock np.memmap with dummy data
        data_mock = torch.randint(0, 50000, (dataset_size,), dtype=torch.long)
        np_mock = unittest.mock.Mock()
        np_mock.return_value = data_mock.numpy()

        with unittest.mock.patch('numpy.memmap', np_mock):
            ix = torch.randint(len(data_mock) - block_size, (batch_size,))
            x = torch.stack([data_mock[i:i + block_size] for i in ix])
            y = torch.stack([data_mock[i + 1:i + 1 + block_size] for i in ix])

        self.assertEqual(x.size(0), batch_size)
        self.assertEqual(y.size(0), batch_size)
        self.assertEqual(x.size(1), block_size)
        self.assertEqual(y.size(1), block_size)


class TestModelInitialization(unittest.TestCase):

    def test_model_initialization_from_scratch(self):
        from model import GPT, GPTConfig

        # Set up a simple GPTConfig
        config = GPTConfig(
            vocab_size=50304,
            block_size=1024,
            n_layer=12,
            n_head=12,
            n_embd=768
        )
        model = GPT(config)

        self.assertIsInstance(model, GPT)
        self.assertEqual(model.config.vocab_size, 50304)
        self.assertEqual(model.config.block_size, 1024)

    def test_model_loading_from_checkpoint(self):
        from model import GPT, GPTConfig
        import os
        # Assume we have a checkpoint path
        checkpoint = {
            'model_args': {'vocab_size': 50304, 'block_size': 1024, 'n_layer': 12, 'n_head': 12, 'n_embd': 768},
            'model': torch.load('test_ckpt.pt')  # Mock a state_dict loading
        }

        gpt_config = GPTConfig(**checkpoint['model_args'])
        model = GPT(gpt_config)
        model.load_state_dict(checkpoint['model'])

        # Check the model configuration
        self.assertEqual(model.config.vocab_size, 50304)
        self.assertEqual(model.config.block_size, 1024)

        # Check that model was loaded properly
        self.assertTrue(model is not None)


if __name__ == '__main__':
    unittest.main()
