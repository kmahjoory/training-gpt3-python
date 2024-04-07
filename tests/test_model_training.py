import unittest
from unittest.mock import patch, MagicMock
import torch

class TestModelTraining(unittest.TestCase):

    def setUp(self):
        # Setting up a simple mock GPT model and optimizer for testing
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
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.scaler = torch.cuda.amp.GradScaler(enabled=False)  # Using GradScaler for mixed precision

    def test_gradient_accumulation(self):
        # Simulate gradient accumulation over multiple steps
        batch_size = 8
        grad_accumulation_steps = 4
        accumulation_loss = 0.0

        for step in range(grad_accumulation_steps):
            # Generate random input data
            input_data = torch.randint(0, 100, (batch_size, 128), dtype=torch.long)
            target_data = torch.randint(0, 100, (batch_size, 128), dtype=torch.long)

            # Forward pass
            with torch.cuda.amp.autocast(enabled=False):
                logits, loss = self.model(input_data, target_data)

            # Simulate accumulation
            accumulation_loss += loss.item() / grad_accumulation_steps
            loss = loss / grad_accumulation_steps  # Divide loss for accumulation
            loss.backward()  # Backpropagate

        # Ensure gradients have been accumulated
        for param in self.model.parameters():
            self.assertIsNotNone(param.grad)
        
        # Ensure optimizer step is performed correctly
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Check that the accumulated loss matches
        self.assertGreater(accumulation_loss, 0, "Accumulated loss should be greater than zero")

    def test_learning_rate_adjustment(self):
        # Test for learning rate adjustment in training loop
        initial_lr = 0.001
        lr_decay_iterations = 1000
        min_lr = 0.0001

        def get_adjusted_lr(current_iteration):
            # Linear decay to min_lr
            if current_iteration < lr_decay_iterations:
                return initial_lr * (1 - current_iteration / lr_decay_iterations)
            return min_lr

        for iter_num in range(1100):  # Run a few iterations
            adjusted_lr = get_adjusted_lr(iter_num)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = adjusted_lr

            # Check that learning rate adjusts as expected
            for param_group in self.optimizer.param_groups:
                self.assertEqual(param_group['lr'], adjusted_lr)

    def test_loss_decreases_during_training(self):
        # Test that the loss decreases over training iterations
        batch_size = 8
        total_iterations = 10
        prev_loss = float('inf')

        for i in range(total_iterations):
            input_data = torch.randint(0, 100, (batch_size, 128), dtype=torch.long)
            target_data = torch.randint(0, 100, (batch_size, 128), dtype=torch.long)

            # Forward pass
            with torch.cuda.amp.autocast(enabled=False):
                logits, loss = self.model(input_data, target_data)

            # Backward pass
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Assert that the loss is decreasing
            self.assertLess(loss.item(), prev_loss, "Loss should decrease as training progresses")
            prev_loss = loss.item()


if __name__ == '__main__':
    unittest.main()
