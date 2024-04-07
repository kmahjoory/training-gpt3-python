import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class CustomLayerNorm(nn.Module):
    """ Custom LayerNorm with optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, dimension, use_bias):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(dimension))
        self.bias = nn.Parameter(torch.zeros(dimension)) if use_bias else None

    def forward(self, input_tensor):
        return F.layer_norm(input_tensor, self.weights.shape, self.weights, self.bias, 1e-5)

class SelfAttention(nn.Module):
    """ Implements a causal self-attention mechanism """

    def __init__(self, config):
        super().__init__()
        assert config.embedding_dim % config.num_heads == 0, "Embedding dimension must be divisible by the number of heads"
        # Key, Query, Value projections for all heads
        self.attn_projection = nn.Linear(config.embedding_dim, 3 * config.embedding_dim, bias=config.use_bias)
        # Output projection
        self.output_projection = nn.Linear(config.embedding_dim, config.embedding_dim, bias=config.use_bias)
        # Regularization
        self.attention_dropout = nn.Dropout(config.dropout_rate)
        self.residual_dropout = nn.Dropout(config.dropout_rate)
        self.num_heads = config.num_heads
        self.embedding_dim = config.embedding_dim
        self.dropout_rate = config.dropout_rate

        # Check if flash attention is available in PyTorch
        self.flash_attention_available = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash_attention_available:
            print("WARNING: Flash Attention requires PyTorch >= 2.0, using slower attention mechanism")
            # Create causal mask to ensure attention is only applied to previous tokens
            self.register_buffer("causal_mask", torch.tril(torch.ones(config.max_sequence_length, config.max_sequence_length))
                                  .view(1, 1, config.max_sequence_length, config.max_sequence_length))

    def forward(self, x):
        B, T, C = x.size()  # B: batch size, T: sequence length, C: embedding dimension

        # Compute queries, keys, and values for all heads
        q, k, v = self.attn_projection(x).split(self.embedding_dim, dim=2)
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)

        # Apply causal self-attention
        if self.flash_attention_available:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                                                 dropout_p=self.dropout_rate if self.training else 0, is_causal=True)
        else:
            attn_scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            attn_scores = attn_scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.attention_dropout(attn_weights)
            y = attn_weights @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Apply output projection and residual dropout
        y = self.residual_dropout(self.output_projection(y))
        return y

class FeedForward(nn.Module):
    """ Implements a simple feedforward network used in transformers """

    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.embedding_dim, 4 * config.embedding_dim, bias=config.use_bias)
        self.gelu_activation = nn.GELU()
        self.fc2 = nn.Linear(4 * config.embedding_dim, config.embedding_dim, bias=config.use_bias)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu_activation(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    """ Single transformer block with self-attention and feedforward layers """

    def __init__(self, config):
        super().__init__()
        self.layer_norm1 = CustomLayerNorm(config.embedding_dim, use_bias=config.use_bias)
        self.self_attention = SelfAttention(config)
        self.layer_norm2 = CustomLayerNorm(config.embedding_dim, use_bias=config.use_bias)
        self.feed_forward = FeedForward(config)

    def forward(self, x):
        x = x + self.self_attention(self.layer_norm1(x))
        x = x + self.feed_forward(self.layer_norm2(x))
        return x

@dataclass
class ModelConfig:
    """ Configuration for the transformer model """
    max_sequence_length: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size padded for efficiency
    num_layers: int = 12
    num_heads: int = 12
    embedding_dim: int = 768
    dropout_rate: float = 0.1
    use_bias: bool = True  # Bias in linear layers and LayerNorm

class TransformerModel(nn.Module):
    """ Full transformer model with attention and feedforward blocks """

    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.vocab_size is not None, "Vocab size is required"
        assert config.max_sequence_length is not None, "Max sequence length is required"
        self.config = config

        self.transformer = nn.ModuleDict({
            "token_embedding": nn.Embedding(config.vocab_size, config.embedding_dim),
            "position_embedding": nn.Embedding(config.max_sequence_length, config.embedding_dim),
            "dropout": nn.Dropout(config.dropout_rate),
            "layers": nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)]),
            "final_layer_norm": CustomLayerNorm(config.embedding_dim, use_bias=config.use_bias),
        })
        self.output_layer = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)

        # Tie token embeddings and output layer weights
        self.transformer.token_embedding.weight = self.output_layer.weight

        # Initialize all model weights
        self.apply(self.initialize_weights)

        # Custom initialization for projections
        for name, param in self.named_parameters():
            if name.endswith('output_projection.weight'):
                torch.nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * config.num_layers))

    def initialize_weights(self, module):
        """ Initialize model weights """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, target_ids=None):
        device = input_ids.device
        batch_size, seq_length = input_ids.size()
        assert seq_length <= self.config.max_sequence_length, "Sequence length exceeds maximum limit"
        
        # Create position indices and get embeddings
        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device)
        token_embeddings = self.transformer.token_embedding(input_ids)
        position_embeddings = self.transformer.position_embedding(position_ids)
        x = self.transformer.dropout(token_embeddings + position_embeddings)

        # Apply transformer blocks
        for layer in self.transformer.layers:
            x = layer(x)

        x = self.transformer.final_layer_norm(x)

        # Compute logits and loss if target IDs are provided
        if target_ids is not None:
            logits = self.output_layer(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1), ignore_index=-1)
        else:
            logits = self.output_layer(x[:, [-1], :])  # Only predict the last token for inference
            loss = None

        return logits, loss
