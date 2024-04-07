"""
Generate text samples from a trained language model.
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
load_model_from = 'resume'  # 'resume' for saved checkpoints or a GPT-2 model variant (e.g., 'gpt2-xl')
output_dir = 'output'  # Ignored if load_model_from is not 'resume'
start_prompt = "\n"  # Prompt for generation or a file name prefixed with "FILE:"
num_samples_to_generate = 10  # Number of samples to generate
max_tokens_to_generate = 500  # Maximum number of tokens per generated sample
sampling_temperature = 0.8  # Sampling temperature: lower = more deterministic, higher = more random
top_k_tokens = 200  # Top-k sampling: only the top_k most likely tokens are considered
random_seed = 1337  # Random seed for reproducibility
device_type = 'cuda'  # Device to use: 'cpu', 'cuda', 'cuda:0', etc.
tensor_dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'  # Precision: 'float32', 'bfloat16', 'float16'
use_compile = False  # Use PyTorch 2.0's compile feature to optimize the model (optional)
exec(open('configurator.py').read())  # Overrides from command line or configuration file
# -----------------------------------------------------------------------------

# Set random seeds for reproducibility
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on matmul operations
torch.backends.cudnn.allow_tf32 = True  # Allow TF32 for cuDNN

# Set up device and dtype context for model inference
selected_device = 'cuda' if 'cuda' in device_type else 'cpu'
precision_dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[tensor_dtype]
inference_context = nullcontext() if selected_device == 'cpu' else torch.amp.autocast(device_type=selected_device, dtype=precision_dtype)

# Load the model
if load_model_from == 'resume':
    # Load from a checkpoint
    checkpoint_path = os.path.join(output_dir, 'checkpoint.pt')
    checkpoint_data = torch.load(checkpoint_path, map_location=device_type)
    model_config = GPTConfig(**checkpoint_data['model_args'])
    model_instance = GPT(model_config)
    state_dict = checkpoint_data['model']

    # Handle any unwanted prefixes in the saved state dict keys
    unwanted_prefix = '_orig_mod.'
    for key, value in list(state_dict.items()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix):]] = state_dict.pop(key)

    # Load the model state
    model_instance.load_state_dict(state_dict)

elif load_model_from.startswith('gpt2'):
    # Load a pre-trained GPT-2 model variant
    model_instance = GPT.from_pretrained(load_model_from, dict(dropout=0.0))

# Move the model to the selected device and switch to evaluation mode
model_instance.eval()
model_instance.to(device_type)
if use_compile:
    model_instance = torch.compile(model_instance)  # Optional PyTorch 2.0 optimization

# Check for meta data (if available) to handle custom encoding/decoding
load_meta_data = False
if load_model_from == 'resume' and 'config' in checkpoint_data and 'dataset' in checkpoint_data['config']:
    meta_file_path = os.path.join('data', checkpoint_data['config']['dataset'], 'meta.pkl')
    load_meta_data = os.path.exists(meta_file_path)

if load_meta_data:
    print(f"Loading meta data from {meta_file_path}...")
    with open(meta_file_path, 'rb') as f:
        meta_data = pickle.load(f)
    stoi_mapping, itos_mapping = meta_data['stoi'], meta_data['itos']
    encode_fn = lambda text: [stoi_mapping[char] for char in text]
    decode_fn = lambda indices: ''.join([itos_mapping[index] for index in indices])
else:
    # Default to GPT-2 encodings if no meta file is found
    print("No meta.pkl found, using default GPT-2 encodings...")
    encoding_tool = tiktoken.get_encoding("gpt2")
    encode_fn = lambda text: encoding_tool.encode(text, allowed_special={"<|endoftext|>"})
    decode_fn = lambda indices: encoding_tool.decode(indices)

# Encode the start of the prompt
if start_prompt.startswith('FILE:'):
    with open(start_prompt[5:], 'r', encoding='utf-8') as file:
        start_prompt = file.read()

start_prompt_ids = encode_fn(start_prompt)
input_tensor = torch.tensor(start_prompt_ids, dtype=torch.long, device=device_type)[None, ...]

# Perform the text generation process
with torch.no_grad():
    with inference_context:
        for i in range(num_samples_to_generate):
            output_tensor = model_instance.generate(input_tensor, max_tokens_to_generate, temperature=sampling_temperature, top_k=top_k_tokens)
            print(decode_fn(output_tensor[0].tolist()))
            print('---------------')
