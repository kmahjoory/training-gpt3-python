"""
Training script for both single GPU (debug mode) and Distributed Data Parallel (DDP) training.

For single GPU, example usage:
$ python train.py --batch_size=32 --compile=False

For DDP on 4 GPUs on a single node:
$ torchrun --standalone --nproc_per_node=4 train.py

For DDP on 4 GPUs across 2 nodes:
- On the master node (IP 123.456.123.456):
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- On the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
"""