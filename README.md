
## install

```
pip install -r requirements.txt
```






```bash
# single GPU Training
python train.py --batch_size=32 --compile=False

# Multiple GPU Training
$torchrun --standalone --nproc_per_node=4 train.py

# Multiple GPU Multiple Nodes
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
```


```bash
# Sampling from the trained model
python sample.py --out_dir=out-shakespeare-char
```



### To do
 - Training done!
 - Try training with different batch size
 - Chck weights of the training model, if any understandable pattern!
 - Train small LLMs to be used for inference on Raspberry Pi 