import sys
sys.path.append('../src/')
import math
import os
import glob
import random
from functools import partial
import numpy as np
import torch
import torch.nn.functional as F
from contextlib import nullcontext
import matplotlib.pyplot as plt
from sentencepiece import SentencePieceProcessor
from data_loader import *
from utils import *
from model import *

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
device_type = 'cuda' if 'cuda' in device else 'cpu'

DATA_CACHE_DIR = '../instruct_dataset_minimal/'
out_dir = '../build/models/'
os.makedirs(out_dir, exist_ok=True)
# https://huggingface.co/karpathy/tinyllamas/tree/main
pretrained_model_path = '../res/asset/models/karpathy/tinyllamas/stories15M.pt'
tokenizer = SentencePieceProcessor('../res/asset/tokenizer.model')
vocab_size = tokenizer.vocab_size()


# training
# mixed precision settings

is_p100 = torch.cuda.get_device_properties(device).name == "Tesla P100-PCIE-16GB"
if is_p100:
    print("using Tesla P100 GPU")
    dtype = 'float16'  # Use float16 for mixed precision on P100
    # Do not enable TF32 settings as they are not supported by P100
else:
    dtype = 'bfloat16'  # Use bfloat16 for mixed precision on other GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Set the appropriate PyTorch data type based on the dtype variable
ptdtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}.get(dtype, torch.float16)

ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))


# model
checkpoint = torch.load(pretrained_model_path, map_location=device)
model_args = ModelArgs(**checkpoint['model_args'])
model_args.max_seq_len = 350
model = Transformer(model_args)

state_dict = checkpoint['model']
model.load_state_dict(state_dict, strict=False)

# freeze

for p in model.parameters():
    p.requires_grad = False

print(f'Number of parameters: {sum(p.nelement() for p in model.parameters())}')

# add lora

lora_config = {
    'rank': 2,
    'dropout': 0.1,
    'alpha': 1.0,
    'targets': ['wk', 'wq', 'wo', 'wv']
}

apply_lora(
    model,
    verbose=False,
    **lora_config
)
#tie_lora_weights(model.output, model.tok_embeddings)

print(f'Number of parameters: {sum(p.nelement() for p in model.parameters())}')

model.to(device);

# data

max_seq_len = model_args.max_seq_len
print(max_seq_len)

batch_size = 64

wanted_batch_size = 4 * 128
gradient_accumulation_steps = wanted_batch_size // batch_size

print(f'Wanted batch_size: {wanted_batch_size}, gradient accumulation steps: {gradient_accumulation_steps}, batch_size: {batch_size}')

iter_batches = partial(
    iter_batch_func,
    device=device,
    batch_size=batch_size,
    max_seq_len=max_seq_len,
    data_cache_dir=DATA_CACHE_DIR
)


# optimizer

learning_rate = 5e-4
optimizer = get_optimizer(
    model=model,
    device_type='cuda',
    learning_rate=learning_rate,  # max learning rate
    weight_decay = 1e-1,
    beta1 = 0.9,
    beta2 = 0.95,
)

# training loop

max_iters = 5000
eval_iters = 100
best_val_loss = 1e9
grad_clip = 1

iter_num = 0

train_batch_iter = iter_batches(split='train')
X, Y = next(train_batch_iter)

while True:
    lr = get_lr(iter_num, max_iters=max_iters)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    if iter_num % eval_iters == 0 :
        losses = estimate_loss(
            model=model,
            iter_batches=iter_batches,
            eval_iters=eval_iters,
            ctx=ctx
        )
        print(f"step {iter_num}: lr {lr}, train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
            if iter_num > 0:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    model_args=model_args,
                    iter_num=iter_num,
                    out_dir=out_dir,
                    lora_config=lora_config
                )
                _, paragraph = generate_paragraph(
                    model,
                    prompt='Write a story. In the story, try to use the verb "eat", the noun "cat" and the adjective "sad". The story has the following features: the story should contain at least one dialogue. Possible story:',
                    tokenizer=tokenizer,
                    device=device,
                    max_new_tokens=300
                )
                print(paragraph)

    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits = model(X)
            loss = compute_loss(logits, Y)
            loss = loss / gradient_accumulation_steps
        X, Y = next(train_batch_iter)
        scaler.scale(loss).backward()

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)


    iter_num += 1
    if iter_num > max_iters:
        break
