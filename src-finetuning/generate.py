import sys
sys.path.append('../src/')

import os
import torch
from sentencepiece import SentencePieceProcessor
from model import *
import torch.nn.functional as F

tokenizer_path = '../tokenizer.model'
checkpoint_path = '../models/lora_story_teller_110M.pt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = SentencePieceProcessor(model_file=tokenizer_path)

# load model
def remove_unwanted_prefix_from_state_dict(state_dict, unwanted_prefix):
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    return state_dict

def load_model(checkpoint_path, device, unwanted_prefix='_orig_mod'):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['model_args'] if isinstance(checkpoint['model_args'], ModelArgs) else ModelArgs(**checkpoint['model_args'])
    model = Transformer(config)
    if checkpoint.get('lora_finetune'):
        apply_lora(
            model,
            targets=checkpoint['lora_targets'],
            rank=checkpoint['lora_rank'],
            dropout=checkpoint['lora_dropout'],
            alpha=checkpoint['lora_alpha']
        )
    print(f"Number of parameters: {sum([p.nelement() for p in model.parameters()])}")
    state_dict = checkpoint['model']
    state_dict = remove_unwanted_prefix_from_state_dict(state_dict=state_dict, unwanted_prefix=unwanted_prefix)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(device)
    return model, checkpoint

instruct_model, ckpt = load_model(
    checkpoint_path=checkpoint_path,
    device=device,
    unwanted_prefix='',
)


def generate_paragraph(
    model,
    prompt,
    max_new_tokens=400,
    temperature=0.1,
    top_k=10
):
    tokenized_prompt = [tokenizer.bos_id()] + tokenizer.encode(prompt)
    tokenized_prompt = (torch.tensor(tokenized_prompt, dtype=torch.long, device=device)[None, ...])

    paragraph = []
    context_tokens = tokenized_prompt
    for _ in range(max_new_tokens):
        context_tokens = context_tokens[:, -min(model.params.max_seq_len, context_tokens.size(1)):]
        output = model(context_tokens)
        logits = output[:, -1, :]
        logits = logits / temperature
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float('Inf')
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        context_tokens = torch.cat((context_tokens, next_token), dim=1)
        paragraph.append(next_token.item())
        if next_token.item() == tokenizer.eos_id() or tokenizer.decode(paragraph[-3:]) == 'The end.':
            break
    return context_tokens, paragraph, tokenizer.decode(paragraph)

prompt = 'Write a story. In the story, try to use the verb "climb", the noun "ring" and the adjective "messy". Possible story:'

_, tokens, paragraph = generate_paragraph(
    model=instruct_model,
    prompt=prompt,
    max_new_tokens=400,
    temperature=0.5,
    top_k=10
)
print(paragraph)
