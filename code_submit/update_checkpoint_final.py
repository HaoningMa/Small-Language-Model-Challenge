import torch
import json

ckpt_path = 'runs/final-bigram-4k/checkpoint.pt'
ckpt = torch.load(ckpt_path, map_location='cpu')

# 更新 config
ckpt['config']['bigram_lambda'] = 0.08
ckpt['config']['bigram_beta'] = 50.0
ckpt['config']['local_cache_lambda'] = 0.30
ckpt['config']['local_cache_k'] = 2.0

torch.save(ckpt, ckpt_path)
print('Updated checkpoint config:')
print(json.dumps(ckpt['config'], indent=2))
