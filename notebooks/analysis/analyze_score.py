from pathlib import Path
import numpy as np 
from compression.prune_sd.prune_utils import calculate_score
import pickle

result_dir = Path('/mnt/bn/bytenn-yg2/ycq/workspace/bytenn_diffusion_tools/results/NaivePrune/bk-sdm-tiny') 
result_dir = Path('/mnt/bn/bytenn-yg2/liuhj/hybrid_sd/bytenn_diffusion_tools/results/debug')
latent_path = [result_dir / 'origin'/ 'latents.npy']
keys = []
for i in range(1, 10):
    latent_path.append(result_dir / f'prune_resnet_ratio0.5/prune_resnet_{i}/latents.npy')

for i in range(1, 10):
    latent_path.append(result_dir / f'prune_selfatt_heads4/prune_selfatt_{i}/latents.npy')

for i in range(1, 10):
    latent_path.append(result_dir / f'prune_crossatt_heads4/prune_crossatt_{i}/latents.npy')

eval_latents = []

for path in latent_path:
    keys.append(path.parent.name)
    eval_latents.append(np.load(path))

keys = keys[1:]
scores = calculate_score(eval_latents)

score_dict = {}
for k, score in zip(keys, scores):
    score_dict[k] = score
print(score_dict)

with open(result_dir / 'score.pkl', 'wb') as f:
    pickle.dump(score_dict, f)