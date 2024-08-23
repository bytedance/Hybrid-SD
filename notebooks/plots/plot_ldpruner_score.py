#%%
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns

def calculate_score(latents_path):
    def get_latent_avg_std(latent):
        avg = np.mean(latent, axis=0)
        std = np.std(latent, axis=0)
        return avg, std
    org_latent = np.load(latents_path[0])    
    org_avg, org_std = get_latent_avg_std(org_latent)
    labels = []
    print(org_avg.shape)
    print(org_std.shape)
    scores = []
    for latent_path in latents_path[1:]:
        mod_latent = np.load(latent_path)
        mod_avg, mod_std = get_latent_avg_std(mod_latent)
        dis_avg = np.linalg.norm(org_avg - mod_avg)
        dis_std = np.linalg.norm(org_std - mod_std)
        score = dis_avg + dis_std
        scores.append(score)
        labels.append(latent_path.parent.name)
    return scores, labels
    
result_dir = Path('/mnt/bn/ycq-lq/workspace/bytenn_diffusion_tools/results/bk-sdm-small') 
latents_path = [result_dir / 'baseline/latents.npy']
for i in range(1, 13):
    latents_path.append(result_dir / f'prune_resnet_ratio0.5/prune_resnet_{i}/latents.npy')
# latents_path.append(result_dir / 'prune_resnet_ratio0.5/prune_resnet_4_5_7_11/latents.npy')
scores, labels = calculate_score(latents_path)
print(scores)
df = pd.DataFrame([{'id': i, 'score': score, 'label': label} for i, (score, label) in enumerate(zip(scores, labels))])
display(df)

plt.figure(figsize=(10, 3), dpi=200)

ax = sns.scatterplot(data=df, x="id", y="score")
for row_id, row in df.iterrows():
    ax.annotate(row.label, (row.id, row.score-2), fontsize=5)
plt.savefig(result_dir / 'prune_resnet_ratio0.5/score.png')


# %%
result_dir = Path('/mnt/bn/ycq-lq/workspace/bytenn_diffusion_tools/results/bk-sdm-small') 
latents_path = [result_dir / 'baseline/latents.npy']
for i in range(1, 10):
    latents_path.append(result_dir / f'prune_selfatt_heads6/prune_selfatt_{i}/latents.npy')
scores, labels = calculate_score(latents_path)
print(scores)
df = pd.DataFrame([{'id': i, 'score': score, 'label': label} for i, (score, label) in enumerate(zip(scores, labels))])
display(df)

plt.figure(figsize=(8, 3), dpi=200)

ax = sns.scatterplot(data=df, x="id", y="score")
for row_id, row in df.iterrows():
    ax.annotate(row.label, (row.id, row.score-2), fontsize=5)
plt.savefig(result_dir / 'prune_selfatt_heads6/prune_selfatt_heads6_score.png')
# %%
result_dir = Path('/mnt/bn/ycq-lq/workspace/bytenn_diffusion_tools/results/bk-sdm-small') 
latents_path = [result_dir / 'baseline/latents.npy']
for i in range(1, 10):
    latents_path.append(result_dir / f'prune_crossatt_heads2/prune_crossatt_{i}/latents.npy')
scores, labels = calculate_score(latents_path)
print(scores)
df = pd.DataFrame([{'id': i, 'score': score, 'label': label} for i, (score, label) in enumerate(zip(scores, labels))])
display(df)

plt.figure(figsize=(8, 3), dpi=200)

ax = sns.scatterplot(data=df, x="id", y="score")
for row_id, row in df.iterrows():
    ax.annotate(row.label, (row.id, row.score-2), fontsize=5)
plt.savefig(result_dir / 'prune_crossatt_heads2/prune_crossatt_heads2_score.png')

# %%
