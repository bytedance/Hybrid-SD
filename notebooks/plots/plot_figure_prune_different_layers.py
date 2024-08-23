# %%
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

def grid_plot(imgs_path, img_name='img512/img_1.png', n_row=2, save_path=None):
    n_row = 2
    n_col = len(imgs_path) // n_row

    width, height = 512, 512
    dpi = 100
    figsize = (width / float(dpi) * n_col, height / float(dpi) * n_row) 
    _, axs = plt.subplots(n_row, n_col, dpi=dpi, figsize=figsize, sharex=True, sharey=True)
    axs = axs.flatten()
    for img_dir, ax in zip(imgs_path, axs):
        title = img_dir.name
        params = ''
        with open(img_dir / 'model_info.txt') as f:
            params = f.readlines()[0]
        params = params.split(':')[1].strip()
        img_path = img_dir / img_name
        im_data = plt.imread(img_path)
        ax.imshow(im_data)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'{title}\n({params})', fontsize=30)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
        print(f"save fig to {save_path}")
    else:
        plt.show()

result_dir = Path('/mnt/bn/bytenn-yg2/ycq/workspace/bytenn_diffusion_tools/results/NaivePrune/bk-sdm-small') 
#%%
imgs_path = [result_dir / 'origin']
for i in range(1, 13):
    imgs_path.append(result_dir / f'prune_resnet_ratio0.5/prune_resnet_{i}')
# imgs_path.append(result_dir / 'prune_resnet_ratio0.5/prune_resnet_4_5_7_11')
for img_id in range(4):
    img_name = f'im512/img_{img_id}.png'
    grid_plot(imgs_path, img_name, n_row=2)

# %%
imgs_path = [result_dir / 'origin']
for i in range(1, 10):
    imgs_path.append(result_dir / f'prune_selfatt_heads4/prune_selfatt_{i}')
for img_id in range(4):
    img_name = f'im512/img_{img_id}.png'
    grid_plot(imgs_path, img_name, n_row=2)

# %%
imgs_path = [result_dir / 'origin']
for i in range(1, 10):
    imgs_path.append(result_dir / f'prune_crossatt_heads4/prune_crossatt_{i}')
for img_id in range(4):
    img_name = f'im512/img_{img_id}.png'
    grid_plot(imgs_path, img_name, n_row=2)

# %%
imgs_path = [result_dir / 'origin']
for i in range(1, 10):
    imgs_path.append(result_dir / f'prune_crossatt_heads2/prune_crossatt_{i}')
for img_id in range(4):
    img_name = f'im512/img_{img_id}.png'
    grid_plot(imgs_path, img_name, n_row=2)
# %%
