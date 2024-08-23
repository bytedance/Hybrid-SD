# %%
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib
from pathlib import Path
from PIL import Image
from networkx import dfs_edges
import pickle
import pandas as pd
import seaborn as sns
from pathlib import Path
import numpy as np

WORKSPACE = Path(__file__).parent.parent.parent
PATH_RESULT = WORKSPACE / 'results'
MODEL_ID='bk-sdm-tiny'
print("WORKSPACE=", WORKSPACE)


def grid_plot(imgs_path, img_name='img512/img_1.png', n_row=2, save_path=None):
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

        #  取消图像的黑边
        for spine in ['top', 'right', 'left', 'bottom']:
            ax.spines[spine].set_visible(False)

        ax.imshow(im_data)
        ax.set_xticks([])
        ax.set_yticks([])
        
        font = FontProperties(fname='TimesNewRoman.ttf', weight='extra bold', size=46)
        if title == 'origin':
            title = 'Baseline'
        
        title = title.replace("prune_resnet_", "Resnet-")
        title = title.replace("prune_selfatt_", "SelfAttn-")
        title = title.replace("prune_crossatt_", "CrossAttn-")

        ax.set_title(f'{title}', fontproperties=font, pad=15)

    plt.subplots_adjust(bottom=0.8)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
        print(f"save fig to {save_path}")
    else:
        plt.show()

def plot_prune_modules(save_path):
    result_dir = PATH_RESULT / f'NaivePrune/{MODEL_ID}'
    imgs_path = [result_dir / 'origin']
    layer_list = [1, 2, 4, 10, 11]
    for i in layer_list: 
        imgs_path.append(result_dir / f'prune_resnet_ratio0.5/prune_resnet_{i}')

    layer_list = [1, 2, 4, 5, 8, 9]
    for i in layer_list:
        imgs_path.append(result_dir / f'prune_selfatt_heads4/prune_selfatt_{i}')
        
    layer_list = [1, 2, 4, 5, 8, 9]
    for i in layer_list:
        imgs_path.append(result_dir / f'prune_crossatt_heads4/prune_crossatt_{i}')
        
    img_id = 2
    img_name = f'im512/img_{img_id}.png'
    grid_plot(imgs_path, img_name, n_row=3, save_path=save_path)

# save_path = Path(__file__).parent / f'figures/naive_prune.pdf'
# plot_prune_modules(save_path)

def get_layer_id(base_arch, k):
    layer_id = int(k.split('_')[-1])
    if 'resnet' in k:
        layer_type = 'ResNet'
    if 'selfatt' in k:
        layer_type = 'SelfAttention'
    if 'crossatt' in k:
        layer_type = 'CrossAttention'

    if base_arch == 'bk-sdm-small':
        if layer_type == 'ResNet':
            if layer_id <= 4:
                layer_id += (layer_id - 1) * 2
            elif layer_id in (5, 6, 7, 8):
                layer_id += 6
            else:
                layer_id += 6 + (layer_id - 8) * 2
        if layer_type == 'SelfAttention':
            if layer_id <= 3:
                layer_id = layer_id * 3 - 1
            else:
                layer_id = 12 + (layer_id - 3) * 3 - 1
        if layer_type == 'CrossAttention':
            if layer_id <= 3:
                layer_id *= 3
            else:
                layer_id = 12 + (layer_id - 3) * 3
    
    if base_arch == 'bk-sdm-tiny':
        if layer_type == 'ResNet':
            layer_id = (layer_id - 1) * 3 + 1
        if layer_type == 'SelfAttention':
            layer_id = layer_id * 3 - 1
        if layer_type == 'CrossAttention':
            layer_id = layer_id * 3
    return layer_type, layer_id

def plot_score(score_path, save_path=None):
    with open(score_path, 'rb') as f:
        scores_dict = pickle.load(f)
    plt.figure(figsize=(10, 3), dpi=200)

    data = []
    resnet_scores = []
    selfattn_scores = []
    crossattn_scores = []
    for k, score in scores_dict.items():
        
        layer_type, layer_id = get_layer_id(MODEL_ID, k)
        # if 'resnet' in k:
        #     layer_type = 'ResNet'
        #     get_layer_id(base_arch, k)
        #     if layer_id <= 4:
        #         layer_id += (layer_id - 1) * 2
        #     elif layer_id in (5, 6, 7, 8):
        #         layer_id += 6
        #     else:
        #         layer_id += 6 + (layer_id - 8) * 2
                
        #     resnet_scores.append(score)
        # if 'selfatt' in k:
        #     layer_type = 'SelfAttention'
        #     if layer_id <= 3:
        #         layer_id = layer_id * 3 - 1
        #     else:
        #         layer_id = 12 + (layer_id - 3) * 3 - 1
        #     selfattn_scores.append(score)
        # if 'crossatt' in k:
        #     layer_type = 'CrossAttention'
        #     if layer_id <= 3:
        #         layer_id *= 3
        #     else:
        #         layer_id = 12 + (layer_id - 3) * 3
        #     crossattn_scores.append(score)

        data.append(
            {   'layer_id': layer_id,
                'name': k,
                'score': score,
                'layer_type': layer_type
             }
        )
    df = pd.DataFrame(data)  
    df.sort_values(by=['layer_id'], inplace=True)      
    # display(df)
    avg_resnet = np.array(resnet_scores[1:]).mean()
    avg_crossattn = np.array(crossattn_scores).mean()
    avg_selfattn = np.array(selfattn_scores[1:]).mean()
    
    plt.figure(figsize=(5, 4), dpi=150)
    markers = {"ResNet": "o", "SelfAttention": "s", "CrossAttention": "X"}
    ax = sns.scatterplot(data=df, x="layer_id", y="score", hue="layer_type", style="layer_type", markers=markers)

    font = FontProperties(fname='TimesNewRoman.ttf', size=12)
    ax.legend(loc='upper center', ncols=3, bbox_to_anchor=(0.5, 1.15), prop=font, labelspacing=0.1, columnspacing=0.1)
    plt.xticks(rotation=90, fontproperties=font)
    plt.yticks(fontproperties=font)
    plt.xlabel('Layer ID', fontproperties=font)
    plt.ylabel('Scores', fontproperties=font)
    # plt.axhline(avg_resnet, color='r')
    # plt.axhline(avg_selfattn, color='y')
    # plt.axhline(avg_crossattn, color='b')
    if save_path is not None:
        plt.savefig(save_path)
        print(f"save fig to {save_path}")
    else:
        plt.show()
    
# save_path = Path(__file__).parent / f'figures/scores.png'
save_path=None
# plot_score(PATH_RESULT / f'NaivePrune/{MODEL_ID}/score.pkl', save_path)
plot_score(PATH_RESULT / f'NaivePrune/{MODEL_ID}/prune_oneshot/score.pkl', save_path)
# plot_score(PATH_RESULT / f'NaivePrune/{MODEL_ID}/score.pkl', save_path)
    

# %%
