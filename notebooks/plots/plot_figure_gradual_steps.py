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
print("WORKSPACE=", WORKSPACE)

WORKSPACE = Path(__file__).parent.parent.parent
PATH_RESULT = WORKSPACE / 'results'
DATA_ROOT='/mnt/bn/bytenn-lq2/datasets'
# DATA_ROOT='/opt/tiger/bytenn_diffusion_tools/datasets'
MODEL_LARGE = 'CompVis--stable-diffusion-v1-4'
MODEL_LARGE = 'SG161222--Realistic_Vision_V5.1_noVAE'
MODEL_SMALL = 'segmind--small-sd'
RESULT_DIR = PATH_RESULT / 'HybridSD_v100_inference'
PATH_TARGET = PATH_RESULT / 'HybridSD_v100_inference/{MODEL_LARGE}-{MODEL_SMALL}-'
print("WORKSPACE=", WORKSPACE)

def grid_plot(imgs_path, modelnames, steps, n_row=2, save_path=None):
    n_col = len(imgs_path) // n_row

    width, height = 512, 512
    dpi = 100
    figsize = (width / float(dpi) * n_col, height / float(dpi) * n_row) 
    _, axs = plt.subplots(n_row, n_col, dpi=dpi, figsize=figsize, sharex=True, sharey=True)
    axs = axs.flatten()
    for i, (img_path, ax) in enumerate(zip(imgs_path, axs)):
        row_id = i // n_col
        column_id = i % n_col

        # print(img_dir)
        # print(img_path.parent.parent.name)
        title = (',').join(img_path.parent.parent.name.split('_')[-2:])
        im_data = plt.imread(img_path)
        ax.imshow(im_data)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ['top', 'right', 'left', 'bottom']:
            ax.spines[spine].set_visible(False)
        
        if column_id == 0:  # rowspan.start为1表示新的一行开始
            # 在最左侧添加文本标注，注意调整文本位置使其出现在子图外侧
            model_name = modelnames[row_id]
            ax.text(-0.2, 0.5, f'{model_name}', transform=ax.transAxes,
                    fontsize=20, va='center', ha='right')
        
        font = FontProperties(fname='TimesNewRoman.ttf', size=30)
        # ax.set_title(f'{title}', fontproperties=font, pad=15)
        if i < n_col:
            font = FontProperties(fname='TimesNewRoman.ttf', size=30)
            title = steps[i]
            ax.set_title(f'{title}', fontproperties=font, pad=15)
    plt.tight_layout()
    # plt.subplots_adjust(hspace=1, wspace=0)
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        print(f"save fig to {save_path}")
    else:
        plt.show()

step_list=["25,0", "15,10", "10,15", "5,20", "2,23", "0,25"]
img_paths =[]
prompt_id=2
image_id=0
# for prompt_id in range(1):
    # for image_id in range(1):
print(f'prompt_id={prompt_id}, image_id={image_id}')
# plot_steps = []
# for step in step_list:
prompt_ids=[]
for prompt_id, image_id in zip([0,2,5], [1,0,3]):
    # k = int(step.split(',')[0])
    # middle_imgs_path = output_dir / f'prompt_{prompt_id}/image_{image_id}'
    # print(middle_imgs_path)
    # img_name = ('_').join(list(middle_imgs_path.glob('*.png'))[0].name.split('_')[:-1])
    # middle_imgs = []
    # for i in range(0, 25, 4):
    prompt_ids.append((prompt_id, image_id))
    for step in step_list:
        output_dir = RESULT_DIR / f'{MODEL_LARGE}-{MODEL_SMALL}-{step}'
        img_path = output_dir / f'prompt_{prompt_id}/{image_id}.png'
        img_paths.append(img_path)
    print(len(img_paths))
# model_names=['Realistic_Vision_V5.1', 'Realistic-V5.1 + segmind-small (k=5)', 'Segmind-Small']
grid_plot(img_paths, prompt_ids, step_list, n_row=len(prompt_ids), save_path=WORKSPACE/ f'notebooks/figures/realistic.png')
# %%
