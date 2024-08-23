#%%
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import pandas as pd
from matplotlib.font_manager import FontProperties

WORKSPACE = Path(__file__).parent.parent.parent
PATH_RESULT = WORKSPACE / 'results'
DATA_ROOT='/mnt/bn/bytenn-yg2/datasets'
# DATA_ROOT='/opt/tiger/bytenn_diffusion_tools/datasets'
MODEL_LARGE = 'CompVis--stable-diffusion-v1-4'
# MODEL_LARGE = 'SG161222--Realistic_Vision_V5.1_noVAE'
MODEL_SMALL = 'nota-ai--bk-sdm-small'
PATH_TARGET = PATH_RESULT / 'HybridSD_dpm_guidance7/{MODEL_LARGE}-{MODEL_SMALL}-'
print("WORKSPACE=", WORKSPACE)

model2name = {
    'CompVis--stable-diffusion-v1-4': 'SD-v1.4',
    'nota-ai--bk-sdm-tiny': 'BK-SDM-Tiny',
    'nota-ai--bk-sdm-small': 'BK-SDM-Small',
    'tea_sd14_tiny_a19_b21': 'OursTiny-224',  
}

def insert_every_n(lst, item, n=1):
    for i in range(n, len(lst) + n, n):
        lst.insert(i, item)
    return lst

def grid_plot(imgs_path, n_row, prompts, modelnames, save_path=None):
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
        title = img_path.stem.split('_')[-1]
        # 检查是否为每行的第一个子图
        if column_id == 0:  # rowspan.start为1表示新的一行开始
            # 在最左侧添加文本标注，注意调整文本位置使其出现在子图外侧
            model_name = modelnames[row_id]
            font = FontProperties(fname='TimesNewRoman.ttf', size=30)
            ax.text(-0.2, 0.5, f'{model_name}', transform=ax.transAxes,
                    fontproperties=font, va='center', ha='right')

        if i < n_col:
            font = FontProperties(fname='TimesNewRoman.ttf', size=40)
            title = prompts[i]
            # print(title)
            split_title = title.split(' ')
            insert_every_n(split_title, '\n', 3)
            title = (' ').join(split_title)
            # for i, text in enumerate(split_title):
            font = FontProperties(fname='TimesNewRoman.ttf', size=28)
            ax.set_title(f'{title}', fontproperties=font, pad=12, color="lightgray")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=100)
        print(f"save fig to {save_path}")
    else:
        plt.show()

step_list=["25,0", "10,15", "0,25"]
df_prompts = pd.read_csv(f'{DATA_ROOT}/mscoco_val2014_30k/metadata.csv')
# sub_df = df_prompts.head(10)
img_names = [164, 661, 730, 2139, 2529, 3724, 3742, 3926, 6896]

filter_list = []
for name in img_names:
    name = str(name).rjust(12, '0') 
    filter_list.append(f'COCO_val2014_{name}.jpg')
print(filter_list)
sub_df = df_prompts.query("file_name == @filter_list")
print(sub_df.columns)
img_paths = []
model_names = []

large_model_name = model2name[MODEL_LARGE]
for model_id, model_small in enumerate(['nota-ai--bk-sdm-small']):
    small_model_name = model2name[model_small]
    for i, step in enumerate(step_list):
        path = PATH_RESULT / f'HybridSD_dpm_guidance7/{MODEL_LARGE}-{model_small}-{step}'
        if i == 0 and model_id > 0:
            continue

        k = int(step.split(',')[0])
        if k == 25:
            model_name = large_model_name
        elif k == 0:
            model_name = small_model_name
        else:
            postfix = small_model_name.split('-')[-1]
            model_name = f'{large_model_name} + {postfix}\n(k={k})'
        model_names.append(model_name)
        for row_id, row in sub_df.iterrows():
            # steps = step.split(',')
            # steps_str = ('_').join(steps)
            path = PATH_RESULT / f'HybridSD_dpm_guidance7/{MODEL_LARGE}-{model_small}-{step}'
            filename = row['file_name']
            img_path = path / f'im512/{filename}'
            img_paths.append(img_path)
prompts = list(sub_df.text.unique())
# display(sub_df)
# print(df_prompts.columns)
save_path = WORKSPACE / 'notebooks/figures/hybrid_sdv1-4_small.pdf'
grid_plot(img_paths, len(step_list), prompts, model_names, save_path)


# %%
