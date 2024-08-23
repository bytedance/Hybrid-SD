#%%
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import pandas as pd
from matplotlib.font_manager import FontProperties
import json
import pickle
import seaborn as sns
from sklearn import metrics

WORKSPACE = Path(__file__).parent.parent.parent
PATH_RESULT = WORKSPACE / 'results'
# DATA_ROOT='/mnt/bn/bytenn-yg2/datasets'
DATA_ROOT='/opt/tiger/bytenn_diffusion_tools/datasets'
MODEL_LARGE = 'CompVis--stable-diffusion-v1-4'
# MODEL_LARGE = 'SG161222--Realistic_Vision_V5.1_noVAE'
MODEL_SMALL = 'nota-ai--bk-sdm-tiny'
RESULT_DIR = PATH_RESULT / 'HybridSD_dpm_guidance7'
PATH_TARGET = PATH_RESULT / 'HybridSD_dpm_guidance7/{MODEL_LARGE}-{MODEL_SMALL}-'
print("WORKSPACE=", WORKSPACE)

model2name = {
    'CompVis--stable-diffusion-v1-4': 'SD-v1.4',
    'runwayml--stable-diffusion-v1-5': 'SD-v1.5',
    'nota-ai--bk-sdm-tiny': 'BK-SDM-Tiny',
    'nota-ai--bk-sdm-small': 'BK-SDM-Small',
    'tea_sd14_tiny_a19_b21': 'OursTiny-224',  
}
        
step_list=["25,0", "15,10", "10,15", "5,20", "0,25"]
results = []
# for MODEL_LARGE in ['CompVis--stable-diffusion-v1-4', 'runwayml--stable-diffusion-v1-5']:
for MODEL_LARGE in ['CompVis--stable-diffusion-v1-4']:
    for MODEL_SMALL in ['nota-ai--bk-sdm-tiny', 'nota-ai--bk-sdm-small']:
        for step in step_list:
            # if 'small' in MODEL_SMALL and step == "25,0":
                # continue
            k = step.split(',')[0]
            output_dir = RESULT_DIR / f'{MODEL_LARGE}-{MODEL_SMALL}-{step}'
            print(output_dir)
            model_info_path = output_dir / 'model_info.json'
            # if model_info_path.exists():
                # with open(model_info_path, 'r') as f:
                    # model_info = json.load(f)
                    
            try:
                if (output_dir/'im256_clip.txt').exists():
                    with open(output_dir/'im256_clip.txt', 'r') as f:
                        clip = float(f.readlines()[0].split(' ')[-1])
                else:
                    clip = 0
                with open(output_dir/'im256_fid.txt', 'r') as f:
                    fid = float(f.readlines()[0].split(' ')[-1])
                with open(output_dir/'im256_is.txt', 'r') as f:
                    is_score = float(f.readlines()[0].split(' ')[-1])
            except:
                continue

            results.append([model2name[MODEL_LARGE], model2name[MODEL_SMALL], k, fid, is_score, clip])

columns = ['large_model', 'small_model', 'k', 'FID', 'IS', 'CLIP']
data = []

for result in results:
    data_item = {}
    for i, c in enumerate(columns):
        data_item[c] = result[i]
    data.append(data_item)
    
df = pd.DataFrame(data).round({'FID': 2, 'CLIP': 4, 'IS': 2})
# display(df)

def plot_metric(df, metric):
    sns.set_style("whitegrid")
    plt.figure(figsize=[6, 5])
    ax = sns.lineplot(data=df, x="k", y=metric, style="small_model", markers=True, markersize=10)
    ax.legend_.set_title(None)
    # plt.plot()
    plt.setp(ax.get_legend().get_texts(), fontsize=18) # for legend text
    # plt.setp(ax.get_legend().get_title(), fontsize='25') # for legend title
    font = FontProperties(fname='TimesNewRoman.ttf', size=15)
    # ax.set_title(metric, fontproperties=font, pad=15)
    plt.savefig(WORKSPACE / f'notebooks/figures/diff_steps_{metric}.pdf', dpi=150)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('Split Step k', fontsize=22)
    plt.ylabel(metric, fontsize=20)
    
plot_metric(df, "FID")
plot_metric(df, "CLIP")
plot_metric(df, "IS")
# %%

# sns.lineplot(data=df, x="k", y="CLIP")
# sns.lineplot(data=df, x="k", y="IS")
# sns.lineplot(data=df, x="k", y="IS", style="model_small")
# ax = sns.lineplot(data=df, x="k", y="CLIP", style="small_model", markers=True)
# plt.savefig(WORKSPACE / 'notebooks/figures/diff_steps_clip.pdf', dpi=150)

# ax = sns.lineplot(data=df, x="k", y="IS", style="small_model", markers=True)
# plt.plot()
# plt.savefig(WORKSPACE / 'notebooks/figures/diff_steps_is.pdf', markers=True, dpi=150)
# df.
# %%
