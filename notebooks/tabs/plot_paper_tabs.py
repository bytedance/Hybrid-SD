# %%
import pandas as pd
import numpy as np
import json
from pathlib import Path

# results = [
    # ['Stable Diffusion v1.4(our-reproduce)', 12.22, ],
    # ['BK-SDM-Tiny(our-reproduce)', 16.85, ],
    # ['Stable Diffusion v1.4', 13.05, 36.76, 0.2958, 10],
    # ['BK-SDM-Small', 15.76, 33.79, 0.2878, 10],
    # ['BK-SDM-Tiny', 17.12, 30.09, 0.2653, 10],
    # ['Realistic_Vision_V5.1', 17.40, 38.02, 0.3058, 10],
# ]

# results = [
#     ['SD-v1.4 (our-reproduce)', 12.22, 37.63, 0.30, 16.93],
#     ['BK-SDM-Tiny (our-reproduce)', 16.85, 30.37, 0.27, 10.25],
#     ['SD-v1.4 + Tiny(k=20)', 12.84, 37.96, 0.30, 15.59],
#     ['SD-v1.4 + Tiny(k=15)', 13.55, 37.49, 0.30, 14.26],
#     ['SD-v1.4 + Tiny(k=10)', 14.59, 35.70, 0.29, 12.92],
#     ['SD-v1.4 + Tiny(k=5)', 15.92, 32.66, 0.28, 11.58]
# ]
results = []
data = []

step_list = ["25,0", "0,25", "10,15", "5,20"]
# results/HybridSD/y0_25
LARGE_MODEL = 'CompVis--stable-diffusion-v1-4'
# LARGE_MODEL = 'SG161222--Realistic_Vision_V5.1_noVAE'
# SMALL_MODEL = 'nota-ai--bk-sdm-small'
model2name = {
    'CompVis--stable-diffusion-v1-4': 'SD-v1.4',
    'runwayml--stable-diffusion-v1-5': 'SD-v1.5',
    'SG161222--Realistic_Vision_V5.1_noVAE': 'RV-v5.1',
    'segmind--tiny-sd': 'segmind-tiny',
    'segmind--small-sd': 'segmind-small',
    'nota-ai--bk-sdm-tiny': 'BK-SDM-Tiny',
    'nota-ai--bk-sdm-small': 'BK-SDM-Small',
    'tea_sd14_tiny_a19_b21': '14Tiny224',  
    # 'tea_sd15_tiny_a19_b21': '15Tiny224',
    'tea_sd15_tiny_a9_b17': '15Tiny266',  
}

RESULT_ROOT = Path(__file__).parent.parent.parent / 'results'
RESULT_DIR = RESULT_ROOT / 'HybridSD_dpm_guidance7'

large_models = ['CompVis--stable-diffusion-v1-4']
small_models = ['nota-ai--bk-sdm-small', 'nota-ai--bk-sdm-tiny', 'tea_sd14_tiny_a19_b21', 'tea_sd15_tiny_a9_b17']
for large_model_id, large_model in enumerate(large_models):
    for small_model_id, small_model in enumerate(small_models):
        large_model_name = model2name[large_model]
        print(small_model)
        small_model_name = model2name[small_model]
        for step in step_list:
            k = int(step.split(',')[0])
            if k == 25:
                model_type = large_model_name
            elif k == 0:
                model_type = small_model_name
            else:
                postfix = small_model_name.split('-')[-1]
                model_type = f'Hybrid {large_model_name} + {postfix} (k={k})'

            if small_model_id > 0 and k == 25:
                continue
            
            output_dir = RESULT_DIR / f'{large_model}-{small_model}-{step}'
            try:
                print("output_dir=", output_dir.name)
                model_info_path = output_dir / 'model_info.json'
                if model_info_path.exists():
                    with open(model_info_path, 'r') as f:
                        model_info = json.load(f)
                    total_macs = model_info[large_model]['total_macs'] + model_info[small_model]['total_macs']
                    total_flops = model_info[large_model]['flops'] * k + (25 - k) * model_info[small_model]['flops']
                    latency = model_info['latency']
                    if k == 25:
                        params = model_info[large_model]['params']
                    else:
                        params = model_info[small_model]['params']
                else:
                    total_macs = 0
                    latency = 0

                
                if (output_dir/'im256_clip.txt').exists():
                    with open(output_dir/'im256_clip.txt', 'r') as f:
                        clip = float(f.readlines()[0].split(' ')[-1])
                else:
                    clip = 0
                with open(output_dir/'im256_fid.txt', 'r') as f:
                    fid = float(f.readlines()[0].split(' ')[-1])
                with open(output_dir/'im256_is.txt', 'r') as f:
                    is_score = float(f.readlines()[0].split(' ')[-1])

                print(fid, is_score, clip)
            except:
                continue

            # if step == '25,0':
            #     step = 'Stable Diffusion v1.4'
            # if step == '0,25':
                # step = SMALL_MODEL

            results.append([model_type, fid, is_score, clip, f'{params/1e6:.2f}', f'{total_flops/1e12:.2f}'])

columns = ['Model', 'FID', 'IS', 'CLIP', '#Params (M)', 'FLOPs(T)']
for result in results:
    data_item = {}
    for i, c in enumerate(columns):
        data_item[c] = result[i]
    data.append(data_item)

df = pd.DataFrame(data).round({'FID': 2, 'CLIP': 4, 'IS': 2})
display(df)
# df = df.set_index('Model')
# df = df.transpose()
# print(df.to_latex(
#     label='tab:coco',
#     column_format='llllll'))

print(df.to_latex(
    index=False,
    label='tab:coco',
    column_format='llllll'))


# %%
