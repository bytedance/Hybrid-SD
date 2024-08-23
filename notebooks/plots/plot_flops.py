# %%
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

result_dir = Path('/mnt/bn/ycq-lq/workspace/bytenn_diffusion_tools/results/bk-sdm-small') 
csv_file = result_dir / f'baseline/model_info.csv'
df = pd.read_csv(csv_file)
total_params, total_macs = 0, 0

resnet_conv1 = []
resnet_conv2 = []
self_attention = []
cross_attention = []
others = []

for row_id, row in df.iterrows():
    if 'norm' in row['name']: continue
    total_params += row.params
    total_macs += row.macs
    if 'conv1' in row['name']:
        resnet_conv1.append({'name': row['name'], 'params': row.params, 'macs': row.macs})
    elif 'conv2' in row['name']:
        resnet_conv2.append({'name': row['name'], 'params': row.params, 'macs': row.macs})
    elif 'attn1' in row['name'] and row['name'].split('.')[-1] in ('to_q', 'to_k', 'to_v', '0'):
        self_attention.append({'name': row['name'], 'params': row.params, 'macs': row.macs})
    elif 'attn2' in row['name']:
        cross_attention.append({'name': row['name'], 'params': row.params, 'macs': row.macs})
    else:
        if row.params > 0:
            params = row['params'] / 1e6
            print(row['name'], f'{params:.2f} M')
        others.append({'name': row['name'], 'params': row.params, 'macs': row.macs})

print(total_params)
print(resnet_conv1)
print(self_attention)
print(cross_attention)
print(others)

# %%
resnet_conv1_params = sum([item['params'] for item in resnet_conv1])
resnet_conv2_params = sum([item['params'] for item in resnet_conv1])
self_attention_params = sum([item['params'] for item in self_attention])
cross_attention_params = sum([item['params'] for item in cross_attention])
other_params = total_params - resnet_conv1_params - self_attention_params - cross_attention_params - resnet_conv2_params

y = np.array([resnet_conv1_params, self_attention_params, resnet_conv2_params,  cross_attention_params, other_params])

def make_autopct_params(values):
    def my_autopct(pct):
        total = sum(values)
        val = total * pct / 100 / 1e6
        return f'{pct:.2f}%\n({val:.2f} M)'
    return my_autopct
    
plt.pie(y,
        labels=['ResnetConv1', 'SelfAttn', 'ResnetConv2', 'CrossAttn', 'Others'],
        colors=["#d5695d", "#5d8ca8", "#a564c9", "orange", "#65a479"],
        autopct=make_autopct_params(y))
plt.title("#Params for BK-SDM-small")
plt.savefig(result_dir / 'params.png')
plt.show()
# %%
resnet_conv1_macs = sum([item['macs'] for item in resnet_conv1])
resnet_conv2_macs = sum([item['macs'] for item in resnet_conv1])
self_attention_macs = sum([item['macs'] for item in self_attention])
cross_attention_macs = sum([item['macs'] for item in cross_attention])
other_macs = total_macs - resnet_conv1_macs - self_attention_macs - cross_attention_macs - resnet_conv2_macs

y = np.array([resnet_conv1_macs, self_attention_macs, resnet_conv2_macs, cross_attention_macs, other_macs])

def make_autopct_macs(values):
    def my_autopct(pct):
        total = sum(values)
        val = total * pct / 100 / 1e9
        return f'{pct:.2f}%\n({val:.2f} G)'
    return my_autopct

plt.pie(y,
        labels=['ResnetConv1', 'SelfAttn', 'ResnetConv2', 'CrossAttn', 'Others'],
        colors=["#d5695d", "#5d8ca8", "#a564c9", "orange", "#65a479"],
        autopct=make_autopct_macs(y))
plt.title("#Macs for BK-SDM-small")
plt.savefig(result_dir / 'macs.png')
plt.show()
# %%
