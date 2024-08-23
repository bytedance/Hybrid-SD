
import torch
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import subprocess
import threading
from threading import Thread, current_thread, BoundedSemaphore, active_count, Lock
from compression.analysis import plot_utils as plot_utils


### plot weights
with open('outputs/txt2img-samples/weights.pkl', 'rb') as f:
    weights_dict = pickle.load(f)

# path = "outputs/txt2img-samples/images/boxplot/model_weights.png"
# plot_utils.layer_boxplot(path, weights_dict, title="StableDiffusion v1.5", figsize=(60, 16))

# path = "outputs/txt2img-samples/images/boxplot/linear_weights.png"
# linear_weights = {k: v for k, v in weights_dict.items() if v['class_name'] == 'Linear'}
# plot_utils.layer_boxplot(path, linear_weights, title="Linear Layers", figsize=(40, 16))

# path = "outputs/txt2img-samples/images/boxplot/conv_weights.png"
# conv_weights = {k: v for k, v in weights_dict.items() if v['class_name'] == 'Conv2d'}
# plot_utils.layer_boxplot(path, conv_weights, title="Conv Layers", figsize=(40, 16))

# path = "outputs/txt2img-samples/images/boxplot/to_k.png"
# conv_weights = {k: v for k, v in weights_dict.items() if "to_k" in k}
# plot_utils.layer_boxplot(path, conv_weights, title="K Layers", figsize=(30, 15))

# path = "outputs/txt2img-samples/images/boxplot/to_q.png"
# conv_weights = {k: v for k, v in weights_dict.items() if "to_q" in k}
# plot_utils.layer_boxplot(path, conv_weights, title="Q Layers", figsize=(30, 15))

plot_layers = ["input_blocks.7.0.in_layers.2", "output_blocks.5.0.in_layers.2", "output_blocks.6.0.in_layers.2"]

for layer_id, (name, value) in enumerate(weights_dict.items()):
#     if layer_id == 0: continue
#     # data = value.numpy()
    if name in plot_layers:
        path = f'outputs/txt2img-samples/images/distplot/layer_{layer_id}_{name}.png'
        plot_utils.plot_dist_histogram(path, value["data"], name, figsize=(20, 8))


# prefix_list = ["input_blocks.1.1.transformer_blocks", "input_blocks.2.1.transformer_blocks"]



### plot activations
# act_names = []
# for block_name in ["input_blocks.2.1", "middle_block.1", "output_blocks.11.1"]:
#     for layer_name in ["attn2.to_k", "attn2.to_v", "attn2.to_q", "ff.net.2"]:
#         act_name = f"{block_name}.transformer_blocks.0.{layer_name}"
#         act_names.append(act_name)

# output_dir = Path(f'outputs/sd1.5/plots/boxplots')
# output_dir.mkdir(exist_ok=True)
# for act_name in act_names:
#     tensor_dict = dict()
#     for i in range(1, 51):
#         act_dir = Path(f'outputs/sd1.5/activations/step_{i}')
#         file = act_dir / f"{act_name}.pth"
#         tensor = torch.load(file)
#         # print(f"load tensor {file}", tensor.shape)
#         tensor_dict[f'step_{i}/{act_name}'] = tensor
#     print(f"plotting {act_name}")
#     layer_boxplot(output_dir/f'{act_name}_boxplot.png', tensor_dict, f'Box plot of {act_name} across steps')


# for i in [20, 30, 50]:
#     act_dir = Path(f'outputs/sd1.5/activations/step_{i}')
#     output_dir = Path(f'outputs/sd1.5/plots/step_{i}')
#     output_dir.mkdir(exist_ok=True, parents=True)
#     threedist_dir = output_dir / 'threedist'
#     twodist_dir = output_dir / 'twodist'
#     threedist_dir.mkdir(exist_ok=True)
#     twodist_dir.mkdir(exist_ok=True)

#     for file in Path(act_dir).glob('*.pth'):
#         tensor = torch.load(file)
#         layer_name = file.stem
#         print(f"plotting tensor {layer_name}, shape=({tensor.size()})")

#         if tensor.size(0) == 2:
#             e_t_uncond, e_t = tensor.chunk(2)
#             if (output_dir / f'twodist/{layer_name}.png').exists():
#                 continue
#             img_path = output_dir / f'threedist/{layer_name}_uncond.png'
#             if (img_path).exists():
#                 continue
#             plot_3d_dist_histogram(img_path, e_t_uncond, f'step-{i} {layer_name}_uncond', is_act=True)
#             img_path = output_dir / f'twodist/{layer_name}_uncond.png'
#             plot_dist_histogram(img_path, e_t_uncond, f'step-{i} {layer_name}_uncond')
#             img_path = output_dir / f'threedist/{layer_name}_cond.png'
#             plot_3d_dist_histogram(img_path, e_t, f'step-{i} {layer_name}_cond', is_act=True)
#             img_path = output_dir / f'twodist/{layer_name}_cond.png'
#             plot_dist_histogram(img_path, e_t, f'step-{i} {layer_name}_cond')
#         elif len(list(tensor.size())) == 2:
#             img_path = output_dir / f'twodist/{layer_name}.png'
#             if (img_path).exists():
#                 continue
#             plot_dist_histogram(img_path, e_t, f'step-{i} {layer_name}')
#         else:
#             img_path = output_dir / f'threedist/{layer_name}.png'
#             if (img_path).exists():
#                 continue
#             plot_3d_dist_histogram(img_path, tensor, f'step-{i} {layer_name}', is_act=True)
#             img_path = output_dir / f'twodist/{layer_name}.png'
#             plot_dist_histogram(img_path, tensor, f'step-{i} {layer_name}')
            




