# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
import time
import csv
import torch
from compression.prune_sd.inference_pipeline import InferencePipeline
from compression.prune_sd.models.unet_2d_condition import UNet2DConditionModel
from compression.prune_sd.pruner import UnetPruner
import numpy as np
import pickle

# from diffusers import UNet2DConditionModel
from compression.prune_sd.prune_utils import prune_unet
import logging

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="pretrained_models/nota-ai--bk-sdm-small")    
    parser.add_argument("--save_dir", type=str, default="./results/debug",
                        help="$save_dir/{im256, im512} are created for saving 256x256 and 512x512 images")
    parser.add_argument("--unet_path", type=str, default=None)
    parser.add_argument("--data_list", type=str, default="data/mscoco_val2014_30k/metadata.csv")    
    parser.add_argument("--num_images", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use, cuda:gpu_number or cpu')
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--img_sz", type=int, default=512)
    parser.add_argument("--img_resz", type=int, default=256)
    parser.add_argument("--batch_sz", type=int, default=1)
    parser.add_argument("--prune_resnet_layers", type=str, default='1,2,3,4,5,6,7,8,9')
    parser.add_argument("--keep_resnet_ratio", type=float, default=0.5)
    parser.add_argument("--prune_selfattn_layers", type=str, default='1,2,3,4,5,6,7,8,9')
    parser.add_argument("--keep_selfattn_heads", type=int, default=6)
    parser.add_argument("--prune_crossattn_layers", type=str, default='1,2,3,4,5,6,7,8,9')
    parser.add_argument("--keep_crossattn_heads", type=int, default=6)
    parser.add_argument("--prune_mlp_layers", type=str, default=None)
    parser.add_argument("--keep_mlp_ratio", type=float, default=0.5)
    parser.add_argument("--output_type", type=str, default="pil")
    parser.add_argument("--calc_flops", action="store_true")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action='store_true')
    parser.add_argument("--use_dpm_solver", action='store_true')
    parser.add_argument("--use_ddpm_solver", action='store_true')
    parser.add_argument("--score_file", type=str, default='results/score-prune/bk-sdm-small-diff-pruning/ratio-0.5/scores.pkl')
    parser.add_argument("--th", type=float, default=0)
    parser.add_argument("--a", type=int, default=10)
    parser.add_argument("--b", type=int, default=20)
    parser.add_argument("--base_arch", type=str, default='bk-sdm-small')
    parser.add_argument("--ignore_layers", type=str, default=None)

    args = parser.parse_args()
    return args

def get_keep_ratio(layer_id, a=10, b=20):
    if layer_id in list(range(0, a)):
        return 0.25
    elif layer_id in list(range(a, b)):
        return 0.5
    else:
        return 0.75

if __name__ == "__main__":
    args = parse_args()
    save_dir_src = os.path.join(args.save_dir, f'im{args.img_sz}') # for model's raw output images
    os.makedirs(save_dir_src, exist_ok=True)
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(args.save_dir, 'log.txt')),
            logging.StreamHandler()
        ]
    )
    
    with open(args.score_file, 'rb') as f:
        scores = pickle.load(f)

    for k, v in scores.items():
        if 'resnet' in k:
            scores[k] -= args.th

    ignore_layers = []
    if args.ignore_layers is not None:
        ignore_layers = args.ignore_layers.split(',')
        for layer in ignore_layers:
            del scores[layer]
        print(f'Ignore layers = {ignore_layers}')

    scores = dict(sorted(scores.items(), key=lambda item: item[1])) 
    logging.info(f">>> sorted scores: {scores}")
    total_layers = list(scores.keys())

    

    unet = UNet2DConditionModel.from_pretrained(args.model_id, subfolder="unet", torch_dtype=torch.float16).to(args.device)
    module_name_dict = {name: module for name, module in unet.named_modules()}
    
    example_inputs = [torch.randn((2, 4, 64, 64), dtype=torch.float16).to(args.device), torch.randn(1, dtype=torch.float16).to(args.device), torch.randn((2, 77, 768), dtype=torch.float16).to(args.device)]
    unet.eval()
    pipeline = InferencePipeline(weight_folder = args.model_id,
                                seed = args.seed,
                                device = args.device,
                                args = args)

    # modify ratio according to args
    n_resnet_layer = 0
    n_selfattn_layer = 0
    n_crossattn_layer = 0
    if args.base_arch == 'bk-sdm-small':
        keep_resnet_ratio = [0.5] * 12
    else:
        keep_resnet_ratio = [0.5] * 10
    keep_crossattn_heads_ratio = [0.5] * 9
    keep_selfattn_heads_ratio = [0.5] * 9
    n_layers = len(total_layers)
    for layer_id, layer in enumerate(total_layers):
        if layer in ignore_layers:
            ratio = 1.0
        else:
            ratio = get_keep_ratio(layer_id, args.a, args.b)
        logging.info(f'[{layer_id}], {layer}, keep_ratio = {ratio}')
        if 'resnet' in layer:
            n_resnet_layer = int(layer.split('_')[-1]) - 1
            print(n_resnet_layer, layer)
            keep_resnet_ratio[n_resnet_layer] = ratio
        if 'selfatt' in layer:
            keep_selfattn_heads_ratio[n_selfattn_layer] = ratio
            n_selfattn_layer = int(layer.split('_')[-1]) - 1
        if 'crossatt' in layer:
            n_crossattn_layer = int(layer.split('_')[-1]) - 1
            keep_crossattn_heads_ratio[n_crossattn_layer] = ratio
    args.keep_resnet_ratio=keep_resnet_ratio
    args.keep_crossattn_heads_ratio=keep_crossattn_heads_ratio
    args.keep_selfattn_heads_ratio=keep_selfattn_heads_ratio
    pruner = UnetPruner(args, unet, example_inputs=example_inputs) #keep_resnet_ratio=keep_resnet_ratio, keep_crossattn_heads_ratio=keep_crossattn_heads_ratio, keep_selfattn_heads_ratio=keep_selfattn_heads_ratio)
    pipeline.set_pipe_and_generator()   
    pipeline.pipe.unet = pruner.unet
    params_str = pipeline.get_sdm_params()
    
    # t0 = time.perf_counter()
    val_prompts = ["A hotel room in colors of brown and blue.",
                   "a brown and white cat staring off with pretty green eyes.",
                   "A small white dog looking into a camera.",
                   "Small green vase on counter with floral arrangement."]
    imgs = pipeline.generate(prompt = val_prompts,
                            n_steps = args.num_inference_steps,
                            img_sz = args.img_sz,
                            output_type = 'pil')
    for img_id, img in enumerate(imgs):
        img.save(os.path.join(save_dir_src, f'img_{img_id}.png'))

    pipeline.clear()
    # print(f"{(time.perf_counter()-t0):.2f} sec elapsed")

    flops, macs, params = pruner.calc_flops(output_as_string=False)
    if args.base_arch == 'bk-sdm-small':
        base_params, base_macs = 482346884, 435290767360
    elif args.base_arch == 'bk-sdm-tiny':
        base_params, base_macs = 323384964, 409905397760

    ratio_params = (1 - params / base_params) * 100
    ratio_macs = (1 - macs / base_macs) * 100
    logging.info(f'#Params: {params/1e6:.4f} M ({ratio_params:.2f}%), #MACS: {macs/1e9:.4f} G ({ratio_macs:.2f}%)')
    pruner.unet.save_pretrained(os.path.join(args.save_dir, "unet"))
    model_info_file = os.path.join(args.save_dir, 'model_info.txt')
    with open(model_info_file, 'w') as f:
        f.write(f'#Params: {params/1e6:.4f} M ({ratio_params:.2f}%), #MACS: {macs/1e9:.4f} G ({ratio_macs:.2f}%)')
    logging.info(f"Save result to {args.save_dir}")