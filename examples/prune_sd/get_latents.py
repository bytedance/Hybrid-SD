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
# from diffusers import UNet2DConditionModel

from compression.prune_sd.prune_utils import prune_unet
from compression.utils.misc import get_file_list_from_csv
import pickle

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="pretrained_models/nota-ai--bk-sdm-tiny")    
    parser.add_argument("--base_arch", type=str, default="bk-sdm-tiny")    
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
    parser.add_argument("--prune_resnet_layers", type=str, default=None)
    parser.add_argument("--keep_resnet_ratio", type=float, default=0.5)
    parser.add_argument("--prune_selfattn_layers", type=str, default=None)
    parser.add_argument("--keep_selfattn_heads_ratio", type=float, default=0.5)
    parser.add_argument("--prune_crossattn_layers", type=str, default=None)
    parser.add_argument("--keep_crossattn_heads_ratio", type=float, default=0.5)
    parser.add_argument("--output_type", type=str, default="pil")
    parser.add_argument("--calc_flops", action="store_true")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action='store_true')
    parser.add_argument("--use_dpm_solver", action='store_true')
    parser.add_argument("--use_ddpm_solver", action='store_true')
    parser.add_argument("--generate_teacher", action='store_true')

    args = parser.parse_args()
    return args

def create_pruner(args):
    unet = UNet2DConditionModel.from_pretrained(args.model_id, subfolder="unet", torch_dtype=torch.float16).to(args.device)
    example_inputs = [torch.randn((2, 4, 64, 64), dtype=torch.float16).to(args.device), torch.randn(1, dtype=torch.float16).to(args.device), torch.randn((2, 77, 768), dtype=torch.float16).to(args.device)]
    unet.eval()
    pruner = UnetPruner(args, unet, example_inputs=example_inputs)
    return pruner

def prune_one_layer(args, pipeline, prune_type, pruner):
    pipeline.pipe.unet = pruner.unet
    params_str = pipeline.get_sdm_params()
    
    file_list = get_file_list_from_csv(args.data_list)[:50]
    val_prompts = [file_info[1] for file_info in file_list]
    imgs = pipeline.generate(prompt = val_prompts,
                            n_steps = args.num_inference_steps,
                            img_sz = args.img_sz,
                            output_type = 'latent')
    np_latents = imgs.cpu().detach().numpy()

    flops, macs, params = pruner.calc_flops(output_as_string=False)
    print(f'params: {params/1e6} M, macs: {macs/1e9} G, flops: {flops/1e9} G')
    prune_layer_info = {
        'prune_type': prune_type,
        'params': params,
        'macs': macs,
        'flops': flops,
        'np_latents': np_latents
    }
    return prune_layer_info

    # pruner.unet.save_pretrained(os.path.join(args.save_dir, "unet"))
    # model_info_file = os.path.join(args.save_dir, 'model_info.txt')
    # with open(model_info_file, 'w') as f:
    #     f.write(f'params: {params}\nmacs: {macs}\nflops: {flops}')

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    layer_infos = []

    # create pipeline
    pipeline = InferencePipeline(weight_folder = args.model_id,
                                seed = args.seed,
                                device = args.device,
                                args = args)

    pipeline.set_pipe_and_generator() 
     
    # baseline latents
    layer_info = prune_one_layer(args, pipeline, 'baseline', create_pruner(args))
    layer_infos.append(layer_info)

    for ratio in [0.25, 0.5, 0.75]:
        args.keep_resnet_ratio = ratio
        for resnet_layer_id in range(1, 11):
            args.prune_resnet_layers = str(resnet_layer_id)
            layer_info = prune_one_layer(args, pipeline, f'resnet_{resnet_layer_id}_{ratio}', create_pruner(args))
            layer_infos.append(layer_info)

    args.prune_resnet_layers = None
    for ratio in [0.25, 0.5, 0.75]:
        args.keep_selfattn_heads_ratio = ratio
        for selfattn_layer_id in range(1, 10):
            args.prune_selfattn_layers = str(selfattn_layer_id)
            layer_info = prune_one_layer(args, pipeline, f'selfattn_{selfattn_layer_id}_{ratio}', create_pruner(args))
            layer_infos.append(layer_info)

    args.prune_selfattn_layers = None
    for ratio in [0.25, 0.5, 0.75]:
        args.keep_crossattn_heads_ratio = ratio
        for crossattn_layer_id in range(1, 10):
            args.prune_crossattn_layers = str(crossattn_layer_id)
            layer_info = prune_one_layer(args, pipeline, f'crossattn_{crossattn_layer_id}_{ratio}', create_pruner(args))
            layer_infos.append(layer_info)

    with open(f'{args.save_dir}/layer_infos.pickle', 'wb') as f:
        pickle.dump(layer_infos, f)

    pipeline.clear()