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
from datetime import datetime
import numpy as np
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler,DPMSolverMultistepScheduler
from diffusers.utils import check_min_version, is_wandb_available

from compression.prune_sd.inference_pipeline import InferencePipeline
from compression.utils.misc import get_file_list_from_csv, change_img_size
from compression.utils.seed import set_random_seed
from compression.utils.logger import LoggerWithDepth
from compression.prune_sd.pruner import UnetPruner
from compression.prune_sd.models.unet_2d_condition import UNet2DConditionModel

if is_wandb_available():
    import wandb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="nota-ai/bk-sdm-small")  
    parser.add_argument("--unet_path", type=str, default=None)  
    parser.add_argument("--steps", type=int, default=25, help=("Path to pretrained model or model identifier from huggingface.co/models."))
    parser.add_argument("--data_list", type=str, default="./data/mscoco_val2014_30k/metadata.csv")
    parser.add_argument("--img_sz", type=int, default=512) 
    parser.add_argument("--img_resz", type=int, default=256)
    parser.add_argument("--batch_sz", type=int, default=1)
    parser.add_argument("--num_images", type=int, default=1)
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="guidance_scale.")
    parser.add_argument("--num_images_per_prompt", type=int, default=1, help="num_images_per_prompt.")
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--save_middle", action="store_true", help="Whether or not to save middle step."
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use, cuda:gpu_number or cpu')
    parser.add_argument("--use_dpm_solver", action='store_true', help='use DPMSolverMultistepScheduler')
    parser.add_argument("--use_ddpm_solver", action='store_true', help='use DDPMScheduler')
    parser.add_argument("--is_lora_checkpoint", action='store_true', help='specify whether to use LoRA finetuning')
    parser.add_argument("--lora_weight_path", type=str, default=None, help='dir path including lora.pt and lora_config.json')    
    parser.add_argument("--prune_resnet_layers", type=str, default=None)
    parser.add_argument("--keep_resnet_ratio", type=str)
    parser.add_argument("--prune_selfattn_layers", type=str, default=None)
    parser.add_argument("--keep_selfattn_heads_ratio", type=str)
    parser.add_argument("--prune_crossattn_layers", type=str, default=None)
    parser.add_argument("--keep_crossattn_heads_ratio", type=str)
    parser.add_argument("--max_n_files", type=int, default=None)
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    check_min_version("0.19.0.dev0")
    set_random_seed(args.seed)
    # defile save path and logger
    # save_path = os.path.join(args.output_dir, args.model_id.split("/")[-1]+f"_{args.steps}")
    save_path = args.output_dir
    os.makedirs(save_path, exist_ok=True)
    logger = LoggerWithDepth(
        env_name="log", 
        config=args.__dict__,
        root_dir=save_path,
        setup_sublogger=True
    )
    args.logger = logger
    # init pipeline
    pipeline = InferencePipeline(
                                weight_folder = args.model_id,
                                seed = args.seed,
                                device = args.device,
                                args = args)
    pipeline.set_pipe_and_generator()
    # pipeline.set_lora_ckpt()
    if args.unet_path is not None: # use a separate trained unet for generation        
        # from diffusers import UNet2DConditionModel 
        unet = UNet2DConditionModel.from_pretrained(args.unet_path, subfolder='unet')
        pipeline.pipe.unet = unet.half().to(args.device)
        logger.log(f"** load unet from {args.unet_path}")        

    # example_inputs = [torch.randn((2, 4, 64, 64), dtype=torch.float16).to(args.device), torch.randn(1, dtype=torch.float16).to(args.device), torch.randn((2, 77, 768), dtype=torch.float16).to(args.device)]
    # pruner = UnetPruner(args, pipeline.pipe.unet, example_inputs=example_inputs)
    # pipeline.pipe.unet = unet

    file_list = get_file_list_from_csv(args.data_list)
    if args.max_n_files is not None:
        file_list = file_list[:args.max_n_files]
    print(f'Length of file_list = {len(file_list)}')
    
    params_str = pipeline.get_sdm_params()
    neg_prompts = len(file_list[0]) > 2
    
    save_dir_src = os.path.join(save_path, f'im{args.img_sz}') # for model's raw output images
    os.makedirs(save_dir_src, exist_ok=True)
    save_dir_tgt = os.path.join(save_path, f'im{args.img_resz}') # for resized images for ms-coco benchmark
    os.makedirs(save_dir_tgt, exist_ok=True)       

    
    t0 = time.perf_counter()
    for batch_start in range(0, len(file_list), args.batch_sz):
        batch_end = batch_start + args.batch_sz
        img_names = [file_info[0] for file_info in file_list[batch_start: batch_end]]
        val_prompts = [file_info[1] for file_info in file_list[batch_start: batch_end]]
        neg_prompts = [file_info[2] for file_info in file_list[batch_start: batch_end]] if neg_prompts else None       
        imgs = pipeline.generate(
                                prompt = val_prompts,
                                negative_prompt = neg_prompts,
                                n_steps = args.steps,
                                img_sz = args.img_sz,
                                guidance_scale = args.guidance_scale,
                                num_images_per_prompt = args.num_images_per_prompt
                                )
        for i, (img, img_name, val_prompt) in enumerate(zip(imgs, img_names, val_prompts)):
            img.save(os.path.join(save_dir_src, img_name))
            img.close()
            logger.log(f"{batch_start + i}/{len(file_list)} | {img_name} {val_prompt}")
        logger.log(f"---{params_str}")
    pipeline.clear()
    change_img_size(save_dir_src, save_dir_tgt, args.img_resz)
    logger.log(f"{(time.perf_counter()-t0):.2f} sec elapsed")
