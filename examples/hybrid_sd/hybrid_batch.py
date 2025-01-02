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
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel,DPMSolverMultistepScheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from compression.hybrid_sd.inference_pipeline import HybridInferencePipeline,  hybrid_load_and_set_lora_ckpt, HybridLCMInferencePipeline, HybridSDXLInferencePipeline

from compression.utils.misc import get_file_list_from_csv, change_img_size
from compression.utils.seed import set_random_seed
from compression.utils.logger import LoggerWithDepth
import json
from copy import deepcopy
from compression.prune_sd.calflops import calculate_flops

if is_wandb_available():
    import wandb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--model_id", 
            type=str, 
            default=[
            "SG161222/Realistic_Vision_V4.0",
            "segmind/small-sd",
            "segmind/tiny-sd"],
            nargs="+",
            help=("Path to pretrained model or model identifier from huggingface.co/models."),
    )  
    parser.add_argument(
        "--steps",
        type=str,
        default="20,5",
        help=("Path to pretrained model or model identifier from huggingface.co/models."),
    )
    parser.add_argument(
        "--pretrained_teacher_model",
        type=str,
        default="pretrained_models/CompVis--stable-diffusion-v1-4",
        help="The path to specific teacher model.",
    )
    parser.add_argument("--data_list", type=str, default="./data/mscoco_val2014_30k/metadata.csv")
    parser.add_argument("--img_sz", type=int, default=512) 
    parser.add_argument("--img_resz", type=int, default=256)
    parser.add_argument("--batch_sz", type=int, default=1)
    parser.add_argument("--num_images", type=int, default=1)
    parser.add_argument("--guidance_scale", type=float, default=7.0, help="guidance_scale.")
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
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use, cuda:gpu_number or cpu')
    parser.add_argument("--use_dpm_solver", action='store_true', help='use DPMSolverMultistepScheduler')
    parser.add_argument("--use_pndm_solver", action='store_true', help='use DPMSolverMultistepScheduler')
    parser.add_argument("--is_lora_checkpoint", action='store_true', help='specify whether to use LoRA finetuning')
    parser.add_argument("--lora_weight_path", type=str, default=None, help='dir path including lora.pt and lora_config.json')    
    parser.add_argument("--model_class", type=str, default="SD", help="use lcm inference for lcm models")
    parser.add_argument("--max_n_files", type=int, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--weight_dtype", type=str, default="fp16")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    check_min_version("0.19.0.dev0")
    set_random_seed(args.seed)
    # defile save path and logger
    args.steps = list(map(int, args.steps.split(',')))
    step_str = ""
    model_names = []
    for index, model in enumerate(args.model_id):
        model_name = model.split("/")[-1] 
        if "checkpoint" in model_name:
            assert args.model_name is not None
            # model_arch = args.base_arch
            # model_config = model.split("/")[-3]
            model_name = args.model_name
        step_str += model_name
        model_names.append(model_name)
    step_str += "_".join(map(str,args.steps))
    # save_path = os.path.join(args.output_dir, step_str)
    save_path = args.output_dir
    print(f"Model: {model_names}, Steps: {args.steps}")
    print(f"Save_path: {save_path}")
    os.makedirs(save_path, exist_ok=True)
    logger = LoggerWithDepth(
        env_name="log", 
        config=args.__dict__,
        root_dir=save_path,
        setup_sublogger=True
    )
    args.logger = logger
    # init pipeline
    if args.model_class == "LCM" or args.model_class == "LCM_edge":
        pipeline = HybridLCMInferencePipeline(
                                weight_folders = args.model_id,
                                seed = args.seed,
                                device = args.device,
                                args = args)
    elif args.model_class == "SD":
        pipeline = HybridInferencePipeline(
                                    weight_folders = args.model_id,
                                    seed = args.seed,
                                    device = args.device,
                                    args = args)
    elif args.model_class == "SDXL":
        pipeline = HybridSDXLInferencePipeline(
                                    weight_folders = args.model_id,
                                    seed = args.seed,
                                    device = args.device,
                                    args = args)
    pipeline.set_pipe_and_generator()
    pipeline.set_lora_ckpt()
       
    # calculate flops for each model
    
    if args.model_class == "SDXL":
        pass
    else:
        model_info = dict()
        example_inputs = [torch.randn((2, 4, 64, 64), dtype=torch.float16).to(args.device), torch.randn(1, dtype=torch.float16).to(args.device), torch.randn((2, 77, 768), dtype=torch.float16).to(args.device)]
        for unet_id, unet in enumerate(pipeline.pipe.unets):
            unet_copy = deepcopy(unet)
            # unet_copy.train()
            # print(unet_copy)
            flops, macs, params = calculate_flops(
                model=unet_copy, 
                args=example_inputs,
                output_as_string=False,
                output_precision=4,
                print_detailed=False,
                print_results=False)
            model_name = model_names[unet_id]
            model_info[model_name] = {
                'params': params,
                'flops': flops,
                'macs': macs,
                'total_macs': macs * args.steps[unet_id]
            }
            total_macs = macs * args.steps[unet_id]
            print(f'Model{model_name}, #Params: {params/1e6:2f} M, MACs: {macs/1e9:.2f} G, Total Macs: {total_macs/1e12:.2f} T')

            del unet_copy

    file_list = get_file_list_from_csv(args.data_list)
    if args.max_n_files is not None:
        file_list = file_list[:args.max_n_files]
    params_str = pipeline.get_sdm_params()
    neg_prompts = len(file_list[0]) > 2
    
    save_dir_src = os.path.join(save_path, f'im{args.img_sz}') # for model's raw output images
    os.makedirs(save_dir_src, exist_ok=True)
    save_dir_tgt = os.path.join(save_path, f'im{args.img_resz}') # for resized images for ms-coco benchmark
    os.makedirs(save_dir_tgt, exist_ok=True)       

    
    t0 = time.perf_counter()
    time_list = []
    for batch_start in range(0, len(file_list), args.batch_sz):
        batch_end = batch_start + args.batch_sz
        img_names = [file_info[0] for file_info in file_list[batch_start: batch_end]]
        val_prompts = [file_info[1] for file_info in file_list[batch_start: batch_end]]
        neg_prompts = [file_info[2] for file_info in file_list[batch_start: batch_end]] if neg_prompts else None       

        if args.model_class == "LCM_edge":
            imgs = pipeline.generate_latents(
                            prompt = val_prompts,
                            negative_prompt = neg_prompts,
                            img_sz = args.img_sz,
                            guidance_scale = args.guidance_scale,
                            num_images_per_prompt = args.num_images_per_prompt,
                            output_type = "latent",
                            save_path = None,
                            )
            for i, (img, img_name, val_prompt) in enumerate(zip(imgs, img_names, val_prompts)):
                torch.save(img, os.path.join(save_dir_src, img_name.replace('.jpg', '.pt')))
                #img.save(os.path.join(save_dir_src, img_name))
                logger.log(f"{batch_start + i}/{len(file_list)} | {img_name} {val_prompt}")
            logger.log(f"---{params_str}")
        else:
            imgs = pipeline.generate(
                                prompt = val_prompts,
                                negative_prompt = neg_prompts,
                                #n_steps = args.steps,
                                img_sz = args.img_sz,
                                guidance_scale = args.guidance_scale,
                                num_images_per_prompt = args.num_images_per_prompt,
                                save_path = None,
                                )
            for i, (img, img_name, val_prompt) in enumerate(zip(imgs, img_names, val_prompts)):
                img.save(os.path.join(save_dir_src, img_name))
                img.close()
                logger.log(f"{batch_start + i}/{len(file_list)} | {img_name} {val_prompt}")
            logger.log(f"---{params_str}")

    pipeline.clear()
    
    
    change_img_size(save_dir_src, save_dir_tgt, args.img_resz)
    latency = time.perf_counter() - t0
    logger.log(f"{latency:.2f} sec elapsed")

    avg_latency = latency / (len(file_list)) / args.num_images_per_prompt
    print(model_info)
    model_info_path = os.path.join(save_path, f"model_info.json")
    model_info['latency'] = avg_latency
    with open(model_info_path, 'w') as f:
        json.dump(model_info, f)