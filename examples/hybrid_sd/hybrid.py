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
from diffusers.utils import check_min_version, is_wandb_available


from compression.hybrid_sd.inference_pipeline import HybridInferencePipeline
from compression.utils.seed import set_random_seed
from compression.utils.logger import LoggerWithDepth
from compression.prune_sd.calflops import calculate_flops
from copy import deepcopy
import json

import logging
logger = logging.getLogger(__name__)

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
        "--val_prompts",
        type=str,
        default=None,
        nargs="+",
        help=("A set of prompts evaluated every `--validation_epochs` and logged to `--report_to`."),
    )
    parser.add_argument(
        "--neg_val_prompts",
        type=str,
        default=None,
        nargs="+",
        help=("A set of prompts evaluated every `--validation_epochs` and logged to `--report_to`."),
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        default="./examples/hybrid_sd/prompts.txt",
        help=("A set of prompts evaluated every `--validation_epochs` and logged to `--report_to`."),
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--img_sz", type=int, default=512)
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
    parser.add_argument("--num_images", type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use, cuda:gpu_number or cpu')
    parser.add_argument("--use_dpm_solver", action='store_true', help='use DPMSolverMultistepScheduler')
    parser.add_argument("--use_pndm_solver", action='store_true', help='use DPMSolverMultistepScheduler')
    parser.add_argument("--use_euler_solver", action='store_true', help='use EulerDiscreteScheduler')
    parser.add_argument("--is_lora_checkpoint", action='store_true', help='specify whether to use LoRA finetuning')
    parser.add_argument("--lora_weight_path", type=str, default=None, help='dir path including lora.pt and lora_config.json')    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    check_min_version("0.19.0.dev0")
    set_random_seed(args.seed)
    # defile save path and logger
    args.steps = list(map(int, args.steps.split(',')))
    # step_str = ""
    model_names = []
    for index, model in enumerate(args.model_id):
        model_name = model.split("/")[-1] 
        if "checkpoint" in model_name:
            model_arch = model.split("/")[-5]
            model_config = model.split("/")[-3]
            model_name = f'{model_arch}--{model_config}'
        model_names.append(model_name)
    print(f"Model: {model_names}, Steps: {args.steps}")
    save_path = args.output_dir
    print(f"output_dir={save_path}")
    os.makedirs(save_path, exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(save_path, 'log.txt')),
            logging.StreamHandler()
        ]
    )
    args.logger = logger
    # init pipeline
    pipeline = HybridInferencePipeline(
                                weight_folders = args.model_id,
                                seed = args.seed,
                                device = args.device,
                                args = args)
    pipeline.set_pipe_and_generator()
    pipeline.set_lora_ckpt()

    # calculate flops for each model
    model_info = dict()
    example_inputs = [torch.randn((2, 4, 64, 64), dtype=torch.float16).to(args.device), torch.randn(1, dtype=torch.float16).to(args.device), torch.randn((2, 77, 768), dtype=torch.float16).to(args.device)]
    for unet_id, unet in enumerate(pipeline.pipe.unets):
        unet_copy = deepcopy(unet)
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
            'flops': flops * args.steps[unet_id],
            'total_macs': macs * args.steps[unet_id]
        }
        total_macs = macs * args.steps[unet_id]
        logger.info(f'Model{model_name}, #Params: {params/1e6:2f} M, MACs: {macs/1e9:.2f} G, Total Macs: {total_macs/1e12:.2f} T')

        del unet_copy

    # get prompt and inference
    if args.val_prompts is not None and args.neg_val_prompts is not None:
        val_prompts = args.val_prompts
        neg_val_prompts = args.neg_val_prompts
        assert len(val_prompts) == len(neg_val_prompts)
    elif args.prompts_file is not None:
        with open(args.prompts_file, "r") as file:
            lines = file.readlines()
            val_prompts = lines[0::2]
            neg_val_prompts = lines[1::2] 
            print(val_prompts, len(val_prompts))
            print(neg_val_prompts, len(neg_val_prompts))
            assert len(val_prompts) == len(neg_val_prompts)
    else:
        raise NotImplementedError("lack val prompts!")

    time_list = []
    for index in range(len(val_prompts)):
        prompt = val_prompts[index]
        neg_prompt = neg_val_prompts[index]
        prompt_path =  os.path.join(save_path, f"prompt_{index}")
        os.makedirs(prompt_path, exist_ok=True)
        for i in range(args.num_images):
            logger.info(f"Generate prompt {index} / image {i}")
            image_path = os.path.join(prompt_path, f"image_{i}")
            os.makedirs(image_path, exist_ok=True)

            t0 = time.perf_counter()
            img = pipeline.generate(
                                prompt = prompt,
                                negative_prompt = neg_prompt if neg_prompt != "" else None,
                                img_sz = args.img_sz,
                                guidance_scale = args.guidance_scale,
                                num_images_per_prompt = args.num_images_per_prompt,
                                save_path = image_path if args.save_middle else None,
                                )[0]
            img.save(os.path.join(prompt_path, f"{i}.png"))
            latency = time.perf_counter() - t0
            logger.info(f"{latency:.2f} sec elapsed")
            time_list.append(latency / args.num_images_per_prompt)

    avg_latency = sum(time_list) / len(time_list)
    logger.info(f"Avrage latency: {avg_latency:.2f} sec elapsed")
    pipeline.clear()

    model_info_path = os.path.join(save_path, f"model_info.json")
    model_info['latency'] = latency
    with open(model_info_path, 'w') as f:
        json.dump(model_info, f)
    
