# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023. Nota Inc. All Rights Reserved.
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


from compression.hybrid_sd.inference_pipeline import InferencePipeline, load_and_set_lora_ckpt
from compression.utils.seed import set_random_seed
from compression.utils.logger import LoggerWithDepth

if is_wandb_available():
    import wandb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="nota-ai/bk-sdm-small")  
    parser.add_argument(
        "--steps",
        type=int,
        default=25,
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
    parser.add_argument("--guidance_scale", type=int, default=7.0, help="guidance_scale.")
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
    parser.add_argument("--is_lora_checkpoint", action='store_true', help='specify whether to use LoRA finetuning')
    parser.add_argument("--lora_weight_path", type=str, default=None, help='dir path including lora.pt and lora_config.json')    

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    check_min_version("0.19.0.dev0")
    set_random_seed(args.seed)
    # defile save path and logger
    save_path = os.path.join(args.output_dir, args.model_id.split("/")[-1]+f"_{args.steps}")
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
    pipeline.set_lora_ckpt()
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
            assert len(val_prompts) == len(neg_val_prompts)
    else:
        raise NotImplementedError("lack val prompts!")
    logger.log(f"val_prompts:\n{val_prompts}")
    logger.log(f"neg_val_prompts:\n{neg_val_prompts}")
    for index in range(len(val_prompts)):
        prompt = val_prompts[index]
        neg_prompt = neg_val_prompts[index]
        prompt_path =  os.path.join(save_path, f"prompt_{index}")
        os.makedirs(prompt_path, exist_ok=True)
        for i in range(args.num_images):
            logger.log(f"Generate prompt {index} / image {i}")
            image_path = os.path.join(prompt_path, f"image_{i}")
            os.makedirs(image_path, exist_ok=True)
            t0 = time.perf_counter()
            img = pipeline.generate(
                                prompt = prompt,
                                negative_prompt = neg_prompt if neg_prompt != "" else None,
                                n_steps = args.steps,
                                img_sz = args.img_sz,
                                guidance_scale = args.guidance_scale,
                                num_images_per_prompt = args.num_images_per_prompt,
                                save_path = image_path if args.save_middle else None,
                                )[0]
            img.save(os.path.join(prompt_path, f"{i}.png"))
            logger.log(f"{(time.perf_counter()-t0):.2f} sec elapsed")
    pipeline.clear()
    
